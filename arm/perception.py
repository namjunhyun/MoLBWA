"""시선 -> 컵 선택. YOLOv8-seg 로 컵을 찾고, 시선점이 어느 컵 위에 dwell 하는지 판정한다.

MoLBWA 쪽에서 이미 나오는 값을 그대로 받는다:
  - scene 좌영상 (rectified BGR)
  - 시선점 (u, v)  <- src/gaze_on_scene.py
  - depth map [m]  <- StereoSGBM 또는 RealSense

컵 식별은 화면 x 중심으로 정렬한 좌/중/우 인덱스를 쓴다. 3개 고정 배치에서는
tracker ID 보다 이게 훨씬 안 흔들린다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Cup:
    idx: int                  # 0=좌, 1=중, 2=우
    center_uv: tuple
    bbox: tuple               # (x1,y1,x2,y2)
    conf: float
    mask: np.ndarray | None = None
    p_cam: np.ndarray | None = None   # 헤드캠 좌표 3D [m]


def backproject(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """픽셀 + depth -> 카메라 좌표 3D. K 는 rectified intrinsics 를 써야 한다."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m])


def robust_depth(depth_m: np.ndarray, u: float, v: float, patch: int = 9,
                 mask: np.ndarray | None = None) -> float | None:
    """마스크/패치 안의 유효 depth median. 흰 종이컵은 구멍이 뚫리므로 median 필수."""
    h, w = depth_m.shape[:2]
    r = patch // 2
    u0, v0 = int(round(u)), int(round(v))
    sub = depth_m[max(0, v0 - r):min(h, v0 + r + 1), max(0, u0 - r):min(w, u0 + r + 1)]
    vals = sub[np.isfinite(sub) & (sub > 0.05) & (sub < 3.0)]
    if vals.size < 5 and mask is not None:
        vals = depth_m[mask.astype(bool)]
        vals = vals[np.isfinite(vals) & (vals > 0.05) & (vals < 3.0)]
    if vals.size < 5:
        return None
    return float(np.median(vals))


class CupDetector:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        g = cfg["gaze"]
        self.model = YOLO(g["yolo_model"])
        self.classes = g["yolo_classes"]
        self.conf = g["yolo_conf"]
        self.patch = g["depth_patch_px"]

    def detect(self, frame_bgr: np.ndarray, depth_m: np.ndarray | None, K: np.ndarray) -> list[Cup]:
        res = self.model.predict(frame_bgr, classes=self.classes, conf=self.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        raw = []
        masks = res.masks.data.cpu().numpy() if res.masks is not None else None
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            m = None
            if masks is not None:
                m = masks[i]
                if m.shape != frame_bgr.shape[:2]:
                    import cv2
                    m = cv2.resize(m, (frame_bgr.shape[1], frame_bgr.shape[0]))
                m = m > 0.5
            raw.append(((x1 + x2) / 2, (y1 + y2) / 2, (x1, y1, x2, y2), float(box.conf), m))
        raw.sort(key=lambda r: r[0])          # 좌 -> 우
        cups = []
        for idx, (cu, cv_, bbox, conf, m) in enumerate(raw):
            cup = Cup(idx=idx, center_uv=(cu, cv_), bbox=bbox, conf=conf, mask=m)
            if depth_m is not None:
                d = robust_depth(depth_m, cu, cv_, self.patch, m)
                if d is not None:
                    cup.p_cam = backproject(cu, cv_, d, K)
            cups.append(cup)
        return cups


@dataclass
class GazeDwell:
    """Midas touch 방지. 같은 컵에 dwell_s 만큼 시선이 머물러야 선택으로 친다."""

    dwell_s: float = 1.0
    release_s: float = 0.4

    _cand: int | None = None
    _since: float = 0.0
    _last_seen: float = 0.0
    selected: int | None = None
    progress: float = 0.0
    history: list = field(default_factory=list)

    def update(self, cups: list[Cup], gaze_uv, t: float | None = None) -> int | None:
        t = time.time() if t is None else t
        hit = None
        if gaze_uv is not None:
            gu, gv = gaze_uv
            for c in cups:
                x1, y1, x2, y2 = c.bbox
                if c.mask is not None:
                    iv, iu = int(round(gv)), int(round(gu))
                    if 0 <= iv < c.mask.shape[0] and 0 <= iu < c.mask.shape[1] and c.mask[iv, iu]:
                        hit = c.idx
                        break
                elif x1 <= gu <= x2 and y1 <= gv <= y2:
                    hit = c.idx
                    break

        if hit is not None:
            if hit != self._cand:
                self._cand, self._since = hit, t
            self._last_seen = t
        elif self._cand is not None and t - self._last_seen > self.release_s:
            self._cand, self._since, self.progress = None, 0.0, 0.0

        if self._cand is None:
            self.progress = 0.0
        else:
            self.progress = min(1.0, (t - self._since) / self.dwell_s)
            if self.progress >= 1.0 and self.selected != self._cand:
                self.selected = self._cand
                self.history.append((t, self._cand))
                log.info("컵 %d 선택 (dwell %.2fs)", self._cand, t - self._since)
        return self.selected

    def reset(self):
        self._cand, self._since, self.progress, self.selected = None, 0.0, 0.0, None
