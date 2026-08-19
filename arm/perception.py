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
    idx: int                  # 0=좌, 1=중, 2=우 (화면 순서 — 프레임마다 바뀔 수 있으니 식별자로 쓰지 말 것)
    center_uv: tuple
    bbox: tuple               # (x1,y1,x2,y2)
    conf: float
    mask: np.ndarray | None = None
    p_cam: np.ndarray | None = None   # 헤드캠 좌표 3D [m]
    base_uv: tuple | None = None      # 컵-테이블 접점 픽셀 (마스크 최하단 중심)


def backproject(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """픽셀 + depth -> 카메라 좌표 3D. K 는 rectified intrinsics 를 써야 한다."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m])


def cup_base_pixel(cup: "Cup") -> tuple:
    """컵-테이블 접점 픽셀 = 마스크 최하단 행의 중심. 마스크가 없으면 bbox 하단 중심."""
    if cup.mask is not None:
        rows = np.flatnonzero(cup.mask.any(axis=1))
        if rows.size:
            v_bot = int(rows[-1])
            cols = np.flatnonzero(cup.mask[v_bot])
            if cols.size:
                return (float(cols.mean()), float(v_bot))
    x1, _, x2, y2 = cup.bbox
    return ((x1 + x2) / 2.0, float(y2))


def ray_plane_intersect(u: float, v: float, K: np.ndarray,
                        plane_point_hc: np.ndarray, plane_normal_hc: np.ndarray,
                        max_range_m: float = 3.0):
    """픽셀 광선과 평면의 교점 (헤드캠 좌표). 뒤쪽/너무 먼 해는 버린다."""
    d = np.linalg.inv(K) @ np.array([u, v, 1.0])
    d = d / np.linalg.norm(d)
    denom = float(d @ plane_normal_hc)
    if abs(denom) < 1e-6:                       # 광선이 평면과 거의 평행
        return None
    t = float(plane_point_hc @ plane_normal_hc) / denom
    if not (0.05 < t < max_range_m):
        return None
    return t * d


def cup_position_on_table(cup: "Cup", T_hc_ab: np.ndarray, table_z: float,
                          K: np.ndarray, cup_center_height_m: float = 0.04):
    """★ depth 센서 없이 컵의 3D 위치를 구한다 — "컵은 테이블 위에 있다"는 사전정보 이용.

    StereoSGBM 은 흰 종이컵처럼 텍스처 없는 물체에서 정확히 그 물체 위에만 구멍이 뚫린다
    (docs/11 기준 유효픽셀 32%). 대신 컵-테이블 접점 픽셀에서 광선을 쏴 알려진 테이블
    평면과 교차시키면 depth 가 아예 필요 없다.

    T_hc_ab : p_hc = T_hc_ab @ p_ab (anchor.T_headcam_to_armbase() 결과)
    table_z : armbase 좌표계에서의 테이블 상면 높이 [m] (config task.table_z)
    반환    : 헤드캠 좌표계의 컵 중심 3D 점, 못 구하면 None
    """
    if T_hc_ab is None:
        return None
    R, t = T_hc_ab[:3, :3], T_hc_ab[:3, 3]
    plane_point_hc = R @ np.array([0.0, 0.0, table_z]) + t
    plane_normal_hc = R @ np.array([0.0, 0.0, 1.0])

    u, v = cup.base_uv if cup.base_uv is not None else cup_base_pixel(cup)
    p_hc = ray_plane_intersect(u, v, K, plane_point_hc, plane_normal_hc)
    if p_hc is None:
        return None
    # 접점 -> 컵 중심 높이만큼 위로 (armbase +z 방향)
    return p_hc + plane_normal_hc * cup_center_height_m


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
            cup.base_uv = cup_base_pixel(cup)     # 테이블 평면 교차용 접점 픽셀
            if depth_m is not None:
                d = robust_depth(depth_m, cu, cv_, self.patch, m)
                if d is not None:
                    cup.p_cam = backproject(cu, cv_, d, K)
            cups.append(cup)
        return cups


@dataclass
class GazeDwell:
    """Midas touch 방지. 같은 컵에 dwell_s 만큼 시선이 머물러야 선택으로 친다.

    ★ 후보를 "화면 좌->우 인덱스"가 아니라 **화면상 위치**로 추적한다 (2026-08-19).
    인덱스는 프레임마다 다시 매겨지므로, 컵 하나가 한 프레임 검출에서 빠지면 나머지
    컵들의 인덱스가 통째로 밀린다. 예전 구현은 그 때 dwell 을 처음부터 다시 시작했고
    (선택이 계속 안 됨), 더 나쁘게는 sticky 한 selected 인덱스를 다음 프레임의
    컵 리스트에 대입할 여지가 있었다 — 다른 컵을 잡는 사고로 이어질 수 있다.

    이제 update() 는 인덱스가 아니라 **선택된 그 순간 프레임의 Cup 객체**를 돌려준다.
    호출자가 인덱스를 다시 조회할 일이 없으니 이 종류의 사고가 구조적으로 불가능하다.
    """

    dwell_s: float = 1.0
    release_s: float = 0.4
    track_radius_px: float = 60.0     # 이 반경 안이면 같은 컵으로 본다

    _cand_uv: tuple | None = None
    _since: float = 0.0
    _last_seen: float = 0.0
    selected: "Cup | None" = None
    progress: float = 0.0
    history: list = field(default_factory=list)

    def _hit(self, cups: list, gaze_uv) -> "Cup | None":
        """시선 픽셀이 올라가 있는 컵. 마스크 우선, 없으면 bbox."""
        if gaze_uv is None:
            return None
        gu, gv = gaze_uv
        for c in cups:
            if c.mask is not None:
                iv, iu = int(round(gv)), int(round(gu))
                if 0 <= iv < c.mask.shape[0] and 0 <= iu < c.mask.shape[1] and c.mask[iv, iu]:
                    return c
            else:
                x1, y1, x2, y2 = c.bbox
                if x1 <= gu <= x2 and y1 <= gv <= y2:
                    return c
        return None

    def update(self, cups: list, gaze_uv, t: float | None = None):
        """-> 선택 확정된 Cup 객체 (이번 프레임 것), 아직이면 None."""
        t = time.time() if t is None else t
        hit = self._hit(cups, gaze_uv)

        if hit is not None:
            same = (self._cand_uv is not None and
                    float(np.hypot(hit.center_uv[0] - self._cand_uv[0],
                                   hit.center_uv[1] - self._cand_uv[1])) <= self.track_radius_px)
            if not same:
                self._since = t
                self.selected = None
            self._cand_uv = hit.center_uv
            self._last_seen = t
        elif self._cand_uv is not None and t - self._last_seen > self.release_s:
            self._cand_uv, self._since, self.progress, self.selected = None, 0.0, 0.0, None

        if self._cand_uv is None or hit is None:
            self.progress = 0.0 if self._cand_uv is None else self.progress
            return None

        self.progress = min(1.0, (t - self._since) / self.dwell_s)
        if self.progress >= 1.0 and self.selected is None:
            self.selected = hit
            self.history.append((t, hit.center_uv))
            log.info("컵 %d 선택 (dwell %.2fs, uv=%s)", hit.idx, t - self._since,
                     tuple(round(x, 1) for x in hit.center_uv))
        return self.selected

    def reset(self):
        self._cand_uv, self._since, self.progress, self.selected = None, 0.0, 0.0, None
