#!/usr/bin/env python3
"""
탑다운 카메라 ↔ 테이블 평면 공통 로직.

`topdown_click`(캘리브레이션)과 `control_panel`(통합 GUI)이 같이 쓴다.
카메라 열기, 호모그래피 로드/적용, 색 기반 물체 검출이 전부 여기 있다.

호모그래피는 "화면 픽셀 ↔ 로봇 베이스 (x, y)" 대응이다. 책상이 평면이고
카메라가 고정이라 3×3 행렬 하나로 끝난다. 깊이도 SLAM도 필요 없다.
"""

import json
import os
import struct
import subprocess
import threading
import time

import cv2
import numpy as np
import yaml


class TableHomography:
    """화면 픽셀 <-> 로봇 베이스 좌표 변환."""

    def __init__(self):
        self.H = None
        self.H_inv = None
        self.reproj_error_mm = None
        self.num_points = None

    @property
    def ready(self):
        return self.H is not None

    def load(self, path):
        """저장된 캘리브레이션을 읽는다. 실패 시 (False, 사유)."""
        if not os.path.exists(path):
            return False, f"파일이 없습니다: {path}"
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            H = np.array(data["H"], dtype=float)
            if H.shape != (3, 3):
                return False, "H 가 3x3 이 아닙니다"
            self.set(H)
            self.reproj_error_mm = data.get("reproj_error_mm")
            self.num_points = data.get("num_points")
        except Exception as exc:                        # noqa: BLE001
            return False, f"읽기 실패: {exc}"
        return True, None

    def set(self, H):
        self.H = np.asarray(H, dtype=float)
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H_inv = None

    def to_robot(self, u, v):
        """화면 픽셀 -> 로봇 (x, y). 한 번의 행렬 곱."""
        if self.H is None:
            return None
        p = self.H @ np.array([float(u), float(v), 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_pixel(self, x, y):
        """로봇 (x, y) -> 화면 픽셀. 선택 결과를 영상 위에 되돌려 그릴 때 쓴다."""
        if self.H_inv is None:
            return None
        p = self.H_inv @ np.array([float(x), float(y), 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])


class ObjectDetector:
    """색(HSV) 기반 물체 검출.

    컵 색이 일정하다는 전제다. YOLO 를 붙일 거면 이 클래스만 갈아끼우면 된다
    (반환 형식만 맞추면 나머지는 그대로 돈다).

    빨강은 HSV 색상환의 양 끝(0 근처와 180 근처)에 걸쳐 있어서, 하한이 상한보다
    크면 두 구간을 OR 로 합친다. 붉은 컵의 절반을 놓치는 흔한 함정을 막는다.
    """

    def __init__(self, hsv_lower, hsv_upper, min_area=400):
        self.lower = np.array(hsv_lower, dtype=np.uint8)
        self.upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = int(min_area)

    def mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo, hi = self.lower, self.upper
        if int(lo[0]) > int(hi[0]):      # 색상환을 넘어가는 범위(예: 170~10)
            a = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8),
                            np.array([hi[0], hi[1], hi[2]], np.uint8))
            b = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]], np.uint8),
                            np.array([179, hi[1], hi[2]], np.uint8))
            m = cv2.bitwise_or(a, b)
        else:
            m = cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def detect(self, frame, homography):
        """검출 결과를 [(픽셀(u,v), 로봇(x,y), contour), ...] 로 돌려준다.

        호모그래피가 없으면 로봇 좌표를 낼 수 없으므로 빈 목록.
        """
        if not homography.ready:
            return []
        contours, _ = cv2.findContours(self.mask(frame), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
            xy = homography.to_robot(u, v)
            if xy is not None:
                found.append(((u, v), xy, c))
        # 로봇 기준 왼쪽 -> 오른쪽 순으로 번호가 안정되게 정렬
        found.sort(key=lambda it: it[1][1])
        return found


class YoloDetector:
    """YOLO 기반 물체 검출. ObjectDetector 와 같은 형식으로 돌려준다.

    torch 가 든 별도 파이썬(yolo_python)에서 yolo_worker.py 를 자식 프로세스로 돌리고,
    프레임은 JPEG 로 넘긴다. 워커가 바쁘면 그 프레임은 건너뛰고 직전 결과를 쓴다
    (GUI 가 검출 때문에 멈추지 않는다). 박스는 사각형 contour 로 바꿔 넘긴다.
    """

    def __init__(self, python, model, conf=0.25, classes=("cup", "bottle"), min_area=400,
                 logger=None, anchor="bottom"):
        self.cmd = [os.path.expanduser(python), os.path.join(os.path.dirname(__file__), "yolo_worker.py"),
                    os.path.expanduser(model), str(conf), ",".join(classes)]
        self.min_area = int(min_area)
        self.anchor = anchor   # "bottom"(바닥 접촉점) | "center"(박스 중심)
        self.log = logger
        self._proc = None
        self._lock = threading.Lock()
        self._dets = []
        self._busy = False
        self._ready = False
        self._retry_at = 0.0

    def _warn(self, msg):
        if self.log:
            self.log.warn(msg)

    def _start(self):
        if time.time() < self._retry_at:
            return False
        self._retry_at = time.time() + 5.0
        try:
            self._proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, bufsize=0)
        except OSError as e:
            self._warn(f"YOLO 워커를 못 띄웠습니다: {e}")
            self._proc = None
            return False
        self._ready = False
        self._busy = False
        # 코드리뷰(2026-09-28): 재시작 중(yolo11l 로딩 수 초) 죽은 워커의 마지막 박스가 계속
        # 좌표로 나가 치운 컵으로 스냅될 수 있었다 -> 이전 결과를 비운다.
        with self._lock:
            self._dets = []
        threading.Thread(target=self._read_loop, args=(self._proc,), daemon=True).start()
        return True

    def _read_loop(self, proc):
        for line in iter(proc.stdout.readline, b""):
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                if msg.get("ready"):
                    self._ready = True
                if "dets" in msg:
                    self._dets = msg["dets"]
                    self._busy = False

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _submit(self, frame):
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        data = jpg.tobytes()
        try:
            self._proc.stdin.write(struct.pack("<I", len(data)) + data)
            self._busy = True
        except (BrokenPipeError, OSError):
            self._proc = None

    def detect(self, frame, homography):
        if not homography.ready:
            return []
        if self._proc is None or self._proc.poll() is not None:
            if not self._start():
                return []
        with self._lock:
            ready, busy = self._ready, self._busy
        if ready and not busy:
            self._submit(frame)
        with self._lock:
            dets = list(self._dets)
        found = []
        for _, _, x1, y1, x2, y2 in dets:
            if (x2 - x1) * (y2 - y1) < self.min_area:
                continue
            # 박스 중심이 아니라 아래쪽 가운데(바닥 접촉점)를 쓴다. 카메라가 비스듬히
            # 내려다보므로, 컵처럼 높이가 있는 물체는 박스 중심을 쓰면 카메라 반대쪽으로
            # 몇 cm씩 밀린 좌표가 나온다(2026-09-27 실측 12cm). 바닥에 가까운 아래쪽 변이
            # 실제 접촉점에 훨씬 가깝다.
            # 2026-09-28: topdown_click 의 호모그래피는 손끝 높이 0.13m(≈컵 테두리 높이) 평면으로
            # 캘리브돼 있어서 박스 중심이 맞다 — 아래 변을 쓰면 7cm 빗나갔다(컵이 B' 표시 위에
            # 있는데 (0.312,-0.178) 로 나옴). 노드마다 anchor 로 고른다.
            u, v = (x1 + x2) / 2.0, (y2 if self.anchor == "bottom" else (y1 + y2) / 2.0)
            xy = homography.to_robot(u, v)
            if xy is None:
                continue
            box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32).round().astype(np.int32)
            found.append(((u, v), xy, box.reshape(-1, 1, 2)))
        found.sort(key=lambda it: it[1][1])
        return found


class HomographyCalibration:
    """호모그래피 캘리브레이션 절차.

    로봇 자신을 자로 쓴다:
      1. 테이블면 위 지점으로 팔을 보낸다 (관절각은 IK로 푼다)
      2. 그 그리퍼 끝이 화면 어디인지 사람이 클릭한다
      3. 4쌍 이상 모이면 cv2.findHomography 가 행렬을 준다

    ★ 캘리브 점은 전부 같은 높이여야 한다. 호모그래피는 평면↔평면 대응이라,
      높이가 흩어지면 재투영 오차는 작게 나오면서 실제 테이블면 오차가 커진다.
      (관절각을 손으로 적어두면 이 함정에 빠진다 — docs/13_gaze_to_arm.md 2-2)
    """

    def __init__(self, geo, table_z=0.0, calib_height=0.005,
                 ransac_threshold_m=0.005, logger=None):
        self.geo = geo
        self.table_z = float(table_z)
        self.calib_height = float(calib_height)
        self.threshold = float(ransac_threshold_m)
        self.log = logger
        self.poses = []          # 각 지점의 관절각
        self.targets = []        # 각 지점의 테이블면 (x, y)
        self.pixels = []         # 사람이 클릭한 픽셀
        self.world = []          # 그에 대응하는 로봇 좌표
        self.index = 0
        self.result = None       # (H, 평균 재투영 오차 mm, 사용된 점 수)

    # ------------------------------------------------------------------
    def build(self, calib_xy):
        """지점 목록을 IK로 풀어 자세를 만든다. 도달 가능한 것만 남긴다."""
        from gaze_hri.kinematics import IKError, solve_with_fallback

        z = self.table_z + self.calib_height
        self.poses, self.targets = [], []
        skipped = []
        for (x, y) in calib_xy:
            try:
                q, _ = solve_with_fallback([x, y, z], self.geo)
            except IKError:
                skipped.append((x, y))
                continue
            self.poses.append(q)
            self.targets.append((float(x), float(y)))
        if self.log and skipped:
            self.log.warn(f"도달 못 하는 캘리브 지점 {len(skipped)}개를 건너뜁니다: "
                          f"{[(round(a, 2), round(b, 2)) for a, b in skipped]}")
        return len(self.poses)

    @property
    def total(self):
        return len(self.poses)

    @property
    def done(self):
        return self.index >= self.total

    def current_pose(self):
        """지금 로봇이 잡아야 할 관절각. 끝났으면 None."""
        if self.done:
            return None
        return self.poses[self.index]

    def current_target(self):
        if self.done:
            return None
        return self.targets[self.index]

    def add_click(self, u, v):
        """현재 지점에 대한 클릭을 기록하고 다음으로 넘어간다."""
        if self.done:
            return False
        self.pixels.append([float(u), float(v)])
        self.world.append(list(self.targets[self.index]))
        self.index += 1
        return True

    def undo(self):
        """직전 클릭 취소."""
        if self.index == 0:
            return False
        self.index -= 1
        self.pixels.pop()
        self.world.pop()
        return True

    # ------------------------------------------------------------------
    def solve(self):
        """수집된 대응점으로 호모그래피를 구한다. (H, 오차mm, 점수) 또는 (None, 사유)."""
        if len(self.pixels) < 4:
            return None, f"대응점이 {len(self.pixels)}개뿐입니다(최소 4개)"
        src = np.array(self.pixels, dtype=np.float32)
        dst = np.array(self.world, dtype=np.float32)
        # ★ 임계값 단위: dst 가 미터이므로 임계값도 미터다.
        #   (5.0 같은 값을 주면 '5미터'가 되어 오클릭을 하나도 못 거른다)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.threshold)
        if H is None:
            return None, "호모그래피 계산 실패 — 클릭 지점을 확인하세요"
        inliers = int(mask.sum()) if mask is not None else len(src)
        if inliers < 4:
            return None, f"유효 대응점이 {inliers}개뿐입니다"
        pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err_mm = float(np.linalg.norm(pred - dst, axis=1).mean()) * 1000.0
        self.result = (H, err_mm, inliers)
        return self.result, None

    def save(self, path):
        if self.result is None:
            return False
        H, err, n = self.result
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump({"H": H.tolist(), "reproj_error_mm": err,
                            "num_points": n}, f, default_flow_style=False)
        return True


def hsv_range_from_pixel(frame, u, v, window=5, h_margin=10,
                         s_margin=70, v_margin=70):
    """클릭한 자리의 색으로 HSV 검출 범위를 만든다.

    실기에서 컵 색을 맞추는 데 시간이 가장 많이 든다. 컵을 한 번 클릭하면
    주변 픽셀의 중앙값을 잡아 범위를 만들어 준다.

    빨강처럼 색상환 양 끝(0/179 근처)에 걸치는 색은 하한 > 상한 형태로
    돌려준다. ObjectDetector 가 그 경우 두 구간을 합쳐서 처리한다.
    """
    h, w = frame.shape[:2]
    u, v = int(round(u)), int(round(v))
    x0, x1 = max(0, u - window), min(w, u + window + 1)
    y0, y1 = max(0, v - window), min(h, v + window + 1)
    patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    med = np.median(patch.reshape(-1, 3), axis=0)
    hue, sat, val = float(med[0]), float(med[1]), float(med[2])

    lo_h = (hue - h_margin) % 180.0
    hi_h = (hue + h_margin) % 180.0
    lower = [int(round(lo_h)), int(max(40, sat - s_margin)),
             int(max(40, val - v_margin))]
    upper = [int(round(hi_h)), int(min(255, sat + s_margin)),
             int(min(255, val + v_margin))]
    return lower, upper, (hue, sat, val)


def find_camera(preferred=0, width=640, height=480, max_index=8):
    """카메라를 연다. preferred 가 안 되면 다른 인덱스를 훑는다.

    실기에서 USB 를 다시 꽂으면 인덱스가 바뀌는 일이 잦다.
    preferred 가 음수면 처음부터 자동 탐색한다.
    """
    candidates = ([] if preferred is None or int(preferred) < 0
                  else [int(preferred)])
    candidates += [i for i in range(max_index) if i not in candidates]
    tried = []
    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            ok, _ = cap.read()
            if ok:
                return cap, idx
        cap.release()
        tried.append(idx)
    raise RuntimeError(
        f"카메라를 열 수 없습니다 (시도한 인덱스 {tried}). "
        "`ls /dev/video*` 로 확인하고, 다른 프로그램이 쓰고 있지 않은지 보세요."
    )


def open_camera(index, width, height):
    """카메라를 열고 해상도를 맞춘다. 실패하면 RuntimeError."""
    cap = cv2.VideoCapture(int(index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if not cap.isOpened():
        raise RuntimeError(
            f"카메라 {index} 를 열 수 없습니다. `ls /dev/video*` 로 인덱스를 "
            "확인하세요. 다른 프로그램이 쓰고 있어도 안 열립니다."
        )
    return cap
