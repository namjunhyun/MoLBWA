#!/usr/bin/env python3
"""
창종설 0단계 → 융합 징검다리 — "지금 뭘 보고 있는지"를 씬(외부) 카메라 영상에 표시.

레포의 FrontCameraTracker(Orlosky3DEyeTrackerFrontCamera.py) 아이디어를 리눅스로 옮긴 것.
원본을 그대로 못 쓰는 이유 2가지:
  1) cv2.CAP_MSMF = 윈도우 전용 백엔드. 리눅스는 CAP_V4L2.
  2) 원본 FrontCamera 파일은 numpy2 uint8 overflow 패치가 안 되어 있다(line 86/111).
     → 검출기는 이미 패치·검증된 3DTracker/Orlosky3DEyeTracker.py를 재사용하고,
       회전·투영 로직만 FrontCamera에서 가져온다.

원리:
  1점 캘리브 — 'c'를 누른 순간의 시선방향을 "씬 카메라 정면 [0,0,1]"으로 놓는 회전 R을 구한다
  (Rodrigues). 벡터쌍 1개라 캘리브 지점에서 멀어질수록 오차가 커진다.

  다점 캘리브('m') — 실제 물체/점을 씬 카메라 앞에 두고 응시한 상태에서, 화면의 그 실제
  위치를 마우스로 클릭 → (그 순간의 시선벡터, 클릭 픽셀→광선) 쌍을 여러 개 모아서
  Wahba's problem(Kabsch/SVD)로 전체 오차를 최소화하는 회전 R을 통계적으로 구한다.
  1점보다 훨씬 안정적 — 점을 늘릴수록 정확도 개선.

  둘 다 눈↔씬 카메라 외부파라미터(extrinsic)를 따로 구할 필요가 없다(docs/01의 2D→2D 매핑 정신).

카메라:
  눈  = Sonix UVC IR (by-id 자동탐색)
  씬  = oCamS-1MGN-U. YUYV raw에서 Y채널=왼쪽, UV채널=오른쪽 (스테레오쌍).
        여기선 왼쪽만 쓴다. 노출은 기본값으로 충분(수동으로 올리지 말 것).
        ocams_calib.py의 rectify 맵으로 왜곡보정+정렬(rectify)까지 거친 이미지를 쓴다 —
        docs/03_fusion.md의 스테레오 깊이(K=ocams_calib.RECTIFIED_K)와 좌표계를 맞추기 위함.
        640x480 고정(캘리브레이션 해상도와 다르면 rectify 맵이 안 맞음).

사용:
  python gaze_on_scene.py                 # 창 두 개(눈/씬). 눈 굴려 모델 수렴 → 'c'로 캘리브
  키: c=1점 캘리브 / m=다점 캘리브 모드 토글(클릭으로 포인트 추가) / r=리셋 / q=종료

시각화(docs/09_visualization.md A-2, 2026-07-30):
  cv2 창(다점 캘리브용 마우스 클릭 때문에 그대로 둠)에 더해, 같은 프레임을 Rerun에도 로깅한다
  ("eye/ir", "scene/image", "scene/gaze_cursor", "scene/calib_points"). 나중에 pose(C)/깊이(B)가
  붙으면 여기 3D 엔티티만 추가하면 됨 — cv2 쪽은 안 건드려도 된다. --no-rerun으로 끌 수 있음.
"""
import os
import sys
import argparse
import json
import time
import numpy as np
import cv2
try:
    import rerun as rr
except ImportError:
    rr = None

import ocams_calib
import eye_scene_extrinsic
import fusion
from gaze_udp_sender import GazeUdpSender
from gaze_px_udp import GazePixelSender

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "external", "EyeTracker", "3DTracker")))
os.chdir(HERE)

import Orlosky3DEyeTracker as tracker  # noqa: E402  (numpy2 패치 + 2026-07-30 반환값 패치본)

BY_ID = "/dev/v4l/by-id"


def find_cam(keyword):
    """USB 재연결마다 /dev/videoN이 밀리므로 by-id의 index0(캡처 노드)을 이름으로 찾는다."""
    for link in sorted(os.listdir(BY_ID)):
        if keyword.lower() in link.lower() and link.endswith("index0"):
            path = os.path.realpath(os.path.join(BY_ID, link))
            print(f"[cam] {keyword}: {link} -> {path}")
            return path
    raise RuntimeError(f"'{keyword}' 카메라를 {BY_ID}에서 못 찾음 — USB 연결 확인")


def rotation_from_a_to_b(a, b):
    """R @ a = b 인 회전행렬 (Rodrigues). FrontCameraTracker에서 가져옴."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-6:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        axis = np.array([0.0, 1.0, 0.0]) if abs(a[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        s = 1.0
        c = -1.0
    else:
        v = v / s
    vx, vy, vz = v
    K = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + K * s + (K @ K) * ((1 - c) / (s ** 2))).astype(np.float32)


def pixel_to_ray(u, v, fx, fy, cx, cy):
    """씬 이미지 픽셀 -> 카메라 좌표계 단위 광선 (project()의 역변환)."""
    r = np.array([(u - cx) / fx, (cy - v) / fy, 1.0], dtype=np.float32)
    return r / np.linalg.norm(r)


def reprojection_error(fx_val, dirs, pixels, R, cx, cy):
    """현재 R, fx로 각 시선벡터를 투영했을 때 실제 클릭 픽셀과의 오차 제곱합."""
    err = 0.0
    for d, (u, v) in zip(dirs, pixels):
        g = R @ d
        if g[2] <= 1e-6:
            err += 1e6
            continue
        up = cx + fx_val * (g[0] / g[2])
        vp = cy - fx_val * (g[1] / g[2])
        err += (up - u) ** 2 + (vp - v) ** 2
    return err


def calibrate_multi(dirs, pixels, fx0, cx, cy, iters=8, n_grid=21):
    """R(회전)과 fx(초점거리/스케일)를 같이 최적화.
    fx 근사가 틀리면 화면 중심에서 멀수록 오차가 커지는데, 회전만으론 그 오차를 못 없앤다 —
    점 3개 이상이면 fx 후보마다 그때그때 최적 R을 다시 풀어(nested) 재투영오차가 가장 작은
    fx를 grid search로 좁혀간다 (고정 R로 fx만 흔들면 국소최적에 갇힘 — 그래서 매번 R도 다시 품).
    점이 2개면 fx0 고정, 회전만 SVD로 푼다."""
    def best_R_for(fx_val):
        rays = [pixel_to_ray(u, v, fx_val, fx_val, cx, cy) for u, v in pixels]
        R = solve_rotation_svd(dirs, rays)
        return R, reprojection_error(fx_val, dirs, pixels, R, cx, cy)

    fx_val = float(fx0)
    R, _ = best_R_for(fx_val)
    if len(dirs) < 3:
        return R, fx_val

    width = 0.5 * fx_val
    for _ in range(iters):
        candidates = np.linspace(max(fx_val - width, 1.0), fx_val + width, n_grid)
        results = [best_R_for(f) for f in candidates]
        errs = [e for _, e in results]
        best_idx = int(np.argmin(errs))
        fx_val = float(candidates[best_idx])
        R = results[best_idx][0]
        width *= 0.5
    return R, fx_val


AFFINE_FEATURES_VERSION = 2   # 1 = (x/z, y/z, 1), 2 = (x, y, 1)


def gaze_features(direction):
    """시선 단위벡터 -> affine 입력 (x, y, 1).

    예전(v1)에는 원근 정규화 (x/z, y/z) 를 썼다. 그런데 눈 카메라가 눈을 비스듬히
    보기 때문에 z 가 0.48~0.94 로 크게 흔들리고, 나누는 순간 가로축까지 휜다.
    2026-09-23 10점 실측: 원시 x 와 화면 u 의 상관 0.990 인데 x/z 로는 가로 잔차
    31px. 원시 (x, y) 로 같은 점을 맞추면 RMS 47.6 -> 36.8px.
    """
    d = np.asarray(direction, dtype=np.float64)
    return np.array([d[0], d[1], 1.0], dtype=np.float64)


def calibrate_affine(dirs, pixels):
    """가로/세로 스케일과 교차축 영향을 독립적으로 맞추는 2D affine 시선 매핑."""
    X = np.stack([gaze_features(d) for d in dirs])
    Y = np.asarray(pixels, dtype=np.float64)
    if len(X) < 3 or np.linalg.matrix_rank(X) < 3:
        raise ValueError("affine 계산에는 서로 다른 방향의 점이 최소 3개 필요")
    coeff, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return coeff.T.astype(np.float32)


def affine_reprojection_error(affine, dirs, pixels):
    predicted = np.stack([affine @ gaze_features(d) for d in dirs])
    actual = np.asarray(pixels, dtype=np.float64)
    return float(np.sum((predicted - actual) ** 2))


def print_calib_report(affine, dirs, pixels):
    """점별 오차와 방향 분포를 찍는다. 평균만으로는 원인을 못 가린다.

    - 한두 점만 크면 -> 그 점만 잘못 들어간 것
    - 전부 고르게 크면 -> 모델이 안 맞는 것(시차/비선형)
    - 방향 분포가 작으면 -> 눈을 안 움직이고 고개를 돌린 것
    """
    X = np.stack([gaze_features(d) for d in dirs])
    pred = X @ np.asarray(affine, dtype=np.float64).T
    act = np.asarray(pixels, dtype=np.float64)
    per = np.linalg.norm(pred - act, axis=1)

    arr = np.asarray(dirs, dtype=np.float64)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    spread = float(np.degrees(np.arccos(np.clip(arr @ arr.T, -1, 1))).max())
    cond = float(np.linalg.cond(X))

    print(f"[진단] 시선 방향 분포={spread:.1f}도  조건수={cond:.1f}")
    med = float(np.median(per))
    for i, (e, (u, v)) in enumerate(zip(per, pixels), 1):
        flag = "  <-- 이 점이 나쁨" if e > 2.5 * max(med, 1.0) else ""
        print(f"        {i:2d}. 클릭({u:3d},{v:3d})  오차 {e:6.1f}px{flag}")
    if spread < 10.0:
        print("[진단] 방향이 거의 안 벌어졌다 — 고개를 고정하고 '눈만' 움직였는지 확인할 것.")


def current_eye_model():
    """F 로 고정된 안구 모델 (중심 픽셀, 반지름). 자동 갱신 중이면 None."""
    if tracker.eye_sphere_adjustment_enabled:
        return None
    cx_, cy_ = tracker.prev_model_center_avg
    return {"center": [int(cx_), int(cy_)], "radius": float(tracker.max_observed_distance)}


def restore_eye_model(eye_model):
    """저장된 안구 중심을 트래커에 넣고 고정한다. 시선 방향은 이 중심만으로 계산된다
    (Orlosky3DEyeTracker.compute_gaze_vector). 반지름은 표시용."""
    tracker.prev_model_center_avg = tuple(int(c) for c in eye_model["center"])
    tracker.max_observed_distance = float(eye_model.get("radius", 0.0))
    tracker.eye_sphere_adjustment_enabled = False


def last_point_is_outlier(affine, dirs, pixels):
    """마지막 점이 print_calib_report 와 같은 기준(중앙값의 2.5배)으로 튀는지."""
    X = np.stack([gaze_features(d) for d in dirs])
    per = np.linalg.norm(X @ np.asarray(affine, dtype=np.float64).T
                         - np.asarray(pixels, dtype=np.float64), axis=1)
    return per[-1] > 2.5 * max(float(np.median(per)), 1.0), float(per[-1])


def save_affine_calibration(path, affine, dirs, pixels, width, height):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    px_error = np.sqrt(affine_reprojection_error(affine, dirs, pixels) / len(dirs))
    payload = {
        "version": AFFINE_FEATURES_VERSION,
        "model": "gaze_direction_to_scene_pixel_affine",
        "features": "raw_xy",
        "affine_2x3": np.asarray(affine).tolist(),
        "sample_count": len(dirs),
        "mean_pixel_error": float(px_error),
        "image_width": int(width),
        "image_height": int(height),
        # 원시 쌍을 남긴다. 이게 없으면 "왜 오차가 컸는지" 를 되짚을 방법이 없다.
        "gaze_dirs": np.asarray(dirs, dtype=float).tolist(),
        "target_pixels": [[int(u), int(v)] for u, v in pixels],
        # affine 은 이 안구 중심을 기준으로 한 시선 벡터에만 맞는다. 재시작하면 중심이
        # 새로 추정되므로 같이 남긴다 (--restore-eye-model). 자동 모드였으면 None.
        "eye_model": current_eye_model(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return px_error


def load_affine_calibration(path, width, height):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        affine = np.asarray(payload["affine_2x3"], dtype=np.float32)
        if affine.shape != (2, 3):
            raise ValueError(f"행렬 크기가 {affine.shape}, 기대값은 (2, 3)")
        if (payload.get("image_width"), payload.get("image_height")) != (width, height):
            raise ValueError("저장 당시와 현재 씬 영상 해상도가 다름")
        if payload.get("version", 1) < AFFINE_FEATURES_VERSION:
            # 옛 입력 모델로 맞춘 행렬은 지금 gaze_features 와 안 맞는다. 원시 쌍이
            # 남아 있으면 새 모델로 다시 맞추고, 없으면 쓰지 않는다.
            dirs, pixels = payload.get("gaze_dirs"), payload.get("target_pixels")
            if not dirs or not pixels:
                raise ValueError("옛 형식(x/z) 파일인데 원시 시선/픽셀 쌍이 없어 재계산 불가")
            affine = calibrate_affine(dirs, [tuple(p) for p in pixels])
            payload["mean_pixel_error"] = float(np.sqrt(
                affine_reprojection_error(affine, dirs, pixels) / len(dirs)))
            print(f"[calib] 옛 형식 파일 -> 원시 (x,y) 로 재계산: {os.path.basename(path)} "
                  f"({len(dirs)}점, {payload['mean_pixel_error']:.1f}px)")
        return affine, payload
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        print(f"[calib] 저장 파일 불러오기 실패: {e}")
        return None


def solve_rotation_svd(dirs, rays):
    """Wahba's problem: sum||R@d_i - r_i||^2 최소화하는 회전 R (Kabsch/SVD).
    벡터쌍 2개 이상이면 1점 캘리브(rotation_from_a_to_b)보다 안정적으로 전체 오차를 분산시킴."""
    A = np.zeros((3, 3), dtype=np.float64)
    for d, r in zip(dirs, rays):
        A += np.outer(np.asarray(r, dtype=np.float64), np.asarray(d, dtype=np.float64))
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    R = U @ D @ Vt
    return R.astype(np.float32)


# --------------------------------------------------------------------------
# ROS2 토픽에서 프레임 받기 (--source ros)
#
# 카메라가 라즈베리파이에 붙어 있고 Pi 는 토픽만 쏜다. 무거운 동공 검출과
# 시선 계산은 이 데스크탑에서 한다.
#
# 중요: Pi 의 ocams_stereo_imu_node 는 좌/우 분리와 rectification 을 이미 끝내고
# 발행한다 (right=split[0], left=split[1] — 이 파일 scene_stereo 와 같은 규약).
# 그래서 여기서 또 쪼개거나 remap 하면 안 된다.
# --------------------------------------------------------------------------

class RosFrameSource:
    """압축 토픽을 구독해 스트림별 '최신 한 장'만 들고 있는다.

    큐를 쌓지 않는 이유: 데스크탑 처리가 느려져도 지연이 누적되면 안 된다.
    실시간에서는 밀린 프레임보다 건너뛴 프레임이 낫다.
    """

    def __init__(self, eye_topic, left_topic, right_topic, stale_after=1.0):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
        import threading

        self._stale_after = stale_after
        self._lock = threading.Lock()
        self._frames = {}          # topic -> (image, 수신시각)

        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self._node = Node("gaze_on_scene_source")

        # BEST_EFFORT: 무선 구간에서 재전송을 기다리느니 프레임을 버린다.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.eye_topic, self.left_topic, self.right_topic = eye_topic, left_topic, right_topic
        for t in (eye_topic, left_topic, right_topic):
            self._node.create_subscription(
                CompressedImage, t, lambda m, _t=t: self._on_image(_t, m), qos)

        self._exec = rclpy.executors.SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        print(f"[ros] 구독: {eye_topic} / {left_topic} / {right_topic}")

    def enable_pose(self, topic="/orbslam3/pose", stale_after=0.3):
        """SLAM 헤드 포즈 구독을 이 노드에 얹는다.

        별도 노드/executor 를 만들지 않는 이유: 이미 도는 executor 에 구독 하나를
        더 거는 게 훨씬 싸고, 영상과 포즈가 같은 스레드에서 갱신돼 경합이 준다.

        stale_after 가 영상(1.0s)보다 짧은 게 핵심이다. stereo-inertial 노드는
        추적 상태가 OK 일 때만 pose 를 낸다(stereo-inertial-node.cpp 의
        GetTrackingState() == OK 가드). 즉 "포즈가 끊겼다" == "추적을 잃었다" 이고,
        이 staleness 가 그대로 유효성 게이트가 된다. 추적을 잃은 뒤의 마지막 포즈를
        붙들고 있으면 엉뚱한 세계 좌표를 로봇팔로 쏘게 된다.
        """
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        self._pose_stale_after = stale_after
        self._pose = None          # (T_WS 4x4, 수신시각)
        self.pose_topic = topic
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._node.create_subscription(PoseStamped, topic, self._on_pose, qos)
        print(f"[ros] SLAM 포즈 구독: {topic} (staleness {stale_after:.2f}s = 추적 상실 판정)")

    def _on_pose(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        try:
            T_WS = fusion.pose_to_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        except ValueError:
            return
        with self._lock:
            self._pose = (T_WS, time.time())

    def latest_pose(self):
        """최신 T_WS (4x4). 추적 상실/미수신이면 None."""
        with self._lock:
            got = getattr(self, "_pose", None)
        if got is None:
            return None
        T_WS, t = got
        if time.time() - t > self._pose_stale_after:
            return None
        return T_WS

    def _spin(self):
        try:
            self._exec.spin()
        except Exception:
            pass

    def _on_image(self, topic, msg):
        buf = np.frombuffer(msg.data, np.uint8)
        # 씬은 mono8, 눈은 컬러. UNCHANGED 로 원본 채널 수를 보존한다.
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        with self._lock:
            self._frames[topic] = (img, time.time())

    def _latest(self, topic):
        with self._lock:
            got = self._frames.get(topic)
        if got is None:
            return None
        img, t = got
        if time.time() - t > self._stale_after:
            return None          # 발행이 끊겼다 — 호출자가 재연결 대기로 처리
        return img

    def eye_frame(self):
        """눈 영상 BGR. 없으면 None."""
        img = self._latest(self.eye_topic)
        if img is None:
            return None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def scene_pair(self):
        """(left, right) 모노. Pi 가 이미 분리·rectify 한 것을 그대로 쓴다."""
        left, right = self._latest(self.left_topic), self._latest(self.right_topic)
        if left is None or right is None:
            return None, None
        if left.ndim == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if right.ndim == 3:
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(left), np.ascontiguousarray(right)

    def shutdown(self):
        try:
            self._exec.shutdown()
            self._node.destroy_node()
        except Exception:
            pass


class RosEyeCap:
    """눈 카메라 자리에 끼우는 shim. 기존 eye_cap.read()/.release() 를 그대로 쓴다."""

    def __init__(self, source):
        self._src = source

    def read(self):
        f = self._src.eye_frame()
        return (f is not None), f

    def release(self):
        pass


def open_eye(width, height, fps):
    cap = cv2.VideoCapture(find_cam("Sonix"), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError("눈 카메라 열기 실패 (다른 프로세스가 점유 중인지 확인)")
    return cap


def open_scene(width, height):
    cap = cv2.VideoCapture(find_cam("oCamS"), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # raw YUYV: Y=left, UV=right
    if not cap.isOpened():
        raise RuntimeError("씬 카메라(oCamS) 열기 실패")
    return cap


def scene_left(cap, rectify_maps=None):
    """oCamS raw YUYV → 왼쪽 영상(Y채널) BGR.
    rectify_maps가 주어지면 왜곡보정+스테레오 정렬까지 적용(ocams_calib 좌표계로 맞춤)."""
    ok, f = cap.read()
    if not ok or f is None:
        return None
    if f.ndim == 3 and f.shape[2] == 2:
        left = f[:, :, 0]
    elif f.ndim == 2:
        left = f[:, 0::2]
    else:
        left = None  # 이미 BGR인 경우 아래에서 그대로 반환

    if left is None:
        bgr = f
    else:
        left = np.ascontiguousarray(left)
        if rectify_maps is not None:
            map1, map2 = rectify_maps
            left = cv2.remap(left, map1, map2, cv2.INTER_LINEAR)
        bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    return bgr


def scene_stereo(cap, left_maps=None, right_maps=None):
    """동일한 oCamS raw 프레임에서 rectified 좌/우 모노 영상을 함께 반환."""
    ok, raw = cap.read()
    if not ok or raw is None:
        return None, None
    if raw.ndim == 3 and raw.shape[2] == 2:
        # 이 oCamS 실기에서는 보정 파일 기준 left=두 번째 바이트, right=첫 번째 바이트.
        left = np.ascontiguousarray(raw[:, :, 1])
        right = np.ascontiguousarray(raw[:, :, 0])
    elif raw.ndim == 2:
        left = np.ascontiguousarray(raw[:, 1::2])
        right = np.ascontiguousarray(raw[:, 0::2])
    else:
        return None, None
    if left_maps is not None:
        left = cv2.remap(left, left_maps[0], left_maps[1], cv2.INTER_LINEAR)
    if right_maps is not None:
        right = cv2.remap(right, right_maps[0], right_maps[1], cv2.INTER_LINEAR)
    return left, right


def make_stereo_matcher():
    block_size = 7
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=block_size,
        P1=8 * block_size ** 2, P2=32 * block_size ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
    )


def depth_at(disparity, u, v, fx, baseline_m, radius=3, min_valid=5,
             min_valid_ratio=0.7, max_relative_mad=0.08,
             max_relative_spread=0.2):
    if disparity is None:
        return None, 0
    h, w = disparity.shape
    u, v = int(u), int(v)
    x1, x2 = max(0, u - radius), min(w, u + radius + 1)
    y1, y2 = max(0, v - radius), min(h, v + radius + 1)
    roi = disparity[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    required = max(min_valid, int(np.ceil(roi.size * min_valid_ratio)))
    if len(valid) < required:
        return None, int(len(valid))
    disp = float(np.median(valid))
    mad = float(np.median(np.abs(valid - disp)))
    p10, p90 = np.percentile(valid, [10, 90])
    if (disp <= 0 or mad / disp > max_relative_mad or (p90 - p10) / disp > max_relative_spread):
        return None, int(len(valid))
    return float(fx * baseline_m / disp), int(len(valid))


def interpolate_affine(depth_m, affine_05, affine_10):
    alpha = float(np.clip((depth_m - 0.5) / 0.5, 0.0, 1.0))
    return ((1.0 - alpha) * affine_05 + alpha * affine_10).astype(np.float32)


# 로봇팔로 보낼 깊이의 허용 범위. 팔이 닿는 거리(테이블 위 컵)를 넘어서는 값은
# 대부분 배경을 잘못 짚은 것이라 보내지 않는다.
FUSE_MIN_DEPTH_M = 0.15
FUSE_MAX_DEPTH_M = 2.5


def fuse_and_send(sender, ros_src, gaze_valid, u, v, disparity, baseline_m,
                  last_depth_m, scene_flipped, sw, sh):
    """시선 픽셀 + 스테레오 깊이 + SLAM 포즈 -> 세계좌표 p_W -> UDP 송신.

    반환: (갱신된 last_depth_m, 상태문자열)

    유효하지 않으면 valid=False 패킷을 보낸다. 조용히 안 보내면 gaze_bridge 쪽에서
    "아직 안 왔다"와 "지금 못 믿는다"를 구분할 수 없고, dwell_detector 가 마지막
    유효 샘플을 계속 붙들게 된다.
    """
    if not gaze_valid:
        sender.send([0.0, 0.0, 0.0], valid=False)
        return last_depth_m, "시선없음"

    # --scene-flip 은 표시용 이미지만 180도 돌린다. 캘리브레이션은 그 돌아간 화면 위에서
    # 클릭해서 만들었으므로 (u,v) 는 '돌아간' 좌표계다. 반면 disparity 는 돌리기 전
    # left_gray/right_gray 로 계산했고 K(RECTIFIED_K) 도 돌리기 전 좌표계 기준이다.
    # 깊이 조회와 역투영은 둘 다 돌리기 전 좌표로 해야 한다.
    if scene_flipped:
        u_cam, v_cam = sw - 1 - u, sh - 1 - v
    else:
        u_cam, v_cam = u, v

    D, _n = depth_at(disparity, u_cam, v_cam,
                     ocams_calib.RECTIFIED_K[0, 0], baseline_m, radius=5)
    if D is None:
        sender.send([0.0, 0.0, 0.0], valid=False)
        return last_depth_m, "깊이없음"
    if not (FUSE_MIN_DEPTH_M <= D <= FUSE_MAX_DEPTH_M):
        sender.send([0.0, 0.0, 0.0], valid=False)
        return last_depth_m, f"깊이범위밖 {D:.2f}m"

    T_WS = ros_src.latest_pose() if ros_src is not None else None
    if T_WS is None:
        # 포즈는 추적 OK 일 때만 발행된다 -> 없음 == 추적 상실.
        sender.send([0.0, 0.0, 0.0], valid=False)
        return D, "SLAM끊김"

    p_W, _origin, _ray = fusion.gaze_point_world(
        u_cam, v_cam, D, ocams_calib.RECTIFIED_K, T_WS)
    if not np.all(np.isfinite(p_W)):
        sender.send([0.0, 0.0, 0.0], valid=False)
        return D, "p_W 비정상"

    sender.send(p_W, valid=True, T_WS=T_WS)
    return D, f"송신 ({p_W[0]:+.2f},{p_W[1]:+.2f},{p_W[2]:+.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye-width", type=int, default=640)
    ap.add_argument("--eye-height", type=int, default=480)
    ap.add_argument("--eye-fps", type=int, default=30)
    ap.add_argument("--scene-width", type=int, default=ocams_calib.IMAGE_WIDTH,
                    help="ocams_calib.py의 캘리브레이션 해상도와 같아야 rectify 맵이 맞는다 (기본 640)")
    ap.add_argument("--scene-height", type=int, default=ocams_calib.IMAGE_HEIGHT,
                    help="위와 동일 (기본 480)")
    ap.add_argument("--fx", type=float, default=None,
                    help="씬 카메라 초점거리(px) 강제 지정. 미지정시 ocams_calib의 rectified "
                         "캘리브레이션 값을 그대로 씀(근사 아님) — 스테레오 깊이(docs/03)와 "
                         "좌표계를 맞추려면 이 기본값을 그대로 쓸 것")
    ap.add_argument("--baseline-m", type=float, default=None,
                    help="SGBM 깊이 baseline(m) 강제 지정, 진단용. 미지정시 0.5/1.05/1.5m 실측으로 "
                         "검증한 ocams_calib.DEPTH_BASELINE_M을 사용한다.")
    ap.add_argument("--smooth", type=float, default=0.25,
                    help="시선벡터 EMA 계수(0=고정,1=생값). 7/2 노트의 프레임간 튐(std0.18) 완화")
    # 주의: --flip 은 이름과 달리 180도 회전(상하+좌우)이다. 기존 동작을 바꾸면
    # 이미 이걸로 맞춰둔 설정이 깨지므로 그대로 두고, 축별 옵션을 따로 뒀다.
    ap.add_argument("--flip", action="store_true",
                    help="눈 영상 180도 회전 (상하+좌우 동시)")
    ap.add_argument("--flip-v", action="store_true", help="눈 영상 상하만 반전")
    ap.add_argument("--flip-h", action="store_true", help="눈 영상 좌우만 반전")
    ap.add_argument("--scene-flip", action="store_true", help="씬 카메라(oCamS) 180도 반전 (카메라가 거꾸로 장착된 경우)")
    ap.add_argument("--enable-experimental-depth", action="store_true",
                    help="검증되지 않은 스테레오 깊이 기반 0.5m/1.0m affine 보간 사용")
    ap.add_argument("--no-mirror-x", action="store_true",
                    help="시선 x축 반전을 끈다. 기본은 켬 — 눈 카메라는 사용자를 마주보므로 "
                         "씬 카메라와 좌우가 뒤집힌다(1점 캘리브의 최소회전으로는 못 고침)")
    ap.add_argument("--mirror-y", action="store_true", help="시선 y축도 반전(상하가 뒤집힐 때)")
    ap.add_argument("--source", choices=["local", "ros"], default="local",
                    help="local=USB 카메라 직접 / ros=라즈베리파이가 쏘는 토픽 구독")
    ap.add_argument("--eye-topic", default="/eye/image_raw/compressed")
    ap.add_argument("--left-topic", default="/camera/left/compressed")
    ap.add_argument("--right-topic", default="/camera/right/compressed")
    # --- 로봇팔 연동 (융합 송신) ---
    ap.add_argument("--send-udp", action="store_true",
                    help="시선 3D 세계좌표를 gaze_bridge 로 UDP 송신. --source ros 필요 "
                         "(SLAM 포즈 T_WS 를 같은 rclpy 노드에서 구독한다)")
    ap.add_argument("--udp-host", default="127.0.0.1",
                    help="gaze_bridge 가 도는 호스트 (기본 127.0.0.1)")
    ap.add_argument("--udp-port", type=int, default=55055,
                    help="gaze_bridge 의 udp_port 파라미터와 같아야 한다 (기본 55055)")
    ap.add_argument("--pose-topic", default="/orbslam3/pose",
                    help="SLAM 헤드 포즈 토픽 (PoseStamped, frame=map)")
    ap.add_argument("--pose-stale-sec", type=float, default=0.3,
                    help="이 시간 동안 포즈가 안 오면 추적 상실로 보고 송신을 막는다")
    ap.add_argument("--send-gaze-px", action="store_true",
                    help="시선 픽셀 (u,v) 를 arm/run_demo.py --tag-direct 로 UDP 송신 (SLAM 불필요)")
    ap.add_argument("--gaze-px-host", default="127.0.0.1")
    ap.add_argument("--gaze-px-port", type=int, default=55056)
    ap.add_argument("--no-rerun", action="store_true", help="Rerun 로깅 끄기 (cv2 창만 사용)")
    ap.add_argument("--restore-eye-model", action="store_true",
                    help="캘리브 파일에 저장된 안구 중심을 복원하고 고정한다. 안경을 벗지 않고 "
                         "뷰어만 재시작했을 때용 — 다시 썼으면 새로 캘리브할 것")
    ap.add_argument("--rerun-every", type=int, default=3,
                    help="Rerun에 영상을 N프레임마다 1번만 로깅 (기본 3). 매 프레임 원본 "
                         "해상도로 다 보내면 gRPC 버퍼(1GiB)가 몇 분 안에 차서 뷰어가 죽는다 "
                         "(2026-07-30에 실제로 겪음) — cv2 창 표시/캘리브는 영향 없음, "
                         "Rerun 쪽 영상만 덜 자주 보냄")
    args = ap.parse_args()

    baseline_m = args.baseline_m if args.baseline_m is not None else ocams_calib.DEPTH_BASELINE_M
    if args.baseline_m is not None:
        print(f"[baseline] 강제 지정: {baseline_m:.4f}m "
              f"(검증 기본값 {ocams_calib.DEPTH_BASELINE_M:.4f}m 대신)")

    if args.enable_experimental_depth:
        print("[경고] SGBM 거리 스케일은 검증됐지만 거리별 시선 affine 보간은 아직 실험 기능입니다.")

    if rr is None and not args.no_rerun:
        print("[rerun] 패키지 없음 — cv2 창만 사용")
        args.no_rerun = True
    if not args.no_rerun:
        rr.init("molbwa_gaze_on_scene", spawn=True)
    rerun_ok = not args.no_rerun  # gRPC 전송 에러 나면 죽이지 말고 이후로는 꺼버림

    # 눈/씬 카메라가 서로 마주보는 데서 오는 축 반전. 런타임에 x/y 키로 토글 가능.
    sign = np.array([1.0 if args.no_mirror_x else -1.0,
                     -1.0 if args.mirror_y else 1.0,
                     1.0], dtype=np.float32)

    if args.send_udp and args.source != "ros":
        # 로컬 USB 모드에는 SLAM 포즈를 받을 rclpy 노드가 없다. 조용히 안 보내는
        # 것보다 여기서 멈추는 게 낫다 — 로봇팔이 대기만 하다 끝나는 걸 디버깅하기 어렵다.
        ap.error("--send-udp 는 --source ros 가 필요하다 (SLAM 포즈 구독 경로가 거기에만 있다)")

    ros_src = None
    if args.source == "ros":
        ros_src = RosFrameSource(args.eye_topic, args.left_topic, args.right_topic)
        # 눈은 shim 으로 기존 eye_cap 경로를 그대로 태운다.
        # 씬은 이미 rectify 된 쌍이 오므로 루프에서 따로 분기한다.
        eye_cap = RosEyeCap(ros_src)
        scene_cap = None
        if args.send_udp:
            ros_src.enable_pose(args.pose_topic, args.pose_stale_sec)
    else:
        try:
            eye_cap = open_eye(args.eye_width, args.eye_height, args.eye_fps)
        except (OSError, RuntimeError) as e:
            eye_cap = None
            print(f"[startup] 눈 카메라 없음 — 연결 대기: {e}")
        try:
            scene_cap = open_scene(args.scene_width, args.scene_height)
        except (OSError, RuntimeError) as e:
            scene_cap = None
            print(f"[startup] 씬 카메라 없음 — 연결 대기: {e}")

    px_sender = None
    if args.send_gaze_px:
        px_sender = GazePixelSender(args.gaze_px_host, args.gaze_px_port)
        print(f"[gaze-px] 시선 픽셀 송신: {args.gaze_px_host}:{args.gaze_px_port}")

    udp_sender = None
    if args.send_udp:
        udp_sender = GazeUdpSender(args.udp_host, args.udp_port)
        print(f"[udp] 시선 3D점 송신: {args.udp_host}:{args.udp_port}")
        # 스케일 불일치 경고. depth 와 pose 가 서로 다른 '미터'를 쓰면 p_W 는 의미가 없다.
        # ocams_calib 은 SGBM 용으로 10.5cm(2026-09-22 실측 보정), SLAM 설정은 기하값
        # 17.16cm 를 쓴다 — 비율 1.63배. 어느 쪽이 맞는지는 줄자 실측으로만 정해진다.
        if abs(baseline_m - ocams_calib.GEOMETRIC_BASELINE_M) > 1e-4:
            print(f"[udp][주의] 깊이 baseline({baseline_m*100:.1f}cm)이 SLAM 기하 "
                  f"baseline({ocams_calib.GEOMETRIC_BASELINE_M*100:.1f}cm)과 다르다 "
                  f"— 비율 {ocams_calib.GEOMETRIC_BASELINE_M/baseline_m:.2f}배. "
                  f"둘 중 하나는 틀렸고, 틀린 쪽만큼 p_W 가 어긋난다. "
                  f"verify_depth_scale.py 로 실측 확인할 것.")

    if (args.scene_width, args.scene_height) != (ocams_calib.IMAGE_WIDTH, ocams_calib.IMAGE_HEIGHT):
        print(f"[경고] --scene-width/height가 캘리브레이션 해상도"
              f"({ocams_calib.IMAGE_WIDTH}x{ocams_calib.IMAGE_HEIGHT})와 다름 — rectify 맵을 못 씀, "
              f"raw(왜곡보정 전) 이미지로 진행. docs/03 융합용으로는 기본 해상도를 쓸 것.")
        left_maps = right_maps = None
    else:
        left_maps, right_maps = ocams_calib.build_rectify_maps()

    if ros_src is not None:
        # 토픽이 붙을 때까지 잠깐 기다린다. 첫 프레임이 없으면 해상도 추정을 못 한다.
        for _ in range(50):
            probe_left, probe_right = ros_src.scene_pair()
            if probe_left is not None:
                break
            time.sleep(0.1)
    else:
        probe_left, probe_right = (scene_stereo(scene_cap, left_maps, right_maps)
                                   if scene_cap is not None else (None, None))
    probe = probe_left
    if probe is None:
        if scene_cap is not None:
            scene_cap.release()
            scene_cap = None
        SH, SW = args.scene_height, args.scene_width
        print("[startup] 씬 프레임 없음 — 창을 유지하고 자동 재연결 대기")
    else:
        SH, SW = probe.shape[:2]
    if args.fx:
        fx = fy = args.fx
        cx, cy = SW / 2.0, SH / 2.0
    elif left_maps is not None:
        fx = fy = ocams_calib.RECTIFIED_K[0, 0]
        cx, cy = ocams_calib.RECTIFIED_K[0, 2], ocams_calib.RECTIFIED_K[1, 2]
    else:
        fx = fy = SW * 0.94  # rectify 못 쓰는 경우의 옛 근사치 (해상도 불일치 시)
        cx, cy = SW / 2.0, SH / 2.0
    fx0 = fx  # 리셋 시 되돌아갈 초기값
    stereo_matcher = make_stereo_matcher()
    latest_disparity = None
    latest_depth_m = None
    depth_valid_count = 0
    udp_status = "대기"
    print(f"[scene] {SW}x{SH}  fx={fx:.1f} cx={cx:.1f} cy={cy:.1f} "
          f"({'rectified 캘리브레이션 값' if left_maps is not None and not args.fx else '근사/수동값'})")
    print("[키] f=안구모델 고정/해제 / c=1점 / m=affine 다점(한 거리) / u=마지막 점 취소 / e(=M)=R,p_eye 다점(여러 거리, 6+, docs/12) "
          "/ d=깊이 좌클릭 모드 / x,y=축 반전 / s=저장 / r=리셋 / q=종료")

    R = np.eye(3, dtype=np.float32)
    gaze_affine = None
    calibrated = False
    smooth_dir = None
    calib_path = os.path.abspath(os.path.join(HERE, "..", "calibration", "gaze_scene_affine.json"))
    profile_05_path = os.path.abspath(os.path.join(HERE, "..", "calibration", "gaze_scene_affine_0.5m.json"))
    profile_10_path = os.path.abspath(os.path.join(HERE, "..", "calibration", "gaze_scene_affine_1.0m.json"))
    loaded_05 = load_affine_calibration(profile_05_path, SW, SH)
    loaded_10 = load_affine_calibration(profile_10_path, SW, SH)
    affine_05 = loaded_05[0] if loaded_05 is not None else None
    affine_10 = loaded_10[0] if loaded_10 is not None else None
    if affine_05 is not None and affine_10 is not None:
        print("[depth-calib] 0.5m/1.0m affine 프로필 불러오기 완료")
    loaded_calib = load_affine_calibration(calib_path, SW, SH)
    if loaded_calib is not None:
        gaze_affine, loaded_meta = loaded_calib
        calibrated = True
        print(f"[calib] affine 자동 불러오기: {calib_path} "
              f"(samples={loaded_meta.get('sample_count')}, "
              f"error={loaded_meta.get('mean_pixel_error', float('nan')):.1f}px)")
        eye_model = loaded_meta.get("eye_model")
        if eye_model and args.restore_eye_model:
            restore_eye_model(eye_model)
            print(f"[calib] 안구 모델 복원·고정: 중심={eye_model['center']} "
                  f"— 안경을 다시 썼다면 틀린다. 'f' 로 풀고 새로 캘리브할 것.")
        elif eye_model:
            print("[calib] 파일에 안구 모델이 있지만 복원 안 함(기본). 이 affine 은 새 세션의 "
                  "안구 모델과 안 맞으니 새로 캘리브하거나 --restore-eye-model 을 쓸 것.")
        else:
            print("[calib] 파일에 안구 모델 없음 — 이 affine 은 새 세션에서 정확하지 않다. "
                  "새로 캘리브할 것.")

    multi_mode = False
    calib_dirs = []    # 다점 캘리브: 클릭 순간의 시선벡터들
    calib_pixels = []  # 다점 캘리브: 클릭한 (u,v) 픽셀들
    click_state = {"pending": None, "depth_pending": None, "extrinsic_pending": None}
    depth_mode = False
    depth_probe_result = None

    # docs/12_eye_scene_extrinsic_calibration.md "M" 모드 — 여러 거리(0.5/1.5/3m 등)에서
    # 실제 지점을 클릭하면 (시선방향, 그 픽셀의 스테레오 깊이로 구한 씬 카메라 3D점)을 모아서
    # eye_scene_extrinsic.calibrate_r_p_eye()로 (R, p_eye) 6자유도를 비선형 최소제곱으로 푼다.
    # 기존 'm'(회전+스케일만 푸는 Wahba)과 별개 — docs/12 배경 참고.
    extrinsic_mode = False
    extrinsic_dirs = []
    extrinsic_points = []
    extrinsic_R = None
    extrinsic_p_eye = None
    extrinsic_residuals = None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if extrinsic_mode:
                click_state["extrinsic_pending"] = (x, y)
            elif depth_mode:
                click_state["depth_pending"] = (x, y)
            else:
                click_state["pending"] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            click_state["depth_pending"] = (x, y)

    win_name = "scene (oCamS left) - gaze"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    def rr_log_safe(entity_path, obj):
        """rr.log 실패(gRPC transport error 등)해도 스크립트 전체가 죽지 않게. 실패하면 이후 로깅 자체를 끔."""
        nonlocal rerun_ok
        if not rerun_ok:
            return
        try:
            rr.log(entity_path, obj)
        except Exception as e:
            print(f"[rerun] 로깅 실패 — 이후 Rerun 로깅 끔(cv2 창은 계속 동작): {e}")
            rerun_ok = False

    frame_idx = 0
    next_eye_retry = 0.0
    next_scene_retry = 0.0

    def reconnect_eye():
        """USB 재연결로 /dev/videoN이 바뀌어도 by-id로 눈 카메라를 다시 연다."""
        nonlocal eye_cap, smooth_dir, next_eye_retry
        if ros_src is not None:
            return          # 토픽 모드에서는 로컬 USB 를 열면 안 된다
        now = time.monotonic()
        if eye_cap is not None or now < next_eye_retry:
            return
        next_eye_retry = now + 1.0
        try:
            eye_cap = open_eye(args.eye_width, args.eye_height, args.eye_fps)
            tracker.reset_tracking_state()
            smooth_dir = None
            print("[reconnect] 눈 카메라 재연결 완료 — 안구모델을 다시 수렴시키세요.")
        except (OSError, RuntimeError) as e:
            print(f"[reconnect] 눈 카메라 대기 중: {e}")

    def reconnect_scene():
        """USB 재연결로 /dev/videoN이 바뀌어도 by-id로 씬 카메라를 다시 연다."""
        nonlocal scene_cap, next_scene_retry
        if ros_src is not None:
            return          # 토픽 모드에서는 로컬 USB 를 열면 안 된다
        now = time.monotonic()
        if scene_cap is not None or now < next_scene_retry:
            return
        next_scene_retry = now + 1.0
        try:
            scene_cap = open_scene(args.scene_width, args.scene_height)
            print("[reconnect] 씬 카메라 재연결 완료.")
        except (OSError, RuntimeError) as e:
            print(f"[reconnect] 씬 카메라 대기 중: {e}")

    try:
        while True:
            # rclpy 는 SIGTERM 을 가로채 컨텍스트만 내린다. 안 보면 kill 로 안 죽는다.
            if ros_src is not None and not ros_src._rclpy.ok():
                print("[ros] 종료 신호 — 루프 종료")
                break
            reconnect_eye()
            reconnect_scene()

            scene = None
            if ros_src is not None:
                # Pi 가 이미 분리·rectify 해서 보낸다. 여기서 또 하면 안 된다.
                left_gray, right_gray = ros_src.scene_pair()
                if left_gray is None or right_gray is None:
                    if frame_idx % 60 == 0:
                        print("[ros] 씬 토픽 대기 중 — Pi 의 ocams.sh 가 떠 있는지 확인")
                else:
                    scene = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
                    if frame_idx % 3 == 0 and (
                            depth_mode or args.enable_experimental_depth or extrinsic_mode
                            or args.send_udp):
                        latest_disparity = (
                            stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0)
                    if args.scene_flip:
                        scene = cv2.flip(scene, -1)
            elif scene_cap is not None:
                left_gray, right_gray = scene_stereo(scene_cap, left_maps, right_maps)
                if left_gray is None or right_gray is None:
                    print("[disconnect] 씬 카메라 프레임 끊김 — 자동 재연결 대기")
                    scene_cap.release()
                    scene_cap = None
                    next_scene_retry = 0.0
                else:
                    scene = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
                    if frame_idx % 3 == 0 and (
                            depth_mode or args.enable_experimental_depth or extrinsic_mode):
                        latest_disparity = (
                            stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0)
                    if args.scene_flip:
                        scene = cv2.flip(scene, -1)

            eye = None
            if eye_cap is not None:
                ok, eye = eye_cap.read()
                if (not ok or eye is None) and ros_src is not None:
                    # 토픽 모드: 이번 프레임만 건너뛴다. shim 을 버리면 reconnect_eye 가
                    # 로컬 USB 를 열려다 영원히 실패하고, 토픽이 돌아와도 복구되지 않는다.
                    eye = None
                    if frame_idx % 60 == 0:
                        print(f"[ros] 눈 토픽 대기 중 — {ros_src.eye_topic}")
                elif not ok or eye is None:
                    print("[disconnect] 눈 카메라 프레임 끊김 — 자동 재연결 대기")
                    eye_cap.release()
                    eye_cap = None
                    smooth_dir = None
                    next_eye_retry = 0.0

            # 눈 카메라가 없어도(2026-08-29: 케이블 파손으로 당일 미보유) 씬 카메라만으로
            # 스테레오 깊이 검증(d/우클릭 모드)은 계속 돌아가야 한다 — 예전에는 여기서
            # continue로 매 프레임 건너뛰어서 depth_pending 처리(아래)와 'd' 키 토글에
            # 도달할 수조차 없었다. eye 관련 계산만 건너뛰고 나머지는 그대로 진행한다.
            log_images_this_frame = rerun_ok and (frame_idx % args.rerun_every == 0)
            if eye is not None:
                if args.flip:
                    eye = cv2.flip(eye, -1)     # 180도
                elif args.flip_v:
                    eye = cv2.flip(eye, 0)      # 상하
                elif args.flip_h:
                    eye = cv2.flip(eye, 1)      # 좌우

                if rerun_ok:
                    rr.set_time("frame", sequence=frame_idx)
                    if log_images_this_frame:
                        rr_log_safe("eye/ir", rr.Image(cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)))

                _ellipse, d = tracker.process_frame(eye)  # 검출 + 안구모델 + 눈 창 표시, 반환값으로 시선벡터 바로 받음

                if d is not None and np.linalg.norm(d) > 1e-6:
                    d = (d / np.linalg.norm(d)) * sign
                    smooth_dir = d if smooth_dir is None else \
                        (1 - args.smooth) * smooth_dir + args.smooth * d
                    smooth_dir /= np.linalg.norm(smooth_dir)
            elif rerun_ok:
                rr.set_time("frame", sequence=frame_idx)

            if scene is None:
                waiting = np.zeros((SH, SW, 3), dtype=np.uint8)
                cv2.putText(waiting, "SCENE CAMERA DISCONNECTED - waiting for reconnect",
                            (20, SH // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.imshow(win_name, waiting)
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
                continue

            if log_images_this_frame:
                rr_log_safe("scene/image", rr.Image(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)))
            if rerun_ok and calib_pixels:
                rr_log_safe("scene/calib_points",
                            rr.Points2D(calib_pixels, radii=8, colors=[255, 0, 255]))

            if click_state["depth_pending"] is not None:
                du, dv = click_state["depth_pending"]
                click_state["depth_pending"] = None
                measured_depth, valid_n = depth_at(
                    latest_disparity, du, dv, ocams_calib.RECTIFIED_K[0, 0],
                    baseline_m, radius=10)
                depth_probe_result = (du, dv, measured_depth, valid_n)
                if measured_depth is None:
                    print(f"[depth] ({du},{dv}) 측정 실패 — 유효 disparity {valid_n}개")
                else:
                    print(f"[depth] ({du},{dv}) = {measured_depth:.3f}m "
                          f"(유효 disparity {valid_n}개)")

            if click_state["extrinsic_pending"] is not None:
                eu, ev = click_state["extrinsic_pending"]
                click_state["extrinsic_pending"] = None
                if smooth_dir is None:
                    print("[R,p_eye] 아직 시선벡터가 없다 — 눈을 굴려 모델을 세우고 다시.")
                else:
                    D, valid_n = depth_at(
                        latest_disparity, eu, ev, ocams_calib.RECTIFIED_K[0, 0],
                        baseline_m, radius=5)
                    if D is None or not (0.1 < D < 5.0):
                        print(f"[R,p_eye] ({eu},{ev}) 깊이 측정 실패(유효 disparity {valid_n}개) "
                              "— 텍스처 있는 곳을 다시 클릭.")
                    else:
                        Kinv = np.linalg.inv(ocams_calib.RECTIFIED_K)
                        X = D * (Kinv @ np.array([eu, ev, 1.0], dtype=np.float64))
                        extrinsic_dirs.append(smooth_dir.copy())
                        extrinsic_points.append(X)
                        print(f"[R,p_eye] 포인트 추가 #{len(extrinsic_dirs)}: "
                              f"D={D:.3f}m X={np.round(X, 3)}")
                        if len(extrinsic_dirs) >= 6:
                            try:
                                extrinsic_R, extrinsic_p_eye, extrinsic_residuals = \
                                    eye_scene_extrinsic.calibrate_r_p_eye(
                                        extrinsic_dirs, extrinsic_points)
                                print(f"[R,p_eye] {len(extrinsic_dirs)}점으로 재계산. "
                                      f"p_eye={np.round(extrinsic_p_eye, 4)}m 잔차(cm): "
                                      f"평균={extrinsic_residuals.mean()*100:.1f} "
                                      f"최대={extrinsic_residuals.max()*100:.1f}")
                            except ValueError as e:
                                print(f"[R,p_eye] 계산 대기: {e}")
                        else:
                            print(f"[R,p_eye] {len(extrinsic_dirs)}/6점 — 최소제곱 계산 대기")

            if click_state["pending"] is not None:
                cu, cv_ = click_state["pending"]
                click_state["pending"] = None
                if not multi_mode:
                    print("[다점] 'm'으로 다점 캘리브 모드를 먼저 켜세요.")
                elif smooth_dir is None:
                    print("[다점] 아직 시선벡터가 없다 — 눈을 굴려 모델을 세우고 다시.")
                else:
                    calib_dirs.append(smooth_dir.copy())
                    calib_pixels.append((cu, cv_))
                    print(f"[다점] 포인트 추가 #{len(calib_dirs)}: 시선={smooth_dir.round(3)} <-> 픽셀=({cu},{cv_})")
                    if tracker.eye_sphere_adjustment_enabled:
                        print("[다점][주의] 안구 모델이 자동 갱신 중 — 'f' 로 고정하지 않으면 "
                              "캘리브 도중 시선 기준이 움직인다.")
                    if len(calib_dirs) >= 3:
                        try:
                            gaze_affine = calibrate_affine(calib_dirs, calib_pixels)
                            calibrated = True
                            px_err = np.sqrt(affine_reprojection_error(
                                gaze_affine, calib_dirs, calib_pixels) / len(calib_dirs))
                            print(f"[다점-affine] {len(calib_dirs)}점으로 재계산. "
                                  f"평균 픽셀오차={px_err:.1f}px")
                            if len(calib_dirs) >= 5:
                                print_calib_report(gaze_affine, calib_dirs, calib_pixels)
                            if len(calib_dirs) >= 9:
                                # 2026-09-23: 17.7px 결과 뒤에 깜빡임 점 하나(142px)가
                                # 들어가 55.8px 로 자동 덮어쓴 일이 있었다.
                                bad, last_err = last_point_is_outlier(
                                    gaze_affine, calib_dirs, calib_pixels)
                                if bad:
                                    print(f"[calib] 마지막 점이 불량({last_err:.0f}px) — "
                                          f"자동 저장 안 함. 'u' 로 되돌리기.")
                                else:
                                    save_affine_calibration(
                                        calib_path, gaze_affine, calib_dirs, calib_pixels, SW, SH)
                                    print(f"[calib] 9점 이상 자동 저장: {calib_path}")
                        except ValueError as e:
                            print(f"[다점-affine] 계산 대기: {e}")
                    else:
                        print(f"[다점-affine] {len(calib_dirs)}/3점 — affine 계산 대기")

            n_model = len(getattr(tracker, "model_centers", []))
            if calibrated and smooth_dir is not None:
                gaze_valid = True
                if (args.enable_experimental_depth
                        and affine_05 is not None and affine_10 is not None):
                    depth_guess = latest_depth_m if latest_depth_m is not None else 0.75
                    measured_depth = None
                    for _ in range(2):
                        active_affine = interpolate_affine(depth_guess, affine_05, affine_10)
                        uv = active_affine @ gaze_features(smooth_dir)
                        if not np.all(np.isfinite(uv)):
                            gaze_valid = False
                            break
                        u = int(np.clip(uv[0], 0, SW - 1))
                        v = int(np.clip(uv[1], 0, SH - 1))
                        measured_depth, depth_valid_count = depth_at(
                            latest_disparity, u, v, ocams_calib.RECTIFIED_K[0, 0],
                            baseline_m, radius=5)
                        if measured_depth is not None and 0.2 <= measured_depth <= 2.0:
                            depth_guess = measured_depth
                    if measured_depth is not None and 0.2 <= measured_depth <= 2.0:
                        latest_depth_m = measured_depth
                elif gaze_affine is not None:
                    uv = gaze_affine @ gaze_features(smooth_dir)
                    gaze_valid = bool(np.all(np.isfinite(uv)))
                    if gaze_valid:
                        u = int(np.clip(uv[0], 0, SW - 1))
                        v = int(np.clip(uv[1], 0, SH - 1))
                else:
                    g = R @ smooth_dir
                    gaze_valid = g[2] > 1e-6
                    if gaze_valid:
                        u = int(np.clip(cx + fx * (g[0] / g[2]), 0, SW - 1))
                        v = int(np.clip(cy - fy * (g[1] / g[2]), 0, SH - 1))

                if udp_sender is not None:
                    latest_depth_m, udp_status = fuse_and_send(
                        udp_sender, ros_src, gaze_valid, u if gaze_valid else None,
                        v if gaze_valid else None, latest_disparity, baseline_m,
                        latest_depth_m, args.scene_flip, SW, SH)

                if px_sender is not None:
                    if gaze_valid:
                        # --scene-flip 은 표시용. 팔 쪽은 뒤집지 않은 /camera/left 를 본다.
                        pu, pv = (SW - 1 - u, SH - 1 - v) if args.scene_flip else (u, v)
                        px_sender.send(pu, pv, valid=True)
                    else:
                        px_sender.send(0, 0, valid=False)

                if gaze_valid:
                    cv2.circle(scene, (u, v), 28, (0, 255, 0), 3)
                    cv2.drawMarker(scene, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
                    depth_label = (f" depth={latest_depth_m:.2f}m"
                                   if latest_depth_m is not None else " depth=N/A")
                    cv2.putText(scene, f"gaze ({u},{v}){depth_label}", (u + 34, v - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    if rerun_ok:
                        rr_log_safe("scene/gaze_cursor",
                                    rr.Points2D([[u, v]], radii=12, colors=[0, 255, 0]))
                else:
                    cv2.putText(scene, "invalid gaze", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if (args.enable_experimental_depth
                        and affine_05 is not None and affine_10 is not None):
                    status = "CALIBRATED DEPTH 0.5-1.0m"
                else:
                    status = "CALIBRATED AFFINE" if gaze_affine is not None else "CALIBRATED R"
                color = (0, 255, 0)
            else:
                if px_sender is not None:
                    px_sender.send(0, 0, valid=False)
                if udp_sender is not None:
                    # 캘리브 전이거나 동공을 놓친 상태. 무소식보다 "지금은 못 믿는다"를
                    # 명시적으로 보내야 로봇팔 쪽이 마지막 유효점을 붙들지 않는다.
                    udp_sender.send([0.0, 0.0, 0.0], valid=False)
                    udp_status = "미캘리브" if not calibrated else "동공놓침"
                status, color = "NOT CALIBRATED - look at scene cam, press 'c'", (0, 200, 255)

            if depth_probe_result is not None:
                du, dv, probe_z, probe_n = depth_probe_result
                probe_color = (0, 255, 255) if probe_z is not None else (0, 0, 255)
                probe_text = (f"{probe_z:.3f}m n={probe_n}" if probe_z is not None
                              else f"depth N/A n={probe_n}")
                cv2.drawMarker(scene, (du, dv), probe_color, cv2.MARKER_CROSS, 24, 2)
                cv2.putText(scene, probe_text, (du + 12, max(20, dv - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, probe_color, 2)

            for i, (pu, pv) in enumerate(calib_pixels):
                cv2.drawMarker(scene, (pu, pv), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
                cv2.putText(scene, str(i + 1), (pu + 10, pv - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            mirror = f"mirror x={'ON' if sign[0] < 0 else 'off'} y={'ON' if sign[1] < 0 else 'off'}"
            # status 가 길면(NOT CALIBRATED ...) model_centers 가 화면 밖으로
            # 밀려서 안 보였다. 같은 줄 오른쪽에 따로 붙여 항상 보이게 한다.
            cv2.putText(scene, status, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            mc_text = f"mc={n_model}"
            (mc_w, _), _ = cv2.getTextSize(mc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            mc_color = (0, 255, 0) if n_model >= 30 else (0, 165, 255)   # 30 미만이면 주황
            cv2.putText(scene, mc_text, (SW - mc_w - 12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, mc_color, 2)
            # 안구 구 모델이 자동 갱신 중이면 캘리브 도중에도 시선 벡터 기준이 움직인다.
            sphere_locked = not tracker.eye_sphere_adjustment_enabled
            sp_text = "sphere=LOCKED" if sphere_locked else "sphere=AUTO (F)"
            (sp_w, _), _ = cv2.getTextSize(sp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(scene, sp_text, (SW - sp_w - 12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if sphere_locked else (0, 165, 255), 2)
            cv2.putText(scene, f"{mirror}  (x/y=toggle)", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            multi_color = (0, 255, 255) if multi_mode else (150, 150, 150)
            mode_text = "R,P_EYE CALIB" if extrinsic_mode else (
                "DEPTH LEFT-CLICK" if depth_mode else (
                    "MULTI CALIB" if multi_mode else "TRACKING"))
            cv2.putText(scene, f"mode={mode_text} points={len(calib_pixels)} "
                                f"map={'affine' if gaze_affine is not None else 'rotation'}", (20, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, multi_color, 2)
            if extrinsic_mode or extrinsic_p_eye is not None:
                ex_text = f"R,p_eye points={len(extrinsic_dirs)}/6+"
                if extrinsic_p_eye is not None:
                    ex_text += (f" p_eye={np.round(extrinsic_p_eye, 3)}m "
                                f"resid(cm) mean={extrinsic_residuals.mean()*100:.1f} "
                                f"max={extrinsic_residuals.max()*100:.1f}")
                cv2.putText(scene, ex_text, (20, 114),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if udp_sender is not None:
                # 초록 = 로봇팔로 실제 좌표가 나가는 중. 그 외는 왜 안 나가는지 이유를 띄운다.
                udp_color = (0, 255, 0) if udp_status.startswith("송신") else (0, 165, 255)
                cv2.putText(scene, f"UDP {udp_status}  [{udp_sender.sent}/{udp_sender.dropped}]",
                            (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.55, udp_color, 2)
            if eye_cap is None:
                cv2.putText(scene, "EYE CAMERA DISCONNECTED (scene-only depth test OK)",
                            (20, SH - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            cv2.imshow(win_name, scene)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('f'), ord('F')):
                # 트래커 자체 루프에만 있던 키. 여기 없어서 그동안 F 가 무시됐다.
                tracker.toggle_eye_sphere_adjustment()
            elif key == ord('c'):
                if smooth_dir is None:
                    print("[calib] 아직 시선벡터가 없다 — 눈을 굴려 모델을 세우고 다시.")
                elif n_model < 30:
                    print(f"[calib] 안구모델 표본 부족(model_centers={n_model}). "
                          f"눈을 상하좌우로 더 굴린 뒤 다시 (30+ 권장).")
                else:
                    R = rotation_from_a_to_b(smooth_dir, np.array([0.0, 0.0, 1.0], np.float32))
                    gaze_affine = None
                    calibrated = True
                    print(f"[calib] 완료. 기준 시선={smooth_dir.round(3)} → 씬 정면 [0,0,1]")
            elif key == ord('d'):
                depth_mode = not depth_mode
                if depth_mode:
                    multi_mode = False
                print(f"[depth] 좌클릭 깊이 모드 {'ON' if depth_mode else 'OFF'}")
            elif key == ord('m'):
                multi_mode = not multi_mode
                print(f"[다점] 모드 {'ON — 실제 지점을 응시한 채 그 위치를 클릭' if multi_mode else 'OFF'}")
            # 'e' 는 'M' 과 같은 기능. Qt 백엔드에서 Shift 조합이 waitKey 에
            # 안 잡히는 경우가 있어 Shift 없이 쓸 수 있는 키를 하나 더 뒀다.
            elif key in (ord('M'), ord('e')):
                extrinsic_mode = not extrinsic_mode
                if extrinsic_mode:
                    depth_mode = False
                    multi_mode = False
                print(f"[R,p_eye] 캘리브 모드 "
                      f"{'ON — 다른 거리(0.5/1.5/3m 등)에서 실제 지점을 응시한 채 클릭' if extrinsic_mode else 'OFF'}")
            elif key == ord('s'):
                if gaze_affine is None or len(calib_dirs) < 3:
                    print("[calib] 저장할 affine 다점 데이터가 없습니다.")
                elif last_point_is_outlier(gaze_affine, calib_dirs, calib_pixels)[0]:
                    print("[calib] 마지막 점이 불량 — 저장 안 함. 'u' 로 되돌린 뒤 다시 's'.")
                else:
                    saved_err = save_affine_calibration(
                        calib_path, gaze_affine, calib_dirs, calib_pixels, SW, SH)
                    print(f"[calib] 저장 완료: {calib_path} (error={saved_err:.1f}px)")
            elif key == ord('u'):
                if not calib_dirs:
                    print("[다점] 되돌릴 점이 없다.")
                else:
                    calib_dirs.pop()
                    du_, dv_ = calib_pixels.pop()
                    print(f"[다점] 마지막 점 ({du_},{dv_}) 삭제 — 남은 {len(calib_dirs)}점")
                    if len(calib_dirs) >= 3:
                        try:
                            gaze_affine = calibrate_affine(calib_dirs, calib_pixels)
                            px_err = np.sqrt(affine_reprojection_error(
                                gaze_affine, calib_dirs, calib_pixels) / len(calib_dirs))
                            print(f"[다점-affine] {len(calib_dirs)}점으로 재계산. "
                                  f"평균 픽셀오차={px_err:.1f}px")
                        except ValueError as e:
                            print(f"[다점-affine] 계산 대기: {e}")
            elif key == ord('r'):
                calibrated = False
                gaze_affine = None
                R = np.eye(3, dtype=np.float32)
                calib_dirs.clear()
                calib_pixels.clear()
                fx = fy = fx0
                extrinsic_dirs.clear()
                extrinsic_points.clear()
                extrinsic_R = None
                extrinsic_p_eye = None
                extrinsic_residuals = None
                print("[calib] 리셋 (다점 캘리브 포인트 + fx + R/p_eye 캘리브 포인트도 초기값으로 복원됨)")
            elif key in (ord('x'), ord('y')):
                # 축 부호를 바꾸면 기존 R은 무효 → 캘리브 리셋 후 다시 'c'
                i = 0 if key == ord('x') else 1
                sign[i] *= -1
                calibrated = False
                gaze_affine = None
                R = np.eye(3, dtype=np.float32)
                calib_dirs.clear()
                calib_pixels.clear()
                fx = fy = fx0
                print(f"[mirror] {'x' if i == 0 else 'y'} 반전 -> {sign[i]:+.0f} "
                      f"(캘리브 리셋됨, 다시 'c' 또는 'm'+클릭)")
    finally:
        if px_sender is not None:
            px_sender.send(0, 0, valid=False)     # 끊길 때 옛 시선을 붙들지 않게
            px_sender.close()
        if udp_sender is not None:
            # 마지막 한 발은 valid=False. 뷰어를 끄면 로봇팔이 옛 좌표를 붙들고
            # 그리로 움직이려 할 수 있다 — 끊길 때 명시적으로 무효화한다.
            try:
                udp_sender.send([0.0, 0.0, 0.0], valid=False)
            except OSError:
                pass
            print(f"[udp] 송신 {udp_sender.sent} / 버림 {udp_sender.dropped}")
            udp_sender.close()
        if ros_src is not None:
            ros_src.shutdown()
        if eye_cap is not None:
            eye_cap.release()
        if scene_cap is not None:
            scene_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
