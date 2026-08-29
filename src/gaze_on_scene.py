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


def gaze_features(direction):
    """시선 단위벡터를 원근 정규화 좌표 (x/z, y/z, 1)로 변환."""
    d = np.asarray(direction, dtype=np.float64)
    z = d[2]
    if abs(z) < 1e-6:
        z = 1e-6 if z >= 0 else -1e-6
    return np.array([d[0] / z, d[1] / z, 1.0], dtype=np.float64)


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


def save_affine_calibration(path, affine, dirs, pixels, width, height):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    px_error = np.sqrt(affine_reprojection_error(affine, dirs, pixels) / len(dirs))
    payload = {
        "version": 1,
        "model": "gaze_direction_to_scene_pixel_affine",
        "affine_2x3": np.asarray(affine).tolist(),
        "sample_count": len(dirs),
        "mean_pixel_error": float(px_error),
        "image_width": int(width),
        "image_height": int(height),
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


def depth_at(disparity, u, v, fx, baseline_m, radius=3, min_valid=5):
    if disparity is None:
        return None, 0
    h, w = disparity.shape
    u, v = int(u), int(v)
    x1, x2 = max(0, u - radius), min(w, u + radius + 1)
    y1, y2 = max(0, v - radius), min(h, v + radius + 1)
    roi = disparity[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    if len(valid) < min_valid:
        return None, int(len(valid))
    disp = float(np.median(valid))
    return float(fx * baseline_m / disp), int(len(valid))


def interpolate_affine(depth_m, affine_05, affine_10):
    alpha = float(np.clip((depth_m - 0.5) / 0.5, 0.0, 1.0))
    return ((1.0 - alpha) * affine_05 + alpha * affine_10).astype(np.float32)


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
                    help="스테레오 baseline(m) 강제 지정, 진단용. 미지정시 ocams_calib.BASELINE_M "
                         "(Kalibr 계산값, 실측 아님). 2026-08-29: 실측 baseline(캘리퍼스 12cm)이 "
                         "8/11 재캘리브레이션 값(10.67cm)과 달라 깊이가 실제보다 짧게 나오는 문제 "
                         "진단 중 — rectification(R1/R2)은 그대로 두고 이 상수만 바꿔서 baseline만 "
                         "따로 검증하기 위해 추가.")
    ap.add_argument("--smooth", type=float, default=0.25,
                    help="시선벡터 EMA 계수(0=고정,1=생값). 7/2 노트의 프레임간 튐(std0.18) 완화")
    ap.add_argument("--flip", action="store_true", help="눈 영상 상하반전")
    ap.add_argument("--scene-flip", action="store_true", help="씬 카메라(oCamS) 180도 반전 (카메라가 거꾸로 장착된 경우)")
    ap.add_argument("--enable-experimental-depth", action="store_true",
                    help="검증되지 않은 스테레오 깊이 기반 0.5m/1.0m affine 보간 사용")
    ap.add_argument("--no-mirror-x", action="store_true",
                    help="시선 x축 반전을 끈다. 기본은 켬 — 눈 카메라는 사용자를 마주보므로 "
                         "씬 카메라와 좌우가 뒤집힌다(1점 캘리브의 최소회전으로는 못 고침)")
    ap.add_argument("--mirror-y", action="store_true", help="시선 y축도 반전(상하가 뒤집힐 때)")
    ap.add_argument("--no-rerun", action="store_true", help="Rerun 로깅 끄기 (cv2 창만 사용)")
    ap.add_argument("--rerun-every", type=int, default=3,
                    help="Rerun에 영상을 N프레임마다 1번만 로깅 (기본 3). 매 프레임 원본 "
                         "해상도로 다 보내면 gRPC 버퍼(1GiB)가 몇 분 안에 차서 뷰어가 죽는다 "
                         "(2026-07-30에 실제로 겪음) — cv2 창 표시/캘리브는 영향 없음, "
                         "Rerun 쪽 영상만 덜 자주 보냄")
    args = ap.parse_args()

    baseline_m = args.baseline_m if args.baseline_m is not None else ocams_calib.BASELINE_M
    if args.baseline_m is not None:
        print(f"[baseline] 강제 지정: {baseline_m:.4f}m (캘리브레이션값 {ocams_calib.BASELINE_M:.4f}m 대신)")

    if args.enable_experimental_depth:
        print("[경고] 스테레오 깊이는 현재 검증 실패 상태입니다. 로봇팔 제어에 사용하지 마세요.")

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

    if (args.scene_width, args.scene_height) != (ocams_calib.IMAGE_WIDTH, ocams_calib.IMAGE_HEIGHT):
        print(f"[경고] --scene-width/height가 캘리브레이션 해상도"
              f"({ocams_calib.IMAGE_WIDTH}x{ocams_calib.IMAGE_HEIGHT})와 다름 — rectify 맵을 못 씀, "
              f"raw(왜곡보정 전) 이미지로 진행. docs/03 융합용으로는 기본 해상도를 쓸 것.")
        left_maps = right_maps = None
    else:
        left_maps, right_maps = ocams_calib.build_rectify_maps()

    probe_left, probe_right = scene_stereo(scene_cap, left_maps, right_maps) if scene_cap is not None else (None, None)
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
    print(f"[scene] {SW}x{SH}  fx={fx:.1f} cx={cx:.1f} cy={cy:.1f} "
          f"({'rectified 캘리브레이션 값' if left_maps is not None and not args.fx else '근사/수동값'})")
    print("[키] c=1점 / m=affine 다점 / M=R,p_eye 최소제곱 다점(6+, docs/12) "
          "/ d=깊이 좌클릭 모드 / s=저장 / r=리셋 / q=종료")

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
            reconnect_eye()
            reconnect_scene()

            scene = None
            if scene_cap is not None:
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
                if not ok or eye is None:
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
                    eye = cv2.flip(eye, -1)

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
                    if len(calib_dirs) >= 3:
                        try:
                            gaze_affine = calibrate_affine(calib_dirs, calib_pixels)
                            calibrated = True
                            px_err = np.sqrt(affine_reprojection_error(
                                gaze_affine, calib_dirs, calib_pixels) / len(calib_dirs))
                            print(f"[다점-affine] {len(calib_dirs)}점으로 재계산. "
                                  f"평균 픽셀오차={px_err:.1f}px")
                            if len(calib_dirs) >= 9:
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
            cv2.putText(scene, f"{status} | model_centers={n_model}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
            if eye_cap is None:
                cv2.putText(scene, "EYE CAMERA DISCONNECTED (scene-only depth test OK)",
                            (20, SH - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            cv2.imshow(win_name, scene)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
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
            elif key == ord('M'):
                extrinsic_mode = not extrinsic_mode
                if extrinsic_mode:
                    depth_mode = False
                    multi_mode = False
                print(f"[R,p_eye] 캘리브 모드 "
                      f"{'ON — 다른 거리(0.5/1.5/3m 등)에서 실제 지점을 응시한 채 클릭' if extrinsic_mode else 'OFF'}")
            elif key == ord('s'):
                if gaze_affine is None or len(calib_dirs) < 3:
                    print("[calib] 저장할 affine 다점 데이터가 없습니다.")
                else:
                    saved_err = save_affine_calibration(
                        calib_path, gaze_affine, calib_dirs, calib_pixels, SW, SH)
                    print(f"[calib] 저장 완료: {calib_path} (error={saved_err:.1f}px)")
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
        if eye_cap is not None:
            eye_cap.release()
        if scene_cap is not None:
            scene_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
