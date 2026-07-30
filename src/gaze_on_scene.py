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
import numpy as np
import cv2
import rerun as rr

import ocams_calib

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
    ap.add_argument("--smooth", type=float, default=0.25,
                    help="시선벡터 EMA 계수(0=고정,1=생값). 7/2 노트의 프레임간 튐(std0.18) 완화")
    ap.add_argument("--flip", action="store_true", help="눈 영상 상하반전")
    ap.add_argument("--scene-flip", action="store_true", help="씬 카메라(oCamS) 180도 반전 (카메라가 거꾸로 장착된 경우)")
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

    if not args.no_rerun:
        rr.init("molbwa_gaze_on_scene", spawn=True)
    rerun_ok = not args.no_rerun  # gRPC 전송 에러 나면 죽이지 말고 이후로는 꺼버림

    # 눈/씬 카메라가 서로 마주보는 데서 오는 축 반전. 런타임에 x/y 키로 토글 가능.
    sign = np.array([1.0 if args.no_mirror_x else -1.0,
                     -1.0 if args.mirror_y else 1.0,
                     1.0], dtype=np.float32)

    eye_cap = open_eye(args.eye_width, args.eye_height, args.eye_fps)
    scene_cap = open_scene(args.scene_width, args.scene_height)

    if (args.scene_width, args.scene_height) != (ocams_calib.IMAGE_WIDTH, ocams_calib.IMAGE_HEIGHT):
        print(f"[경고] --scene-width/height가 캘리브레이션 해상도"
              f"({ocams_calib.IMAGE_WIDTH}x{ocams_calib.IMAGE_HEIGHT})와 다름 — rectify 맵을 못 씀, "
              f"raw(왜곡보정 전) 이미지로 진행. docs/03 융합용으로는 기본 해상도를 쓸 것.")
        left_maps = None
    else:
        left_maps, _right_maps_unused = ocams_calib.build_rectify_maps()

    probe = scene_left(scene_cap, left_maps)
    if probe is None:
        raise RuntimeError("씬 프레임 없음")
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
    print(f"[scene] {SW}x{SH}  fx={fx:.1f} cx={cx:.1f} cy={cy:.1f} "
          f"({'rectified 캘리브레이션 값' if left_maps is not None and not args.fx else '근사/수동값'})")
    print("[키] c=1점 캘리브 / m=다점 캘리브 모드 토글(클릭으로 포인트 추가, 3점+ 시 fx도 자동보정) / r=리셋 / q=종료")

    R = np.eye(3, dtype=np.float32)
    calibrated = False
    smooth_dir = None

    multi_mode = False
    calib_dirs = []    # 다점 캘리브: 클릭 순간의 시선벡터들
    calib_pixels = []  # 다점 캘리브: 클릭한 (u,v) 픽셀들
    click_state = {"pending": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state["pending"] = (x, y)

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
    try:
        while True:
            ok, eye = eye_cap.read()
            if not ok:
                continue
            if args.flip:
                eye = cv2.flip(eye, -1)

            log_images_this_frame = rerun_ok and (frame_idx % args.rerun_every == 0)
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

            scene = scene_left(scene_cap, left_maps)
            if scene is None:
                continue
            if args.scene_flip:
                scene = cv2.flip(scene, -1)

            if log_images_this_frame:
                rr_log_safe("scene/image", rr.Image(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)))
            if rerun_ok and calib_pixels:
                rr_log_safe("scene/calib_points",
                            rr.Points2D(calib_pixels, radii=8, colors=[255, 0, 255]))

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
                    if len(calib_dirs) >= 2:
                        R, fx_new = calibrate_multi(calib_dirs, calib_pixels, fx, cx, cy)
                        fx = fy = fx_new
                        calibrated = True
                        px_err = np.sqrt(reprojection_error(fx, calib_dirs, calib_pixels, R, cx, cy)
                                          / len(calib_dirs))
                        fx_note = "고정(점<3)" if len(calib_dirs) < 3 else "최적화됨"
                        print(f"[다점] {len(calib_dirs)}점으로 재계산. fx={fx:.0f}({fx_note}) "
                              f"평균 픽셀오차={px_err:.1f}px")

            n_model = len(getattr(tracker, "model_centers", []))
            if calibrated and smooth_dir is not None:
                g = R @ smooth_dir
                if g[2] > 1e-6:
                    u = int(np.clip(cx + fx * (g[0] / g[2]), 0, SW - 1))
                    v = int(np.clip(cy - fy * (g[1] / g[2]), 0, SH - 1))
                    cv2.circle(scene, (u, v), 28, (0, 255, 0), 3)
                    cv2.drawMarker(scene, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
                    cv2.putText(scene, f"gaze ({u},{v})", (u + 34, v - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if rerun_ok:
                        rr_log_safe("scene/gaze_cursor",
                                    rr.Points2D([[u, v]], radii=12, colors=[0, 255, 0]))
                else:
                    cv2.putText(scene, "gaze behind camera", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                status, color = "CALIBRATED", (0, 255, 0)
            else:
                status, color = "NOT CALIBRATED - look at scene cam, press 'c'", (0, 200, 255)

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
            cv2.putText(scene, f"multi-calib(m)={'ON - click target' if multi_mode else 'off'} "
                                f"points={len(calib_pixels)}  fx={fx:.0f}", (20, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, multi_color, 2)
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
                    calibrated = True
                    print(f"[calib] 완료. 기준 시선={smooth_dir.round(3)} → 씬 정면 [0,0,1]")
            elif key == ord('m'):
                multi_mode = not multi_mode
                print(f"[다점] 모드 {'ON — 실제 지점을 응시한 채 그 위치를 클릭' if multi_mode else 'OFF'}")
            elif key == ord('r'):
                calibrated = False
                R = np.eye(3, dtype=np.float32)
                calib_dirs.clear()
                calib_pixels.clear()
                fx = fy = fx0
                print("[calib] 리셋 (다점 캘리브 포인트 + fx도 초기값으로 복원됨)")
            elif key in (ord('x'), ord('y')):
                # 축 부호를 바꾸면 기존 R은 무효 → 캘리브 리셋 후 다시 'c'
                i = 0 if key == ord('x') else 1
                sign[i] *= -1
                calibrated = False
                R = np.eye(3, dtype=np.float32)
                calib_dirs.clear()
                calib_pixels.clear()
                fx = fy = fx0
                print(f"[mirror] {'x' if i == 0 else 'y'} 반전 -> {sign[i]:+.0f} "
                      f"(캘리브 리셋됨, 다시 'c' 또는 'm'+클릭)")
    finally:
        eye_cap.release()
        scene_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
