#!/usr/bin/env python3
"""
창종설 0단계 → 융합 징검다리 — "지금 뭘 보고 있는지"를 씬(외부) 카메라 영상에 표시.

레포의 FrontCameraTracker(Orlosky3DEyeTrackerFrontCamera.py) 아이디어를 리눅스로 옮긴 것.
원본을 그대로 못 쓰는 이유 2가지:
  1) cv2.CAP_MSMF = 윈도우 전용 백엔드. 리눅스는 CAP_V4L2.
  2) 원본 FrontCamera 파일은 numpy2 uint8 overflow 패치가 안 되어 있다(line 86/111).
     → 검출기는 이미 패치·검증된 3DTracker/Orlosky3DEyeTracker.py를 재사용하고,
       회전·투영 로직만 FrontCamera에서 가져온다.

원리 (SLAM 없이도 되는 1점 캘리브레이션 트릭):
  'c'를 누른 순간의 시선방향을 "씬 카메라 정면 [0,0,1]"으로 놓는 회전 R을 구한다
  (Rodrigues). 이후 gaze를 R로 돌려 핀홀 투영 → 씬 영상 위 원으로 표시.
  눈↔씬 카메라 외부파라미터(extrinsic)를 따로 구할 필요가 없어진다 (docs/01의 2D→2D 매핑 정신).

카메라:
  눈  = Sonix UVC IR (by-id 자동탐색)
  씬  = oCamS-1MGN-U. YUYV raw에서 Y채널=왼쪽, UV채널=오른쪽 (스테레오쌍).
        여기선 왼쪽만 쓴다. 노출은 기본값으로 충분(수동으로 올리지 말 것).

사용:
  python gaze_on_scene.py                 # 창 두 개(눈/씬). 눈 굴려 모델 수렴 → 'c'로 캘리브
  키: c=정면 응시 상태에서 캘리브레이션 / r=리셋 / q=종료
"""
import os
import sys
import argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "external", "EyeTracker", "3DTracker")))
os.chdir(HERE)

GAZE_TXT = os.path.join(HERE, "gaze_vector.txt")
if os.path.exists(GAZE_TXT):
    os.remove(GAZE_TXT)

import Orlosky3DEyeTracker as tracker  # noqa: E402  (numpy2 패치본)

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


def read_last_gaze():
    """3DTracker가 매 프레임 쓰는 gaze_vector.txt → direction (3,)"""
    try:
        with open(GAZE_TXT) as f:
            lines = [l.strip() for l in f if l.strip()]
        vals = [float(x) for x in lines[-1].replace(",", " ").split()]
        if len(vals) >= 6:
            return np.array(vals[3:6], dtype=np.float32)
    except Exception:
        pass
    return None


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


def scene_left(cap):
    """oCamS raw YUYV → 왼쪽 영상(Y채널) BGR"""
    ok, f = cap.read()
    if not ok or f is None:
        return None
    if f.ndim == 3 and f.shape[2] == 2:
        left = f[:, :, 0]
    elif f.ndim == 2:
        left = f[:, 0::2]
    else:
        return f  # 이미 BGR
    return cv2.cvtColor(np.ascontiguousarray(left), cv2.COLOR_GRAY2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye-width", type=int, default=640)
    ap.add_argument("--eye-height", type=int, default=480)
    ap.add_argument("--eye-fps", type=int, default=30)
    ap.add_argument("--scene-width", type=int, default=1280)
    ap.add_argument("--scene-height", type=int, default=720)
    ap.add_argument("--fx", type=float, default=None,
                    help="씬 카메라 초점거리(px). 미지정시 폭*0.94 근사 "
                         "(원본은 640폭에 600 가정). 원이 너무 크게/작게 움직이면 조절")
    ap.add_argument("--smooth", type=float, default=0.25,
                    help="시선벡터 EMA 계수(0=고정,1=생값). 7/2 노트의 프레임간 튐(std0.18) 완화")
    ap.add_argument("--flip", action="store_true", help="눈 영상 상하반전")
    ap.add_argument("--no-mirror-x", action="store_true",
                    help="시선 x축 반전을 끈다. 기본은 켬 — 눈 카메라는 사용자를 마주보므로 "
                         "씬 카메라와 좌우가 뒤집힌다(1점 캘리브의 최소회전으로는 못 고침)")
    ap.add_argument("--mirror-y", action="store_true", help="시선 y축도 반전(상하가 뒤집힐 때)")
    args = ap.parse_args()

    # 눈/씬 카메라가 서로 마주보는 데서 오는 축 반전. 런타임에 x/y 키로 토글 가능.
    sign = np.array([1.0 if args.no_mirror_x else -1.0,
                     -1.0 if args.mirror_y else 1.0,
                     1.0], dtype=np.float32)

    eye_cap = open_eye(args.eye_width, args.eye_height, args.eye_fps)
    scene_cap = open_scene(args.scene_width, args.scene_height)

    probe = scene_left(scene_cap)
    if probe is None:
        raise RuntimeError("씬 프레임 없음")
    SH, SW = probe.shape[:2]
    fx = fy = args.fx if args.fx else SW * 0.94
    cx, cy = SW / 2.0, SH / 2.0
    print(f"[scene] {SW}x{SH}  fx={fx:.0f} (근사값 — 원 움직임이 과하면 --fx 로 조절)")
    print("[키] c=캘리브레이션(씬 카메라 정면을 응시한 상태에서) / r=리셋 / q=종료")

    R = np.eye(3, dtype=np.float32)
    calibrated = False
    smooth_dir = None

    try:
        while True:
            ok, eye = eye_cap.read()
            if not ok:
                continue
            if args.flip:
                eye = cv2.flip(eye, -1)

            tracker.process_frame(eye)  # 검출 + 안구모델 + gaze_vector.txt 갱신 + 눈 창 표시
            d = read_last_gaze()

            if d is not None and np.linalg.norm(d) > 1e-6:
                d = (d / np.linalg.norm(d)) * sign
                smooth_dir = d if smooth_dir is None else \
                    (1 - args.smooth) * smooth_dir + args.smooth * d
                smooth_dir /= np.linalg.norm(smooth_dir)

            scene = scene_left(scene_cap)
            if scene is None:
                continue

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
                else:
                    cv2.putText(scene, "gaze behind camera", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                status, color = "CALIBRATED", (0, 255, 0)
            else:
                status, color = "NOT CALIBRATED - look at scene cam, press 'c'", (0, 200, 255)

            mirror = f"mirror x={'ON' if sign[0] < 0 else 'off'} y={'ON' if sign[1] < 0 else 'off'}"
            cv2.putText(scene, f"{status} | model_centers={n_model}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(scene, f"{mirror}  (x/y=toggle)", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow("scene (oCamS left) - gaze", scene)

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
            elif key == ord('r'):
                calibrated = False
                R = np.eye(3, dtype=np.float32)
                print("[calib] 리셋")
            elif key in (ord('x'), ord('y')):
                # 축 부호를 바꾸면 기존 R은 무효 → 캘리브 리셋 후 다시 'c'
                i = 0 if key == ord('x') else 1
                sign[i] *= -1
                calibrated = False
                R = np.eye(3, dtype=np.float32)
                print(f"[mirror] {'x' if i == 0 else 'y'} 반전 -> {sign[i]:+.0f} "
                      f"(캘리브 리셋됨, 다시 'c')")
    finally:
        eye_cap.release()
        scene_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
