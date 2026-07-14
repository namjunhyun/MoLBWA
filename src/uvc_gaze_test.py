#!/usr/bin/env python3
"""
창종설 0단계 — UVC IR 눈카메라(모노, /dev/videoN)를 Orlosky '3D' 시선 트래커에 연결.

realsense_gaze_test.py의 UVC판. RealSense 의존(pyrealsense2) 제거.

Orlosky 검출기는 "눈 하나가 프레임을 꽉 채운 클로즈업"을 가정한다
(= 프레임에서 가장 어두운 덩어리 = 동공). 카메라가 눈에서 멀어 얼굴 전체가 잡히면
머리카락/눈썹/반대쪽 눈을 동공으로 오인한다. 물리적으로 못 붙일 땐 --roi로 눈 주변만
잘라 검출기에 먹인다.

사용:
  python uvc_gaze_test.py --live                   # 라이브 창 (눈 굴려 안구모델 세우기)
  python uvc_gaze_test.py --frames 90 --save-overlay  # 헤드리스: 오버레이 png + 벡터 로그
  python uvc_gaze_test.py --live --roi 0.25,0.1,0.5,0.5   # 중앙 ROI만 크롭해서 검출
     (--roi x,y,w,h = 프레임 대비 비율)
"""
import os
import sys
import argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKER_DIR = os.path.join(HERE, "..", "external", "EyeTracker", "3DTracker")
sys.path.insert(0, os.path.abspath(TRACKER_DIR))

# gaze_vector.txt가 cwd에 쓰이므로 결과를 한곳에 모으려고 cwd를 src로 고정
os.chdir(HERE)
GAZE_TXT = os.path.join(HERE, "gaze_vector.txt")
if os.path.exists(GAZE_TXT):
    os.remove(GAZE_TXT)

import Orlosky3DEyeTracker as tracker  # noqa: E402


BY_ID_DIR = "/dev/v4l/by-id"


def find_cam():
    """USB 재연결마다 /dev/videoN 번호가 바뀐다. by-id의 index0(캡처 노드)을 따라간다."""
    if os.path.isdir(BY_ID_DIR):
        for link in sorted(os.listdir(BY_ID_DIR)):
            if link.endswith("index0"):
                path = os.path.realpath(os.path.join(BY_ID_DIR, link))
                print(f"[cam] by-id 매칭: {link} -> {path}")
                return path
    raise RuntimeError(f"{BY_ID_DIR}에 카메라 없음 — USB 연결 확인")


def open_cam(dev, w, h, fps, fourcc):
    src = find_cam() if dev == "auto" else (
        dev if str(dev).startswith("/dev/") else f"/dev/video{dev}")
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"{src} 열기 실패 (다른 프로세스가 점유 중인지 확인)")
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam] {src} {aw}x{ah} @{cap.get(cv2.CAP_PROP_FPS):.0f} ({fourcc})")
    return cap


def parse_roi(s):
    if not s:
        return None
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise ValueError("--roi 는 x,y,w,h (0~1 비율) 4개")
    return v


def crop_roi(frame, roi):
    """비율 ROI로 자른 뒤 검출기가 기대하는 640x480(4:3)으로 리사이즈."""
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = int((x + rw) * w), int((y + rh) * h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    sub = frame[y0:y1, x0:x1]
    if sub.size == 0:
        return frame
    return cv2.resize(sub, (640, 480), interpolation=cv2.INTER_LINEAR)


def read_last_gaze():
    """gaze_vector.txt 마지막 줄 → (origin xyz, dir xyz) 또는 None"""
    if not os.path.exists(GAZE_TXT):
        return None
    try:
        with open(GAZE_TXT) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return None
        vals = [float(x) for x in lines[-1].replace(",", " ").split()]
        if len(vals) >= 6:
            return np.array(vals[:3]), np.array(vals[3:6])
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="auto",
                    help="auto(기본, by-id 자동탐색) | 인덱스(0,1..) | /dev/videoN")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="YUYV", choices=["YUYV", "MJPG"])
    ap.add_argument("--roi", default=None, help="x,y,w,h 비율 (예: 0.25,0.1,0.5,0.5)")
    ap.add_argument("--flip", action="store_true", help="상하반전 (카메라 거꾸로 장착 시)")
    ap.add_argument("--save-overlay", action="store_true", help="검출 오버레이 png 저장")
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "data"))
    args = ap.parse_args()

    roi = parse_roi(args.roi)
    cap = open_cam(args.dev, args.width, args.height, args.fps, args.fourcc)
    os.makedirs(args.outdir, exist_ok=True)

    for _ in range(10):  # AE 워밍업
        cap.read()

    print(f"[start] live={args.live} roi={roi}")
    print("  눈을 상하좌우로 천천히 굴리세요 → 안구모델 수렴 시 시선벡터가 나옵니다.")

    n, got_vec, misses = 0, 0, 0
    total = 10_000_000 if args.live else args.frames
    try:
        while n < total:
            ok, frame = cap.read()
            if not ok:
                misses += 1
                if misses >= 30:  # 조용히 영원히 도는 대신 실패로 끝낸다
                    raise RuntimeError(
                        "프레임 30회 연속 실패 — USB 재연결로 노드가 바뀌었거나 "
                        "다른 프로세스가 카메라를 점유 중일 수 있음")
                continue
            misses = 0
            if args.flip:
                frame = cv2.flip(frame, -1)
            fed = crop_roi(frame, roi) if roi else frame

            tracker.process_frame(fed)  # 검출·모델·벡터·imshow·gaze_vector.txt 저장
            n += 1

            g = read_last_gaze()
            model_n = len(getattr(tracker, "model_centers", []))
            if g is not None:
                got_vec += 1
                o, d = g
                if args.live or n % 15 == 0:
                    print(f"[{n:4d}] model_centers={model_n:3d}  "
                          f"dir=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})  "
                          f"|d|={np.linalg.norm(d):.3f}")
            elif n % 15 == 0:
                print(f"[{n:4d}] model_centers={model_n:3d}  (아직 벡터 없음)")

            if args.save_overlay and n % 15 == 0:
                cv2.imwrite(os.path.join(args.outdir, f"uvc_{n:04d}.png"), fed)

            if args.live:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[done] {n} frames, gaze vector 나온 프레임 {got_vec}, "
          f"최종 model_centers={len(getattr(tracker,'model_centers',[]))}")
    print(f"       벡터 로그: {GAZE_TXT}")


if __name__ == "__main__":
    main()
