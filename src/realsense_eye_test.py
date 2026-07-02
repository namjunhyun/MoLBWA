#!/usr/bin/env python3
"""
창종설 0단계 — RealSense D455로 눈 카메라 자리 대체 시험.
IR 카메라(OV9281) 도착 전 임시. RealSense IR 스트림을 Orlosky 동공검출기에 물린다.

의존 레포: external/EyeTracker (OrloskyPupilDetectorLite.process_frame)

사용:
  python realsense_eye_test.py --live               # 라이브 창 (사람 눈 클로즈업으로)
  python realsense_eye_test.py --frames 30          # 헤드리스: 30프레임 캡처 후 저장(검증용)
  python realsense_eye_test.py --stream color --live # 컬러로 보기
  python realsense_eye_test.py --emitter on          # IR 프로젝터 켜기(기본 off: 동공검출에 유리)
"""
import os
import sys
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs

# --- Orlosky 검출기 import ---
REPO = os.path.join(os.path.dirname(__file__), "..", "external", "EyeTracker")
sys.path.insert(0, os.path.abspath(REPO))
from OrloskyPupilDetectorLite import process_frame  # noqa: E402


def make_pipeline(stream, emitter, width, height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    if stream == "ir":
        # D455 좌측 IR imager (index 1). 850nm 밴드에 가까운 IR 이미지.
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    else:
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    # IR 프로젝터(구조광 점패턴)는 동공 위에 찍혀 검출을 방해할 수 있음 → 기본 off.
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if emitter else 0.0)
    return pipeline


def get_bgr_frame(frames, stream):
    if stream == "ir":
        f = frames.get_infrared_frame(1)
        if not f:
            return None
        img = np.asanyarray(f.get_data())          # HxW uint8 (grayscale)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 검출기가 BGR을 기대
    else:
        f = frames.get_color_frame()
        if not f:
            return None
        return np.asanyarray(f.get_data())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["ir", "color"], default="ir")
    ap.add_argument("--emitter", choices=["on", "off"], default="off")
    ap.add_argument("--live", action="store_true", help="cv2 라이브 창 표시")
    ap.add_argument("--frames", type=int, default=30, help="헤드리스 모드에서 캡처할 프레임 수")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(outdir, exist_ok=True)

    pipeline = make_pipeline(args.stream, args.emitter == "on",
                             args.width, args.height, args.fps)
    print(f"[start] stream={args.stream} emitter={args.emitter} "
          f"{args.width}x{args.height}@{args.fps}  live={args.live}")

    saved = 0
    detected = 0
    try:
        for i in range(10_000_000 if args.live else args.frames):
            frames = pipeline.wait_for_frames()
            frame = get_bgr_frame(frames, args.stream)
            if frame is None:
                continue

            # process_frame이 내부에서 imshow까지 한다(라이브). 타원 rotated_rect 반환.
            rr = process_frame(frame.copy())
            (cx, cy), (w, h), ang = rr
            ok = (w > 0 and h > 0)
            if ok:
                detected += 1

            if args.live:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)
            else:
                # 헤드리스: 첫/중간/마지막 프레임을 검출 오버레이와 함께 저장
                if i in (0, args.frames // 2, args.frames - 1):
                    vis = frame.copy()
                    if ok:
                        cv2.ellipse(vis, rr, (0, 255, 0), 2)
                        cv2.circle(vis, (int(cx), int(cy)), 3, (255, 255, 0), -1)
                    path = os.path.join(outdir, f"rs_{args.stream}_{i:03d}.png")
                    cv2.imwrite(path, vis)
                    saved += 1
                    print(f"  saved {path}  pupil={'yes' if ok else 'no'} "
                          f"center=({cx:.0f},{cy:.0f}) size=({w:.0f},{h:.0f})")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    n = args.frames if not args.live else max(detected, 1)
    print(f"[done] frames processed, pupil detected in {detected} frames"
          + (f" ({saved} images saved to data/)" if not args.live else ""))


if __name__ == "__main__":
    main()
