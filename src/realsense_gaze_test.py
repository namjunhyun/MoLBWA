#!/usr/bin/env python3
"""
창종설 0단계 — RealSense IR을 Orlosky '3D' 시선 트래커에 물려 '시선 벡터'가 뽑히는지 검증.
(Lite는 동공 타원만 준다. 이 스크립트는 안구 구 모델 → 3D gaze direction까지 간다.)

의존 레포: external/EyeTracker/3DTracker/Orlosky3DEyeTracker.py
  - 눈을 여러 방향으로 굴려 안구 중심(sphere) 모델을 30샘플 이상 쌓아야 벡터가 안정된다.
  - 매 프레임 gaze_vector.txt(= 이 스크립트 옆)에 [origin(3), direction(3)] 저장.

사용:
  python realsense_gaze_test.py --live      # 라이브: 눈 굴려 모델 세우고 시선선/벡터 확인
  python realsense_gaze_test.py --frames 120 # 헤드리스: 벡터 로그만 (사람 눈 촬영 중일 때)
"""
import os
import sys
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKER_DIR = os.path.join(HERE, "..", "external", "EyeTracker", "3DTracker")
sys.path.insert(0, os.path.abspath(TRACKER_DIR))

# gaze_vector.txt가 cwd에 쓰이므로, 결과를 한곳에 모으려고 cwd를 src로 고정
os.chdir(HERE)
GAZE_TXT = os.path.join(HERE, "gaze_vector.txt")
if os.path.exists(GAZE_TXT):
    os.remove(GAZE_TXT)

import Orlosky3DEyeTracker as tracker  # noqa: E402
print("GL_SPHERE_AVAILABLE =", getattr(tracker, "GL_SPHERE_AVAILABLE", "?"))


def make_ir_pipeline(w, h, fps):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)
    prof = pipe.start(cfg)
    ds = prof.get_device().first_depth_sensor()
    if ds.supports(rs.option.emitter_enabled):
        ds.set_option(rs.option.emitter_enabled, 0.0)  # 프로젝터 off (동공검출 방해 방지)
    return pipe


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
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    pipe = make_ir_pipeline(args.width, args.height, args.fps)
    print(f"[start] IR {args.width}x{args.height}@{args.fps} live={args.live}")
    print("  눈을 상하좌우로 천천히 굴리세요 → 안구 모델(파란 원)이 잡히면 시선선/벡터가 나옵니다.")

    n = 0
    got_vec = 0
    try:
        total = 10_000_000 if args.live else args.frames
        while n < total:
            frames = pipe.wait_for_frames()
            irf = frames.get_infrared_frame(1)
            if not irf:
                continue
            gray = np.asanyarray(irf.get_data())
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            tracker.process_frame(bgr)   # 내부에서 검출·모델·벡터·imshow·gaze_vector.txt 저장
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

            if args.live:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    print(f"[done] {n} frames, gaze vector 나온 프레임 {got_vec}, "
          f"최종 model_centers={len(getattr(tracker,'model_centers',[]))}")
    print(f"       벡터 로그: {GAZE_TXT}")


if __name__ == "__main__":
    main()
