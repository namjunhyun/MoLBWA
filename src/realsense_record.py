#!/usr/bin/env python3
"""
RealSense IR을 우리 알고리즘(Orlosky 3D 트래커)에 물려 '실시간 표시 + 녹화'.
데모(eye_test.mp4)와 동일한 시각화를 우리 카메라 입력으로 만들어 비교용 영상 저장.

- 라이브 창을 보며 눈을 조준/굴려 안구 모델을 세운다.
- q: 종료 후 data/realsense_demo.mp4 저장.
"""
import os, sys, time
import numpy as np, cv2
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKER_DIR = os.path.join(HERE, "..", "external", "EyeTracker", "3DTracker")
sys.path.insert(0, os.path.abspath(TRACKER_DIR))
os.chdir(HERE)
if os.path.exists("gaze_vector.txt"):
    os.remove("gaze_vector.txt")
import Orlosky3DEyeTracker as tracker  # noqa

OUT = os.path.join(HERE, "..", "data", "realsense_demo.mp4")
TARGET_WIN = "Frame with Ellipse and Rays"

# imshow 몽키패치: 타깃 창은 실제 표시 + 녹화용으로 캡처. 나머지 창은 숨김(깔끔).
_real_imshow = cv2.imshow
captured = {"frame": None}
def _imshow(win, mat):
    if win == TARGET_WIN:
        captured["frame"] = mat.copy()
        _real_imshow(win, mat)
cv2.imshow = _imshow


def main():
    pipe = rs.pipeline(); cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    prof = pipe.start(cfg)
    ds = prof.get_device().first_depth_sensor()
    if ds.supports(rs.option.emitter_enabled):
        ds.set_option(rs.option.emitter_enabled, 0.0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))

    print("[REC] 눈을 화면 가득 채우고 상하좌우로 굴려 모델을 세우세요. q=종료·저장")
    n = 0
    try:
        while True:
            frames = pipe.wait_for_frames()
            irf = frames.get_infrared_frame(1)
            if not irf:
                continue
            gray = np.asanyarray(irf.get_data())
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            captured["frame"] = None
            tracker.process_frame(bgr)   # 검출·모델·벡터·(몽키패치된)imshow
            if captured["frame"] is not None:
                f = captured["frame"]
                if f.shape[1] != 640 or f.shape[0] != 480:
                    f = cv2.resize(f, (640, 480))
                writer.write(f); n += 1

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        pipe.stop(); writer.release(); cv2.destroyAllWindows()

    mc = len(getattr(tracker, "model_centers", []))
    sz = os.path.getsize(OUT)/1e6 if os.path.exists(OUT) else 0
    print(f"[done] {n}프레임 녹화, 최종 model_centers={mc} → {OUT} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
