#!/usr/bin/env python3
"""
eye_test.mp4에 Orlosky 3D 트래커 시각화(동공 타원+안구 구+시선선+3D 벡터)를 입혀
결과 영상 data/eye_tracking_demo.mp4 로 저장.

방법: 트래커가 "Frame with Ellipse and Rays" 창으로 imshow 하는 완성 프레임을
      cv2.imshow 몽키패치로 가로채 VideoWriter에 기록(헤드리스, GUI 창 안 뜸).
1패스: 안구 구 모델 예열(캡처 X) → 2패스: 수렴된 모델로 렌더링(캡처 O).
"""
import os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKER_DIR = os.path.join(HERE, "..", "external", "EyeTracker", "3DTracker")
sys.path.insert(0, os.path.abspath(TRACKER_DIR))
os.chdir(HERE)
if os.path.exists("gaze_vector.txt"):
    os.remove("gaze_vector.txt")
import Orlosky3DEyeTracker as tracker  # noqa

VIDEO = os.path.join(HERE, "..", "external", "EyeTracker", "eye_test.mp4")
OUT   = os.path.join(HERE, "..", "data", "eye_tracking_demo.mp4")
TARGET_WIN = "Frame with Ellipse and Rays"

# --- cv2.imshow 몽키패치: 타깃 창 프레임만 캡처, 나머지/GUI는 무시 ---
_real_imshow = cv2.imshow
captured = {"frame": None}
def _capture_imshow(winname, mat):
    if winname == TARGET_WIN:
        captured["frame"] = mat.copy()
    # 실제 창은 띄우지 않음(헤드리스)
cv2.imshow = _capture_imshow
cv2.waitKey = lambda *a, **k: -1
cv2.destroyAllWindows = lambda *a, **k: None


def run_pass(capture, writer=None):
    cap = cv2.VideoCapture(VIDEO)
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        captured["frame"] = None
        tracker.process_frame(frame)
        n += 1
        if capture and writer is not None and captured["frame"] is not None:
            f = captured["frame"]
            if f.shape[1] != 640 or f.shape[0] != 480:
                f = cv2.resize(f, (640, 480))
            writer.write(f)
    cap.release()
    return n


def main():
    print("1패스: 안구 모델 예열 중...")
    n1 = run_pass(capture=False)
    print(f"  {n1}프레임 처리, model_centers={len(getattr(tracker,'model_centers',[]))}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT, fourcc, 30.0, (640, 480))
    print("2패스: 시각화 렌더링 + 저장...")
    n2 = run_pass(capture=True, writer=writer)
    writer.release()

    sz = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    print(f"  {n2}프레임 렌더링 → {OUT} ({sz/1e6:.1f} MB)")
    print("done")


if __name__ == "__main__":
    main()
