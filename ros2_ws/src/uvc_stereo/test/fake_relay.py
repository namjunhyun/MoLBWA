"""카메라 없이 전체 경로 시험용: stereo_relay.py 의 Cam 을 가짜(30 fps, 좌우 7 ms 어긋남 + 흔들림)로 바꿔 띄운다.
    python3 fake_relay.py --port 55199
"""
import os
import sys
import threading
import time
import collections

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "edge", "uno_q"))
import stereo_relay as R  # noqa: E402
import cv2  # noqa: E402


class FakeCam(threading.Thread):
    def __init__(self, name, dev, a):
        super().__init__(daemon=True)
        self.name, self.q, self.n = name, collections.deque(maxlen=8), 0
        self.phase = 0.0 if name == "왼쪽" else 0.007
        self.fps = a.fps
        self.rng = np.random.default_rng(1 if name == "왼쪽" else 2)
        print(f"{name}: 가짜 카메라 (위상 {self.phase * 1000:.0f} ms)", flush=True)

    def run(self):
        k, t0 = 0, time.monotonic()
        while True:
            t = t0 + self.phase + k / self.fps + self.rng.normal(0, 0.0015)
            while time.monotonic() < t:
                time.sleep(0.0005)
            img = np.full((480, 640), 40, np.uint8)
            cv2.putText(img, f"{self.name[:1]} {k}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)
            self.q.append((t, cv2.imencode(".jpg", img)[1].tobytes()))
            self.n += 1; k += 1


R.Cam = FakeCam
sys.argv = [sys.argv[0], "--left", "fakeL", "--right", "fakeR"] + sys.argv[1:]
R.main()
