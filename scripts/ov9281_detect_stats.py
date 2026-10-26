#!/usr/bin/env python3
"""지금 들어오는 /camera/left·right 를 N 초 받아: 밝기, 선명도, AprilTag 검출률(프레임 중 태그≥4 비율)·평균 태그 수.
   태그가 자꾸 0 으로 떨어질 때 원인(어두움/흐림/반사) 가리기용.   python3 scripts/ov9281_detect_stats.py [초=3]
"""
import sys
import time

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image

DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                              cv2.aruco.DetectorParameters())
sec = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
st = {s: [] for s in ("left", "right")}


def cb(side):
    def f(m):
        g = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
        _, ids, _ = DET.detectMarkers(g)
        st[side].append((float(g.mean()), float(cv2.Laplacian(g, cv2.CV_64F).var()), 0 if ids is None else len(ids)))
    return f


rclpy.init()
n = rclpy.create_node("ov9281_detect_stats")
for s in st:
    n.create_subscription(Image, f"/camera/{s}", cb(s), 5)
t_end = time.time() + sec
while time.time() < t_end:
    rclpy.spin_once(n, timeout_sec=0.05)
for s, v in st.items():
    if not v:
        print(s, "영상 없음"); continue
    a = np.array(v)
    print(f"{s}: {len(a)}장 · 밝기 {a[:, 0].mean():.0f}/255 · 선명도 {a[:, 1].mean():.0f} · "
          f"태그≥4 인 프레임 {100 * (a[:, 2] >= 4).mean():.0f}% · 평균 태그 {a[:, 2].mean():.1f}/36 · 0개 프레임 {100 * (a[:, 2] == 0).mean():.0f}%")
n.destroy_node(); rclpy.try_shutdown()
