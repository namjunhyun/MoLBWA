#!/usr/bin/env python3
"""캘리브레이션 bag 에서 두 카메라가 **동시에** 타깃을 본 프레임 수 (Kalibr 스테레오가 발산할 때 첫 점검).
    python3 scripts/bag_stereo_overlap.py calibration/bags/<이름> [Hz=4]
태그 4개 이상을 '봤다'로 센다. OpenCV 검출기라 Kalibr 보다 조금 덜 잡지만 비율 비교엔 충분하다.
"""
import sys

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

P = cv2.aruco.DetectorParameters()
P.perspectiveRemovePixelPerCell, P.perspectiveRemoveIgnoredMarginPerCell, P.errorCorrectionRate = 8, 0.25, 1.0
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), P)
bag, hz = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
frames, last = {}, {}
while r.has_next():
    topic, data, t = r.read_next()
    side = topic.rsplit("/", 1)[-1]
    m = deserialize_message(data, Image)
    st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    if st - last.get(side, -1e9) < 1.0 / hz - 1e-3 and st not in frames:
        continue
    last[side] = st
    g = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
    _, ids, _ = DET.detectMarkers(g)
    frames.setdefault(st, {})[side] = 0 if ids is None else len(ids)
both = [v for v in frames.values() if len(v) == 2]
L = sum(v.get("left", 0) >= 4 for v in both); R = sum(v.get("right", 0) >= 4 for v in both)
B = sum(v["left"] >= 4 and v["right"] >= 4 for v in both)
print(f"{len(both)} 쌍 ({hz:g} Hz) · 왼쪽만 봄 {L - B} · 오른쪽만 봄 {R - B} · **둘 다 봄 {B}**"
      f" · 둘 다 15개 이상 {sum(v['left'] >= 15 and v['right'] >= 15 for v in both)}")
