#!/usr/bin/env python3
"""캘리브레이션 녹화 중 미리보기: /camera/left·right 에서 AprilGrid 태그를 찾아 표시 + 덮은 영역 누적.
   (Kalibr 는 실시간 화면이 없어서 이걸 보며 판을 움직인다. q 로 닫기 — 녹화는 따로 계속된다)
초록 = 지금 잡힌 태그, 파란 칠 = 지금까지 덮은 영역(귀퉁이까지 다 칠해지면 충분), 위 글자 = 잡힌 태그 수.
"""
import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image

_P = cv2.aruco.DetectorParameters()
# 실측(2026-10-26, 판 ~1 m·태그 ~18 px): 기본값은 후보를 대부분 탈락시켜 6/36 → 아래로 19/36. 이 판독은 미리보기용이고
# 실제 계산은 Kalibr 자체 검출기가 한다.
_P.perspectiveRemovePixelPerCell, _P.perspectiveRemoveIgnoredMarginPerCell, _P.errorCorrectionRate = 8, 0.25, 1.0
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), _P)
img = {"left": None, "right": None}
cov = {}


def on(side):
    def cb(m):
        img[side] = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
    return cb


def draw(side):
    g = img[side]
    if g is None:
        return np.zeros((480, 640, 3), np.uint8)
    c = cov.setdefault(side, np.zeros(g.shape, np.uint8))
    corners, ids, _ = DET.detectMarkers(g)
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    n = 0 if ids is None else len(ids)
    if n:
        for q in corners:
            cv2.fillConvexPoly(c, q.reshape(-1, 2).astype(np.int32), 255)
        cv2.aruco.drawDetectedMarkers(out, corners)
    out[c > 0] = (0.6 * out[c > 0] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
    cv2.putText(out, f"{side} tags {n}/36  cover {100 * (c > 0).mean():.0f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if n >= 8 else (0, 0, 255), 2)
    return out


rclpy.init()
node = rclpy.create_node("ov9281_calib_preview")
for s in img:
    node.create_subscription(Image, f"/camera/{s}", on(s), 2)
while rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.03)
    cv2.imshow("OV9281 calib preview (q: close)", np.hstack([draw("left"), draw("right")]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
node.destroy_node()
rclpy.try_shutdown()
