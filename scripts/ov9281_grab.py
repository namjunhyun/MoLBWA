#!/usr/bin/env python3
"""/camera/left·right 한 장씩 → /tmp/grab_lr.jpg (좌우 나란히, 검출된 태그 표시)."""
import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image

_P = cv2.aruco.DetectorParameters()
# 실측(2026-10-26, 판 ~1 m·태그 ~18 px): 기본값은 후보를 대부분 탈락시켜 6/36 → 아래로 19/36. 이 판독은 미리보기용이고
# 실제 계산은 Kalibr 자체 검출기가 한다.
_P.perspectiveRemovePixelPerCell, _P.perspectiveRemoveIgnoredMarginPerCell, _P.errorCorrectionRate = 8, 0.25, 1.0
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), _P)
img = {}
rclpy.init()
n = rclpy.create_node("ov9281_grab")
for s in ("left", "right"):
    n.create_subscription(Image, f"/camera/{s}",
                          lambda m, s=s: img.__setitem__(s, np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)), 5)
while rclpy.ok() and len(img) < 2:
    rclpy.spin_once(n, timeout_sec=0.1)
out = []
for s in ("left", "right"):
    cv2.imwrite(f"/tmp/grab_{s}.png", img[s])          # 원본(오프라인 검출 시험용)
    c, ids, rej = DET.detectMarkers(img[s])
    o = cv2.cvtColor(img[s], cv2.COLOR_GRAY2BGR)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(o, c, ids)
    cv2.putText(o, f"{s} tags {0 if ids is None else len(ids)} rejected {len(rej)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    out.append(o)
cv2.imwrite("/tmp/grab_lr.jpg", np.hstack(out))
print("저장 /tmp/grab_lr.jpg")
