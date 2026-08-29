#!/usr/bin/env python3
"""/camera/left, /camera/right(ocams_stereo_imu_node의 rectified 출력)를 직접 구독해서
StereoSGBM + depth_at으로 깊이를 잰다. gaze_on_scene.py의 Python cv2.VideoCapture 캡처
경로(YUYV, 디모자이킹 없음)를 완전히 우회하고 캘리브레이션에 실제로 쓰인 것과 동일한
C++ 노드의 이미지 파이프라인(Bayer 디모자이킹 포함)을 그대로 쓴다 — 2026-08-29 깊이가
실제보다 30% 짧게 나오는 문제가 (a) 캘리브레이션 자체 문제인지 (b) gaze_on_scene.py의
Python 캡처 경로에만 있는 버그인지 가르기 위한 진단 스크립트.

창을 띄우고 화면 중앙 disparity의 중앙값으로 깊이를 계속 찍는다 — 클릭 상호작용 없이
바로 확인 가능.
"""
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

sys.path.insert(0, "/home/seng/MoLBWA/src")
import ocams_calib  # noqa: E402


def make_stereo_matcher():
    block_size = 7
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=block_size,
        P1=8 * block_size ** 2, P2=32 * block_size ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
    )


class DepthProbe(Node):
    def __init__(self):
        super().__init__('ros2_depth_probe')
        self.matcher = make_stereo_matcher()
        left_sub = Subscriber(self, Image, '/camera/left')
        right_sub = Subscriber(self, Image, '/camera/right')
        self.sync = ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=5, slop=0.05)
        self.sync.registerCallback(self.cb)
        self.get_logger().info(
            f"시작. fx={ocams_calib.RECTIFIED_K[0,0]:.2f} baseline={ocams_calib.BASELINE_M*100:.2f}cm "
            "— 화면 중앙 30x30 disparity 중앙값으로 깊이 측정")

    def cb(self, left_msg, right_msg):
        left = np.frombuffer(left_msg.data, dtype=np.uint8).reshape(left_msg.height, left_msg.width)
        right = np.frombuffer(right_msg.data, dtype=np.uint8).reshape(right_msg.height, right_msg.width)
        disp = self.matcher.compute(left, right).astype(np.float32) / 16.0

        h, w = disp.shape
        cx, cy = w // 2, h // 2
        roi = disp[cy - 15:cy + 15, cx - 15:cx + 15]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (cx - 15, cy - 15), (cx + 15, cy + 15), (0, 255, 0), 2)

        if len(valid) >= 5:
            d = float(np.median(valid))
            depth = ocams_calib.RECTIFIED_K[0, 0] * ocams_calib.BASELINE_M / d
            text = f"depth={depth:.3f}m  disp={d:.1f}px  n={len(valid)}"
            self.get_logger().info(text, throttle_duration_sec=0.5)
            cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, f"측정 실패 n={len(valid)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("ros2_depth_probe (left + center ROI)", vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = DepthProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
