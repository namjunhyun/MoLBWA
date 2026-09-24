#!/usr/bin/env python3
"""
캘리브레이션 결과(yaml)를 읽어 map -> base_link TF를 상시 발행합니다.

이게 떠 있어야 arm_server가 시선 좌표를 로봇 좌표로 변환할 수 있습니다.
파일이 없으면 항등변환(identity)을 쓰되 경고를 계속 띄웁니다.
"""

import math
import os

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


def matrix_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


class CalibTfPublisher(Node):

    def __init__(self):
        super().__init__("calib_tf_publisher")
        self.declare_parameter("calib_file",
                               os.path.expanduser("~/.ros/gaze_hri_calib.yaml"))
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        path = self.get_parameter("calib_file").value
        T = np.eye(4)
        if os.path.exists(path):
            with open(path) as f:
                data = yaml.safe_load(f)
            T = np.array(data["T_base_world"], dtype=float)
            self.get_logger().info(
                f"캘리브레이션 로드: {path} (RMSE {data.get('rmse_m', 0) * 1000:.1f} mm)")
        else:
            self.get_logger().error(
                f"캘리브레이션 파일이 없습니다: {path}\n"
                "  ros2 run gaze_hri calibrate_world_to_base 를 먼저 실행하세요.\n"
                "  지금은 항등변환을 씁니다 — 로봇이 엉뚱한 곳으로 갑니다."
            )

        # T는 base <- world 변환. TF는 부모 map, 자식 base_link로 발행해야 하므로
        # 역변환을 넣는다.
        R = T[:3, :3]
        t = T[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t

        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.get_parameter("world_frame").value
        tf.child_frame_id = self.get_parameter("base_frame").value
        tf.transform.translation.x = float(t_inv[0])
        tf.transform.translation.y = float(t_inv[1])
        tf.transform.translation.z = float(t_inv[2])
        qx, qy, qz, qw = matrix_to_quat(R_inv)
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)
        tf.transform.rotation.w = float(qw)

        self.bc = StaticTransformBroadcaster(self)
        self.bc.sendTransform(tf)
        self.get_logger().info("map -> base_link static TF 발행")


def main():
    rclpy.init()
    node = CalibTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
