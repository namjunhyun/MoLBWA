#!/usr/bin/env python3
"""docs/12 R,p_eye 캘리브용 — 그리퍼를 여러(근거리~원거리) 자세로 순서대로 보낸다.

arm_server(/arm/goto_joints 구독)가 떠 있어야 한다(예: calibrate_world_to_base.launch.py).
각 자세에서 멈추고 Enter를 누를 때까지 기다린다 — 그동안 gaze_on_scene.py 창에서
그리퍼를 응시하며 'e'(또는 'M') 모드로 좌클릭해서 점을 모으면 된다.

    ros2 launch gaze_hri calibrate_world_to_base.launch.py source:=udp backend:=feetech
    python3 arm/goto_sequence.py
"""
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ros2_ws", "src", "gaze_hri"))
from gaze_hri.kinematics import ArmGeometry, forward_kinematics  # noqa: E402

# (xyz 참고용, q) — 근거리 0.18~0.35m 반경, 높이 8~25cm로 다양하게.
# 2026-09-27 시뮬레이션으로 전환 경로 손끝 최저 3.5cm 이상 확인됨.
POSES = [
    ((0.18, 0.10, 0.15), (0.507, 1.483, -1.222, -1.570)),
    ((0.18, -0.10, 0.15), (-0.507, 1.483, -1.222, -1.570)),
    ((0.35, 0.10, 0.12), (0.278, 0.765, -0.936, -0.353)),
    ((0.35, -0.10, 0.12), (-0.278, 0.765, -0.936, -0.353)),
    ((0.25, 0.00, 0.25), (0.000, 1.670, -1.107, -1.087)),
    ((0.30, 0.20, 0.10), (0.588, 0.534, -0.487, -0.833)),
    ((0.30, -0.20, 0.10), (-0.588, 0.534, -0.487, -0.833)),
    ((0.20, 0.00, 0.08), (0.000, 1.247, -1.433, -1.385)),
]


class Sequencer(Node):
    def __init__(self):
        super().__init__("goto_sequence")
        self.pub_joints = self.create_publisher(Float64MultiArray, "/arm/goto_joints", 10)
        self.pub_role = self.create_publisher(String, "/task/expected_role", 10)
        self.geo = ArmGeometry()

    def goto(self, q):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q] + [0.0, 0.8]  # wrist_roll=0, gripper 살짝 벌림
        self.pub_joints.publish(msg)


def main():
    rclpy.init()
    node = Sequencer()
    node.pub_role.publish(String(data="idle"))
    time.sleep(0.3)
    print(f"\n총 {len(POSES)}개 자세. 각 자세에서 그리퍼를 응시하며 gaze_on_scene.py 창에서")
    print("'e'(R,p_eye 모드) 켜고 좌클릭하세요. 이 터미널에서 Enter 치면 다음 자세로 이동.\n")
    try:
        for i, (xyz, q) in enumerate(POSES, 1):
            node.goto(q)
            x, y, z = forward_kinematics(list(q), node.geo)
            print(f"[{i}/{len(POSES)}] 이동 중... 목표 xyz=({x:.3f},{y:.3f},{z:.3f})")
            time.sleep(3.0)  # 이동 완료 대기 (joint_speed_deg_s=12 기준 넉넉히)
            input(f"    도착. 그리퍼 끝을 응시하고 클릭한 뒤 Enter >> ")
    except KeyboardInterrupt:
        print("\n중단")
    return 0


if __name__ == "__main__":
    sys.exit(main())
