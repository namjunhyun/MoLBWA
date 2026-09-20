#!/usr/bin/env python3
"""
[캘리브레이션] SLAM world(map) <-> 로봇 베이스(base_link) 정합

이 프로젝트에서 가장 중요한 단계입니다. 이거 없으면 로봇은 절대 맞는 곳으로 안 갑니다.

원리:
  1. 로봇팔을 미리 정해둔 N개 자세로 보낸다.
     -> 그리퍼 끝(TCP)이 base_link 기준 어디 있는지는 FK로 정확히 안다. (p_B)
  2. 사용자는 매번 그리퍼 끝을 응시한다.
     -> dwell_detector가 그 지점의 SLAM world 좌표를 준다. (p_W)
  3. (p_W, p_B) 쌍 N개로 Umeyama 정합 -> T_BW (강체 변환)
  4. 결과를 yaml로 저장. 이후 static_transform_publisher가 TF로 뿌린다.

이 방식의 좋은 점은 SLAM 좌표계 오차뿐 아니라 아이트래킹의 체계적 편향
(systematic bias)까지 같이 흡수된다는 겁니다. 대회 심사 때 설명 포인트가 됩니다.

사용법:
    ros2 run gaze_hri calibrate_world_to_base
    (터미널 안내를 따라 그리퍼 끝을 응시. 자세마다 약 2초)

출력:
    ~/.ros/gaze_hri_calib.yaml
    RMSE가 2cm 이하로 나와야 정상입니다. 5cm를 넘으면
    - 그리퍼 끝이 아니라 다른 데를 봤거나
    - 링크 길이(URDF)가 틀렸거나
    - SLAM 스케일이 틀린 것(모노큘러로 돌린 경우)입니다.
"""

import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
import yaml
from gaze_hri_msgs.msg import Fixation
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from gaze_hri.kinematics import (ArmGeometry, alignment_rmse,
                                 forward_kinematics, matrix_to_xyz_rpy,
                                 umeyama_rigid)

# 캘리브레이션 자세 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex)
# 작업 공간에 골고루 퍼지고, 한 평면에 몰리지 않게(중요!) 배치했습니다.
# 세 점이 일직선이거나 모든 점이 한 평면에 있으면 회전이 제대로 안 풀립니다.
CALIB_POSES = [
    (0.0, -0.35, 1.10, -0.60),
    (0.55, -0.20, 0.95, -0.50),
    (-0.55, -0.20, 0.95, -0.50),
    (0.30, 0.15, 0.70, -0.30),
    (-0.30, 0.15, 0.70, -0.30),
    (0.0, -0.70, 1.35, -0.35),
    (0.40, -0.55, 1.25, -0.80),
]


class Calibrator(Node):

    def __init__(self):
        super().__init__("calibrate_world_to_base")

        self.declare_parameter("base_height", 0.0563)
        self.declare_parameter("shoulder_offset", 0.0304)
        self.declare_parameter("l1", 0.1160)
        self.declare_parameter("l2", 0.1350)
        self.declare_parameter("l3", 0.1100)
        self.declare_parameter("output",
                               os.path.expanduser("~/.ros/gaze_hri_calib.yaml"))

        self.geo = ArmGeometry(
            base_height=float(self.get_parameter("base_height").value),
            shoulder_offset=float(self.get_parameter("shoulder_offset").value),
            l1=float(self.get_parameter("l1").value),
            l2=float(self.get_parameter("l2").value),
            l3=float(self.get_parameter("l3").value),
        )

        self.pub_joints = self.create_publisher(Float64MultiArray, "/arm/goto_joints", 10)
        self.pub_role = self.create_publisher(String, "/task/expected_role", 10)
        self.create_subscription(Fixation, "/gaze/fixation", self.on_fixation, 10)

        self._fix_event = threading.Event()
        self._last_fix = None

    def on_fixation(self, msg: Fixation):
        self._last_fix = np.array([msg.point.x, msg.point.y, msg.point.z])
        self._fix_event.set()

    def goto(self, q):
        msg = Float64MultiArray()
        # 그리퍼를 살짝 벌려두면 끝점이 눈에 잘 띕니다
        msg.data = [float(v) for v in q] + [0.0, 0.8]
        self.pub_joints.publish(msg)

    def wait_for_fixation(self, timeout=30.0):
        self._fix_event.clear()
        if not self._fix_event.wait(timeout):
            return None
        return self._last_fix

    def _warn_if_task_pipeline_running(self):
        """캘리브레이션 중에 팔이 제멋대로 움직이는 상황을 미리 막는다.

        target_resolver/task_manager 가 떠 있으면 '그리퍼 끝 응시'가
        '집을 물체 선택'으로 해석되어 캘리브 도중 파지 동작이 시작될 수 있다.
        """
        try:
            names = self.get_node_names()
        except Exception:  # noqa: BLE001
            return
        risky = [n for n in ("target_resolver", "task_manager") if n in names]
        if risky:
            print("\n" + "!" * 62)
            print(f" 경고: {', '.join(risky)} 가 실행 중입니다.")
            print(" 캘리브레이션 중 응시가 '물체 선택'으로 해석되어 팔이")
            print(" 갑자기 집기 동작을 시작할 수 있습니다.")
            print(" 그 노드들을 끄고, 캘리브 전용 런치를 쓰세요:")
            print("   ros2 launch gaze_hri calibrate_world_to_base.launch.py")
            print("!" * 62 + "\n")

    def run(self):
        self._warn_if_task_pipeline_running()
        # target_resolver가 응시점을 가공하지 않도록 idle로 돌려둡니다
        self.pub_role.publish(String(data="idle"))

        print("\n" + "=" * 62)
        print(" SLAM world <-> 로봇 베이스 좌표계 캘리브레이션")
        print("=" * 62)
        print(" 로봇팔이 자세를 잡을 때마다, 그리퍼 '끝'을 가만히 응시하세요.")
        print(" 초록색 커서가 커지면 인식된 겁니다. Ctrl+C로 중단.\n")

        src, dst = [], []
        for i, q in enumerate(CALIB_POSES, 1):
            print(f"[{i}/{len(CALIB_POSES)}] 자세 이동 중...")
            self.goto(q)
            time.sleep(2.5)

            p_B = forward_kinematics(q, self.geo)
            print(f"    로봇 기준 그리퍼 끝: ({p_B[0]:.3f}, {p_B[1]:.3f}, {p_B[2]:.3f})")
            print("    >>> 지금 그리퍼 끝을 응시하세요...")

            p_W = self.wait_for_fixation()
            if p_W is None:
                print("    시간 초과 — 이 자세는 건너뜁니다.")
                continue
            print(f"    시선 기준 좌표:      ({p_W[0]:.3f}, {p_W[1]:.3f}, {p_W[2]:.3f})  OK\n")
            src.append(p_W)
            dst.append(p_B)

        if len(src) < 4:
            print(f"샘플이 {len(src)}개뿐입니다. 최소 4개(권장 6개) 필요합니다.")
            return False

        T_BW = umeyama_rigid(src, dst)
        rmse = alignment_rmse(T_BW, src, dst)

        print("=" * 62)
        print(f" 정합 완료 — 샘플 {len(src)}개, RMSE = {rmse * 1000:.1f} mm")
        if rmse > 0.05:
            print(" !! RMSE가 5cm를 넘습니다. 링크 길이 / SLAM 스케일을 확인하세요.")
        elif rmse > 0.02:
            print(" 주의: 2cm를 넘습니다. 데모는 되지만 샘플을 더 모으면 좋습니다.")
        else:
            print(" 좋습니다.")
        print("=" * 62)

        x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(T_BW)
        data = {
            "T_base_world": T_BW.tolist(),
            "xyz": [float(x), float(y), float(z)],
            "rpy": [float(roll), float(pitch), float(yaw)],
            "rmse_m": float(rmse),
            "num_samples": len(src),
        }
        out = self.get_parameter("output").value
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

        print(f"\n저장 위치: {out}")
        print("\n확인용 TF 수동 실행 명령:")
        print(f"  ros2 run tf2_ros static_transform_publisher \\\n"
              f"    {x:.6f} {y:.6f} {z:.6f} {yaw:.6f} {pitch:.6f} {roll:.6f} \\\n"
              f"    base_link map\n")
        return True


def main():
    rclpy.init()
    node = Calibrator()
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        ok = node.run()
    except KeyboardInterrupt:
        ok = False
        print("\n중단됨")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
