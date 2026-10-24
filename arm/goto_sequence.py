#!/usr/bin/env python3
"""docs/12 R,p_eye 캘리브용 — 그리퍼를 여러(근거리~원거리) 자세로 순서대로 보낸다.

arm_server(/arm/goto_joints 구독)가 떠 있어야 한다(예: calibrate_world_to_base.launch.py).
2026-10-22 양팔: --arm a 면 /arm_a/goto_joints (양팔 GUI 의 arm_server_a). 자세는 목표 xyz 를
지금 기구학(그리퍼 l3=0.17)으로 그 자리에서 IK 로 푼다 — 옛 관절각 목록은 옛 그리퍼(0.135) 기준이었다.
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
from gaze_hri.kinematics import ArmGeometry, forward_kinematics, solve_with_fallback  # noqa: E402

# 2026-10-22 양팔 (팔 A 기준 실제 위치, m). 손끝 최저 5cm, 착용자 시야 전체에 퍼지게 —
# 팔 B 쪽(가운데, y<0)과 높이 7~25cm 를 섞었다. 팔 높이 보정(dual.launch Z_OFFSET)은 명령에만 더한다.
TARGETS_DUAL = [
    (0.25, 0.10, 0.15), (0.25, -0.10, 0.15), (0.35, 0.12, 0.08), (0.35, -0.12, 0.08),
    (0.28, 0.00, 0.25), (0.38, 0.00, 0.15), (0.30, -0.25, 0.10), (0.22, 0.20, 0.10),
    (0.40, -0.10, 0.20), (0.20, -0.15, 0.07),
]
Z_OFFSET = {"a": 0.02, "b": 0.0}       # launch/dual.launch.py 와 같게
PITCHES = [-math.pi / 6, -math.pi / 4, 0.0, -math.pi / 3, -1.2, -math.pi / 2]

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
    def __init__(self, topic="/arm/goto_joints"):
        super().__init__("goto_sequence")
        self.pub_joints = self.create_publisher(Float64MultiArray, topic, 10)
        self.pub_role = self.create_publisher(String, "/task/expected_role", 10)
        self.geo = ArmGeometry()

    def goto(self, q):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q] + [0.0, 0.8]  # wrist_roll=0, gripper 살짝 벌림
        self.pub_joints.publish(msg)


# 양팔 GUI 의 arm_server 가 쓰는 홈/휴식자세 (2026-10-22 로그: 홈 TCP (0.30, 0, 0.19) 수평)
HOME_Q = {"a": [0.0, 1.83, -2.121, 0.291, 0.0, 1.2]}
REST_Q = {"a": [-0.061, 3.056, -2.812, -1.223, -0.08, -0.002]}


def run_drift(node, arm):
    """1점 보정: 홈 자세에서 그리퍼를 보고 SPACE -> gaze_on_scene 'o' 와 같은 보정 (2026-10-22)."""
    import socket

    import cv2
    import numpy as np

    rec = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def show(lines):
        img = np.full((150, 640, 3), 30, np.uint8)
        for k, t in enumerate(lines):
            cv2.putText(img, t, (10, 30 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("drift correction", img)
        cv2.waitKey(1)

    def send(q, wait):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q]
        node.pub_joints.publish(msg)
        time.sleep(wait)

    t0 = time.time()
    while node.pub_joints.get_subscription_count() == 0 and time.time() - t0 < 5.0:
        time.sleep(0.1)
    time.sleep(0.3)
    show(["moving: rest -> home ..."])
    send(HOME_Q[arm], 6.0)
    while True:
        show(["look between the fingers and press SPACE",
              "(keep looking 3 s)   q = finish (go rest)"])
        key = 0
        while key not in (ord(" "), 13, ord("q")):
            key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        rec.sendto(b"drift", ("127.0.0.1", 55060))
        print("1점 보정 요청 보냄", flush=True)
        show(["MEASURING - keep looking at the gripper (3 s) ..."])
        time.sleep(3.5)
    show(["moving: home -> rest ..."])
    send(REST_Q[arm], 6.0)
    cv2.destroyAllWindows()
    return 0


def run_gui(node, arm, keep=False):
    """작은 창으로 한 자세씩. 휴식자세 -> 홈 -> 자세들 -> 홈 -> 휴식자세 (GUI 작업과 같은 경로)."""
    import socket

    import cv2
    import numpy as np

    rec = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(q, wait):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q]
        node.pub_joints.publish(msg)
        time.sleep(wait)

    def show(lines):
        img = np.full((150, 640, 3), 30, np.uint8)
        for k, t in enumerate(lines):
            cv2.putText(img, t, (10, 30 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("goto sequence", img)

    node.pub_role.publish(String(data="idle"))
    # 2026-10-22: 연결되기 전에 보낸 첫 명령(홈)이 사라져 휴식자세에서 첫 자세로 바로 갔다 -> 구독자 확인
    t0 = time.time()
    while node.pub_joints.get_subscription_count() == 0 and time.time() - t0 < 5.0:
        time.sleep(0.1)
    time.sleep(0.3)
    if not keep:
        rec.sendto(b"reset", ("127.0.0.1", 55060))      # gaze_on_scene: 이번 세션 점 비우기
        print("시선 뷰어에 새 캘리브 시작(점 비움) 보냄", flush=True)
    show(["moving: rest -> home ..."])
    cv2.waitKey(1)
    send(HOME_Q[arm], 5.0)
    rnd = 1
    i = 0
    again = False
    while True:
        xyz, q = POSES[i]
        if not again:
            show([f"round {rnd}  pose {i + 1}/{len(POSES)}  moving to {xyz} ..."])
            cv2.waitKey(1)
            node.goto(q)
            print(f"[{rnd}회 {i + 1}/{len(POSES)}] 이동 -> {xyz}", flush=True)
            time.sleep(4.0)
        again = False
        show([f"round {rnd}  pose {i + 1}/{len(POSES)}  at {xyz}",
              "look between the fingers and press SPACE (= record + next)",
              "n = next without recording,  q = finish (go rest)"])
        key = 0
        while key not in (ord(" "), 13, ord("n"), ord("q")):
            key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        if key in (ord(" "), 13):
            # 2026-10-22: 씬 창 클릭 대신 — gaze_on_scene.py 가 UDP 55060 "record" 를 받아 그 순간 시선을 기록
            rec.sendto(b"record", ("127.0.0.1", 55060))
            print(f"    기록 요청 보냄", flush=True)
            # 시선 뷰어가 0.8~1.8초 구간 시선을 평균한다 — 그동안 팔을 움직이면 안 된다
            show([f"round {rnd}  pose {i + 1}/{len(POSES)}  RECORDING - keep looking at the gripper ..."])
            cv2.waitKey(1)
            time.sleep(2.0)
            show([f"round {rnd}  pose {i + 1}/{len(POSES)}  done (failed? see gaze log)",
                  "SPACE = next pose,  r = record again here,  q = finish"])
            key = 0
            while key not in (ord(" "), 13, ord("n"), ord("r"), ord("q")):
                key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                again = True      # 같은 자세에서 다시 기록 (팔은 그대로)
                continue
        i += 1
        if i == len(POSES):
            show([f"round {rnd} done.  r = one more round,  q = finish (go rest)"])
            key = 0
            while key not in (ord("r"), ord("q")):
                key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                break
            rnd += 1
            i = 0
    show(["moving: home -> rest ..."])
    cv2.waitKey(1)
    print("홈 -> 휴식자세", flush=True)
    send(HOME_Q[arm], 5.0)
    send(REST_Q[arm], 6.0)
    cv2.destroyAllWindows()
    print("끝 (토크는 켜진 채 휴식자세 — GUI 재시작 시 꺼짐)", flush=True)
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b"], default=None, help="양팔: 이 팔로 (없으면 옛 단일 팔)")
    ap.add_argument("--drift", action="store_true", help="홈 자세 1점 보정만 (gaze_on_scene 'o' 대신 SPACE)")
    ap.add_argument("--keep", action="store_true", help="--gui: 시선 뷰어의 기존 점을 비우지 않고 이어서")
    ap.add_argument("--gui", action="store_true",
                    help="터미널 Enter 대신 작은 창에서 space=다음, r=한 바퀴 더, q=끝(홈->휴식자세)")
    args = ap.parse_args()
    global POSES
    if args.arm:
        geo, dz = ArmGeometry(), Z_OFFSET[args.arm]
        POSES = []
        for t in TARGETS_DUAL:
            q, _ = solve_with_fallback((t[0], t[1], t[2] + dz), geo, PITCHES)
            POSES.append((t, tuple(q)))
    rclpy.init()
    node = Sequencer(f"/arm_{args.arm}/goto_joints" if args.arm else "/arm/goto_joints")
    if args.drift:
        return run_drift(node, args.arm)
    if args.gui:
        return run_gui(node, args.arm, args.keep)
    node.pub_role.publish(String(data="idle"))
    time.sleep(0.3)
    print(f"\n총 {len(POSES)}개 자세. 각 자세에서 그리퍼를 응시하며 gaze_on_scene.py 창에서")
    print("'e'(R,p_eye 모드) 켜고 좌클릭하세요. 이 터미널에서 Enter 치면 다음 자세로 이동.\n")
    try:
        for i, (xyz, q) in enumerate(POSES, 1):
            node.goto(q)
            x, y, z = forward_kinematics(list(q), node.geo)
            print(f"[{i}/{len(POSES)}] 이동 중... 목표 xyz=({x:.3f},{y:.3f},{z:.3f})")
            time.sleep(4.0)  # 이동 완료 대기 (joint_speed_deg_s=30, 휴식자세에서 첫 이동은 김)
            input(f"    도착. 그리퍼 끝을 응시하고 클릭한 뒤 Enter >> ")
    except KeyboardInterrupt:
        print("\n중단")
    return 0


if __name__ == "__main__":
    sys.exit(main())
