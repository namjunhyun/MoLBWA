#!/usr/bin/env python3
"""SO-101 팔로워를 home / 휴식 자세 / 지정 TCP 로 천천히 옮긴다 (ROS 없이, 실측 맵 사용).

    python3 tools/so101_goto.py save-rest            # 지금 자세를 휴식 자세로 저장 (움직이지 않음)
    python3 tools/so101_goto.py home                 # home: TCP (0.22, 0, 0.16) -45°
    python3 tools/so101_goto.py rest                 # 저장한 휴식 자세로
    python3 tools/so101_goto.py xyz 0.25 0.0 0.12 --pitch -0.785

움직이기 전에 관절 공간 보간 경로를 기구학으로 샘플링해서, 손목/TCP 가 책상(z=0)
아래로 내려가거나 팔꿈치가 베이스 뒤로 더 젖혀지는 구간이 있으면 **움직이지 않는다**.
이동 중에는 so101_driver 감시(추종오차/부하/온도/전압)가 켜져 있다. Ctrl+C = 그 자리 정지.
"""
import argparse
import json
import math
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "ros2_ws", "src", "gaze_hri")
sys.path.insert(0, PKG)
from gaze_hri.kinematics import ArmGeometry, IKError, inverse_kinematics  # noqa: E402
from gaze_hri.so101_driver import JointMap, SafetyStop, So101Bus, move_smooth  # noqa: E402

MAP = os.path.join(PKG, "config", "so101_follower.yaml")
REST = os.path.expanduser("~/.ros/so101_rest_ticks.json")
HOME_XYZ, HOME_PITCH, HOME_GRIP = [0.22, 0.0, 0.16], -0.785, 1.2


def heights(q, g):
    a1 = q[1]; a2 = a1 + q[2]; a3 = a2 + q[3]                          # noqa: E702
    e = g.base_height + g.l1 * math.sin(a1)
    w = e + g.l2 * math.sin(a2)
    t = w + g.l3 * math.sin(a3)
    r_e = g.shoulder_offset + g.l1 * math.cos(a1)
    return e, w, t, r_e


def check_path(q0, q1, g, min_z=0.0):
    """보간 경로 검사. 출발점보다 더 낮아지거나(책상 아래), 팔꿈치가 출발점보다 더 뒤로
    가는 구간이 있으면 이유를 돌려준다. 출발점 자체(휴식 자세는 그리퍼가 책상에 닿아
    있다)는 기준으로만 쓴다."""
    e0, w0, t0, re0 = heights(q0, g)
    floor = min(min_z, w0, t0) - 0.005
    back = min(re0, 0.0) - 0.01
    for i in range(1, 51):
        s = 0.5 - 0.5 * math.cos(math.pi * i / 50)
        q = [a + (b - a) * s for a, b in zip(q0, q1)]
        e, w, t, re = heights(q, g)
        if min(w, t) < floor:
            return f"보간 {i}/50 에서 손목/TCP 높이 {min(w, t) * 100:.1f}cm (책상 아래)"
        if re < back:
            return f"보간 {i}/50 에서 팔꿈치가 베이스 뒤 {-re * 100:.1f}cm (태그판 쪽)"
    return None


def solve(xyz, pitch, g):
    for up in (False, True):                       # 팔꿈치를 높이 드는 해 먼저
        try:
            return list(inverse_kinematics(xyz, g, approach_pitch=pitch, elbow_up=up))
        except IKError:
            continue
    raise SystemExit(f"IK 실패: {xyz} pitch {pitch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["save-rest", "home", "rest", "xyz"])
    ap.add_argument("xyz", nargs="*", type=float)
    ap.add_argument("--pitch", type=float, default=HOME_PITCH)
    ap.add_argument("--speed", type=float, default=15.0, help="deg/s (가장 많이 도는 관절 기준)")
    ap.add_argument("--check-only", action="store_true", help="경로 검사만, 안 움직임")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(MAP))
    m = JointMap(cfg["zero_ticks"], cfg["signs"])
    g = ArmGeometry()
    bus = So101Bus(cfg["port"], cfg["servo_ids"])
    try:
        now = bus.ticks()
        if a.cmd == "save-rest":
            os.makedirs(os.path.dirname(REST), exist_ok=True)
            json.dump(now, open(REST, "w"))
            print(f"휴식 자세 저장: {now}")
            return 0
        q0 = m.to_rad(now)
        if a.cmd == "rest":
            target = json.load(open(REST))
            q1 = m.to_rad(target)
        else:
            xyz = HOME_XYZ if a.cmd == "home" else a.xyz
            if len(xyz) != 3:
                sys.exit("xyz 는 x y z 세 값")
            q1 = solve(xyz, a.pitch, g) + [q0[4], HOME_GRIP]
            target = m.to_ticks(q1)
        why = check_path(q0, q1, g) if a.cmd != "rest" else None
        print(f"현재 q(deg) {[round(math.degrees(v)) for v in q0[:4]]} -> "
              f"목표 {[round(math.degrees(v)) for v in q1[:4]]}")
        if why:
            sys.exit(f"움직이지 않음 — {why}")
        if a.check_only:
            print("경로 검사 통과 (--check-only, 안 움직임)")
            return 0
        bus.enable()
        move_smooth(bus, target, max_deg_s=a.speed)
        print("도착. 토크는 켜 둔 채 자세 유지.")
    except SafetyStop as e:
        print(f"안전 정지: {e}")
        return 2
    except KeyboardInterrupt:
        bus.hold()
        print("중단 — 그 자리에 멈춤")
        return 1
    finally:
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
