#!/usr/bin/env python3
"""SO-101 팔로워 첫 구동 — 관절 하나를 기구학 기준 +N° 천천히 움직였다 제자리로.

맵(zero/sign)이 맞는지 **눈으로** 확인하는 도구다. + 방향 정의(so101_map_calib.py 와 같다):
  1 pan      : 팔의 왼쪽으로 회전 (팔을 마주 본 사람에겐 오른쪽)
  2 shoulder : 위팔이 위로 든다
  3 elbow    : 아래팔이 위로 든다
  4 wrist    : 손목(그리퍼 끝)이 위로 든다
  5 roll     : (방향 확인만)
  6 gripper  : 열린다
반대로 움직이면 즉시 Ctrl+C -> 그 자리에 선다. 맵의 sign 이 틀린 것이다.

    python3 tools/so101_jog.py 4 --deg 5          # 손목 +5° 갔다가 복귀
    python3 tools/so101_jog.py relax --yes         # 토크 OFF (팔을 받치고)
"""
import argparse
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ros2_ws", "src", "gaze_hri"))
from gaze_hri.so101_driver import (JOINT_NAMES, JointMap, SafetyStop,  # noqa: E402
                                   So101Bus, move_smooth)

MAP = os.path.join(HERE, "..", "ros2_ws", "src", "gaze_hri", "config", "so101_follower.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("joint", help="1~6 또는 relax")
    ap.add_argument("--deg", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=10.0, help="deg/s")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(MAP))
    bus = So101Bus(cfg["port"], cfg["servo_ids"])
    try:
        if a.joint == "relax":
            if not a.yes:
                sys.exit("토크를 끄면 팔이 떨어진다. 팔을 받치고 --yes")
            bus.relax()
            print("토크 OFF")
            return 0
        j = int(a.joint) - 1
        if not 0 <= j <= 5 or abs(a.deg) > 20:
            sys.exit("관절 1~6, 각도 ±20° 이내")
        m = JointMap(cfg["zero_ticks"], cfg["signs"])
        bus.enable()
        start = bus.ticks()
        q = m.to_rad(start)
        q[j] += a.deg * 3.14159265 / 180
        target = m.to_ticks(q)
        print(f"{JOINT_NAMES[j]}: {start[j]} -> {target[j]}틱 (+{a.deg}°), {a.speed}°/s")
        move_smooth(bus, target, max_deg_s=a.speed)
        time.sleep(0.8)
        move_smooth(bus, start, max_deg_s=a.speed)
        print("복귀 완료. 토크는 켜 둔 채(현재 자세 유지)로 끝낸다.")
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
