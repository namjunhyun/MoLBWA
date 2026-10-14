#!/usr/bin/env python3
"""리더 팔로 팔로워를 조종한다 (2026-10-14). 한 팔 또는 두 팔 동시.

    python3 tools/so101_teleop.py --arm b                 # 팔 b 하나
    python3 tools/so101_teleop.py --arm a b               # 두 팔 (리더 2대)
    python3 tools/so101_teleop.py --arm b --record        # 궤적 기록 notes/teleop/*.csv
    python3 tools/so101_teleop.py --arm b --rest          # 끝날 때 휴식자세로 접고 토크 OFF

★ GUI(arm_server)가 그 팔 포트를 쓰고 있으면 못 연다 — GUI 를 먼저 끌 것.
★ 시작할 때 팔로워가 리더 자세로 **천천히(start_deg_s)** 붙는다. 리더를 팔로워와 비슷한
  자세에 두고 시작하면 거의 안 움직인다.

안전: 팔로워 쪽은 So101Bus 감시(부하/추종 오차/점프)를 그대로 받는다. 한 스텝 이동은
max_deg_s 로 제한 (리더를 확 꺾어도 팔로워는 그 속도로만 따라간다). Ctrl-C = 그 자리 정지.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ros2_ws", "src", "gaze_hri"))
from gaze_hri import arms  # noqa: E402
from gaze_hri.so101_driver import JointMap, SafetyStop, So101Bus, move_smooth  # noqa: E402
from gaze_hri.teleop import Leader, grip_from_ratio, max_gap_deg, step_toward  # noqa: E402

GRIP_FULL_CLOSE = 0.10      # arm_server gripper_full_close 와 같게
GRIP_OPEN = 2.24            # arm_server gripper_open


class Pair:
    def __init__(self, arm, log_dir):
        cfg = yaml.safe_load(open(arms.map_file(arm)))
        self.arm = arm
        self.fmap = JointMap(cfg["zero_ticks"], cfg["signs"])
        self.leader = Leader(arms.leader_port(arm), arms.leader_map_file(arm))
        self.bus = So101Bus(cfg["port"], cfg["servo_ids"])
        self.bus.enable()
        self.q = self.fmap.to_rad(self.bus.ticks())
        self.synced = False
        self.csv = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, time.strftime(f"teleop_{arm}_%m%d_%H%M%S.csv"))
            self.csv = csv.writer(open(path, "w"))
            self.csv.writerow(["t"] + [f"leader{i}" for i in range(1, 7)]
                              + [f"goal{i}" for i in range(1, 7)]
                              + [f"pos{i}" for i in range(1, 7)]
                              + [f"load{i}" for i in range(1, 7)])
            print(f"[{arm}] 기록: {path}")

    def step(self, t, start_step, run_step):
        lq = self.leader.read()
        target = lq[:5] + [grip_from_ratio(lq[5], GRIP_FULL_CLOSE, GRIP_OPEN)]
        if not self.synced and max_gap_deg(target, self.q) < 2.0:
            self.synced = True
            print(f"[{self.arm}] 리더와 맞춰짐 — 이제 바로 따라갑니다")
        self.q = step_toward(self.q, target, run_step if self.synced else start_step)
        goal = self.bus.write_ticks(self.fmap.to_ticks(self.q))
        if self.csv:
            self.csv.writerow([f"{t:.3f}"] + list(self.leader.last_ticks) + list(goal)
                              + list(self.bus.last_pos or []) + list(self.bus.last_load or []))

    def finish(self, rest):
        try:
            if rest and os.path.exists(arms.rest_file(self.arm)):
                print(f"[{self.arm}] 휴식자세로")
                move_smooth(self.bus, json.load(open(arms.rest_file(self.arm))), max_deg_s=30)
                time.sleep(0.5)
                self.bus.relax()
            else:
                self.bus.hold()
        finally:
            self.bus.close()
            self.leader.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", required=True, choices=sorted(arms.ARMS))
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--max-deg-s", type=float, default=180.0)
    ap.add_argument("--start-deg-s", type=float, default=30.0)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--rest", action="store_true", help="끝나면 휴식자세로 접고 토크 OFF")
    a = ap.parse_args()

    log_dir = os.path.join(HERE, "..", "notes", "teleop") if a.record else None
    pairs = []
    try:
        for arm in a.arm:
            pairs.append(Pair(arm, log_dir))
    except Exception as exc:
        for p in pairs:
            p.finish(False)
        sys.exit(f"시작 실패: {exc}")

    dt = 1.0 / a.hz
    run_step = math.radians(a.max_deg_s) * dt
    start_step = math.radians(a.start_deg_s) * dt
    print(f"텔레옵 시작 ({', '.join(a.arm)}) — Ctrl-C 로 끝")
    t0 = time.monotonic()
    i = 0
    try:
        while True:
            i += 1
            t = time.monotonic() - t0
            for p in pairs:
                p.step(t, start_step, run_step)
            time.sleep(max(0.0, t0 + i * dt - time.monotonic()))
    except SafetyStop as exc:
        print(f"\n안전 정지: {exc}")
    except KeyboardInterrupt:
        print("\n끝")
    finally:
        for p in pairs:
            p.finish(a.rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
