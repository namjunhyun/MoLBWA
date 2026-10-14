#!/usr/bin/env python3
"""손으로 그리는 집기 가능 범위 (2026-10-14, 사용자 제안).

토크를 끈 팔을 사람이 직접 움직여, **그리퍼를 책상과 수평으로 둔 채** 잡는 높이에서
가장 가까운 곳과 가장 먼 곳을 좌우로 쓸어 준다. 수평(±pitch_tol)·잡는 높이(z 범위)일 때만
받침대 중심에서의 거리를 회전각(pan) 칸마다 최소/최대로 모은다 -> 그 팔의 실제 가능 띠.

    python3 tools/so101_reach_trace.py --arm a b --seconds 180

결과: ~/.ros/so101_reach_{arm}.json  {"pan_deg": [...], "r_min": [...], "r_max": [...]}
GUI(control_panel)는 이 파일이 있으면 초록 띠를 이걸로 그린다.

★ 시작하면 토크를 끈다 — 팔이 접힌 휴식자세가 아니면 손으로 받치고 시작할 것.
★ GUI 를 끄고 실행 (같은 포트).
"""
import argparse
import json
import math
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ros2_ws", "src", "gaze_hri"))
from gaze_hri import arms  # noqa: E402
from gaze_hri.kinematics import ArmGeometry, forward_kinematics  # noqa: E402
from gaze_hri.so101_driver import JointMap, So101Bus  # noqa: E402

BIN = 5.0          # pan 칸 (deg)


def reach_file(arm):
    return os.path.expanduser(f"~/.ros/so101_reach_{arm}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", required=True, choices=sorted(arms.ARMS))
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--pitch-tol", type=float, default=7.0, help="수평 허용 (deg)")
    ap.add_argument("--z-min", type=float, default=0.04)
    ap.add_argument("--z-max", type=float, default=0.09)
    a = ap.parse_args()

    geo = ArmGeometry(base_height=0.1166, l3=0.17)
    rigs = {}
    for arm in a.arm:
        c = yaml.safe_load(open(arms.map_file(arm)))
        bus = So101Bus(c["port"], c["servo_ids"], log=lambda s: None)
        bus.relax()
        rigs[arm] = (bus, JointMap(c["zero_ticks"], c["signs"]), {})
    print(f"토크 OFF ({', '.join(a.arm)}). {a.seconds:.0f}초 동안 기록 — 그리퍼를 수평으로, "
          f"잡는 높이({a.z_min * 100:.0f}~{a.z_max * 100:.0f}cm)에서 가장 가까운 곳/먼 곳을 좌우로 쓸어 주세요.")
    t_end = time.monotonic() + a.seconds
    last_print = 0.0
    try:
        while time.monotonic() < t_end:
            line = []
            for arm, (bus, m, bins) in rigs.items():
                q = m.to_rad(bus.ticks())
                p = forward_kinematics(q[:4], geo)
                r = math.hypot(p[0], p[1])
                pan = math.degrees(math.atan2(p[1], p[0]))
                pitch = math.degrees(q[1] + q[2] + q[3])
                ok = abs(pitch) <= a.pitch_tol and a.z_min <= p[2] <= a.z_max
                if ok:
                    k = int(round(pan / BIN))
                    lo, hi = bins.get(k, (r, r))
                    bins[k] = (min(lo, r), max(hi, r))
                line.append(f"{arm}: r {r * 100:5.1f} z {p[2] * 100:5.1f} pitch {pitch:+5.0f}"
                            f" {'기록' if ok else '    '} ({len(bins)}칸)")
            if time.monotonic() - last_print > 0.2:
                print("\r" + "  |  ".join(line), end="", flush=True)
                last_print = time.monotonic()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    print()
    for arm, (bus, m, bins) in rigs.items():
        bus.close()
        if not bins:
            print(f"{arm}: 기록 없음 (수평·잡는 높이 조건을 못 맞춤)")
            continue
        ks = sorted(bins)
        out = {"pan_deg": [k * BIN for k in ks], "r_min": [bins[k][0] for k in ks],
               "r_max": [bins[k][1] for k in ks], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(out, open(reach_file(arm), "w"), indent=1)
        print(f"{arm}: {len(ks)}칸 저장 -> {reach_file(arm)}")
        for k in ks:
            print(f"   pan {k * BIN:+5.0f}°  {bins[k][0] * 100:5.1f} ~ {bins[k][1] * 100:5.1f} cm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
