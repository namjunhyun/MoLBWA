#!/usr/bin/env python3
"""SO-101 리더 팔 준비 (2026-10-14): 포트 지정 + 관절 맵 캘리브 + 실시간 확인.

리더는 **읽기만** 한다 (LeaderBus). 토크/레지스터를 건드리지 않는다.

리더 연결 후 순서:
    python3 tools/so101_leader.py ports                       # 새로 보이는 포트 (팔로워 2대 제외)
    python3 tools/so101_leader.py assign --arm b --port /dev/serial/by-id/usb-...
    python3 tools/so101_leader.py capture B --arm b           # 자세마다 (팔로워와 같은 6자세)
    python3 tools/so101_leader.py capture A --arm b
    python3 tools/so101_leader.py capture C --arm b
    python3 tools/so101_leader.py capture D --arm b
    python3 tools/so101_leader.py capture GC --arm b          # 리더 그리퍼(손잡이) 완전히 닫음
    python3 tools/so101_leader.py capture GO --arm b          # 완전히 엶
    python3 tools/so101_leader.py solve --arm b               # -> config/so101_leader_b.yaml
    python3 tools/so101_leader.py live --arm b                # 리더 각도를 기구학 각도(deg)로 확인

자세 정의는 tools/so101_map_calib.py 와 같다 (B 쭉 편 자세, A ㄱ 자세, C 손목 아래 90°,
D 사용자 자리에서 본인의 오른쪽으로 45°). 리더도 같은 SO-101 기구라 같은 계산을 쓴다.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ros2_ws", "src", "gaze_hri"))
from gaze_hri import arms  # noqa: E402
from gaze_hri.so101_driver import JOINT_NAMES, JointMap, LeaderBus  # noqa: E402
from so101_map_calib import compute_map  # noqa: E402

POSES = ("A", "B", "C", "D", "GC", "GO")


def load_caps(arm):
    path = arms.leader_captures_file(arm)
    return json.load(open(path)) if os.path.exists(path) else {}


def new_ports():
    known = {os.path.realpath(p) for p in arms.ARMS.values()}
    return [p for p in sorted(glob.glob("/dev/serial/by-id/*"))
            if os.path.realpath(p) not in known]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ports", "assign", "capture", "solve", "show", "live"])
    ap.add_argument("pose", nargs="?")
    ap.add_argument("--arm", choices=sorted(arms.ARMS))
    ap.add_argument("--port")
    a = ap.parse_args()

    if a.cmd == "ports":
        ps = new_ports()
        print("팔로워(a/b) 말고 보이는 시리얼 포트:" if ps else "새 포트 없음 — 리더 USB 연결 확인")
        for p in ps:
            print("  ", p)
        return 0
    if not a.arm:
        sys.exit("--arm a|b (이 리더가 조종할 팔로워)")

    if a.cmd == "assign":
        if not a.port or not os.path.exists(a.port):
            sys.exit(f"--port 가 없다: {a.port}")
        if os.path.realpath(a.port) in {os.path.realpath(p) for p in arms.ARMS.values()}:
            sys.exit("그건 팔로워 포트다")
        d = {}
        if os.path.exists(arms.LEADERS_FILE):
            d = yaml.safe_load(open(arms.LEADERS_FILE)) or {}
        d[a.arm] = a.port
        os.makedirs(os.path.dirname(arms.LEADERS_FILE), exist_ok=True)
        yaml.safe_dump(d, open(arms.LEADERS_FILE, "w"))
        print(f"리더 {a.port} -> 팔 {a.arm}  ({arms.LEADERS_FILE})")
        return 0
    if a.cmd == "show":
        print(json.dumps(load_caps(a.arm), indent=1))
        return 0
    if a.cmd == "solve":
        port = arms.leader_port(a.arm) or "?"
        cfg, problems = compute_map(load_caps(a.arm), port)
        cfg["kind"] = "leader"
        out = arms.leader_map_file(a.arm)
        if os.path.exists(out):
            os.rename(out, out.replace(".yaml", time.strftime("_bak_%Y%m%d_%H%M%S.yaml")))
        with open(out, "w") as f:
            f.write("# tools/so101_leader.py solve 결과 (리더 틱 <-> 기구학 각도).\n")
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"저장: {out}")
        return 1 if problems else 0

    port = arms.leader_port(a.arm)
    if not port:
        sys.exit(f"팔 {a.arm} 의 리더 포트가 지정 안 됨 — 먼저 ports / assign")
    bus = LeaderBus(port)
    try:
        if a.cmd == "capture":
            if a.pose not in POSES:
                sys.exit(f"pose 는 {' '.join(POSES)} 중 하나")
            t = bus.ticks()
            caps = load_caps(a.arm)
            caps[a.pose] = t
            path = arms.leader_captures_file(a.arm)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(caps, open(path, "w"))
            print(f"{a.pose}: " + ", ".join(f"{n}={v}" for n, v in zip(JOINT_NAMES, t)))
        elif a.cmd == "live":
            mf = arms.leader_map_file(a.arm)
            if not os.path.exists(mf):
                sys.exit("리더 맵이 없다 — capture 6자세 후 solve")
            c = yaml.safe_load(open(mf))
            m = JointMap(c["zero_ticks"], c["signs"])
            print("Ctrl-C 로 끝. 각도(deg): pan shoulder elbow wrist roll gripper")
            while True:
                q = m.to_rad(bus.ticks())
                print("\r" + " ".join(f"{math.degrees(v):7.1f}" for v in q), end="", flush=True)
                time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
