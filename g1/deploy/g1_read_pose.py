"""젯슨: 지금 관절각(29개)을 0.5초 평균해 JSON 으로 출력. 읽기만 한다(명령 발행 없음). Python 3.8.

    python3 g1_read_pose.py > pose.json

정책 서버가 돌 때는 DDS 를 새로 열면 정책 루프가 깨진다(2026-10-12) → 그때는 거부한다.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
HERE = os.path.dirname(os.path.abspath(__file__))

if subprocess.run(["pgrep", "-f", "g1_motion_server"], stdout=subprocess.DEVNULL).returncode == 0:
    sys.exit("거부: 정책 서버가 돌고 있다 — DDS 를 새로 열지 않는다")
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_  # noqa: E402

out = subprocess.run(["ip", "-4", "-o", "addr"], stdout=subprocess.PIPE, universal_newlines=True).stdout
ChannelFactoryInitialize(0, next(f.split()[1] for f in out.splitlines() if " 192.168.123." in f))
qs = []
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(lambda m: qs.append([m.motor_state[i].q for i in range(29)]), 10)
t0 = time.time()
while time.time() - t0 < 3 and len(qs) < 250:
    time.sleep(0.05)
if not qs:
    sys.exit("lowstate 를 못 받았다")
print(json.dumps({"n": len(qs), "q": [round(sum(c) / len(c), 4) for c in zip(*qs[-250:])]}))
