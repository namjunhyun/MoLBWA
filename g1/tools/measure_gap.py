"""라이브러리에서 '회전 멈춤 → 손 흔들기 팔 움직임 시작' 공백(초)을 잰다. 서버는 turn_* 다음 바로 wave 를 재생한다.
    python3 measure_gap.py LIB_DIR [turn_l30 turn_r105 ...]
공백 = (turn 세그먼트 끝 − yaw 속도 < 5°/s 로 멈춘 시점) + (wave 세그먼트 시작 → 오른팔 관절속도 > 0.5 rad/s 시점).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motion"))
from make_idle_csv import CSV_JOINTS  # noqa: E402

lib_dir = sys.argv[1]
turns = sys.argv[2:] or ["turn_l30", "turn_l90", "turn_r105", "turn_l180"]
meta = json.load(open(f"{lib_dir}/library_meta.json"))
seg = meta["segments_csv"]
csv = [f for f in os.listdir(lib_dir) if f.startswith("molbwa_library") and f.endswith(".csv")][0]
L = np.loadtxt(f"{lib_dir}/{csv}", delimiter=",")
fps = 30.0
arm = [7 + CSV_JOINTS.index(n) for n in CSV_JOINTS if n.startswith("right_shoulder") or n.startswith("right_elbow")]
ws, we = seg["wave"]
v = np.abs(np.diff(L[ws:we, arm], axis=0)).max(1) * fps
onset = int(np.argmax(v > 0.5)) / fps
for t in turns:
    s, e = seg[t]
    x, y, z, w = L[s:e, 3:7].T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    yd = np.degrees(np.abs(np.gradient(yaw))) * fps
    moving = np.where(yd > 5)[0]
    stop = (moving[-1] + 1) / fps if len(moving) else 0.0
    tail = (e - s) / fps - stop
    print(f"{t:10s} 회전 멈춘 뒤 세그먼트 끝까지 {tail:.2f}s + 손 흔들기 시작까지 {onset:.2f}s = 공백 {tail + onset:.2f}s")
