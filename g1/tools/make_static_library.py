"""학습 전 서버 스모크용 정지 라이브러리: 참조 0번 프레임(서 있는 자세)을 반복하고 속도 0.

로봇은 계속 서 있기만 한다. 세그먼트 전환·yaw 재정렬·UDP·이벤트 표 경로를 정책 학습 전에
확인하는 용도다. 회전이 없으므로 이벤트 표의 heading 오차는 -bin 이 나와야 정상이다 (RUNBOOK 5-0).

    python3 make_static_library.py --motion ~/g1_dance_deploy/motion_final.npz --out /tmp/g1_static
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--motion", default=os.path.expanduser("~/g1_dance_deploy/motion_final.npz"))
ap.add_argument("--out", default="/tmp/g1_static")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

M = dict(np.load(args.motion))
L0 = M["joint_pos"].shape[0]
segs = [("idle", 150)] + [(f"turn_l{d}", 100) for d in (45, 90, 135, 180)] \
    + [(f"turn_r{d}", 100) for d in (45, 90, 135)] + [("bow", 120), ("handshake", 120), ("open_arms", 120)]
N = sum(n for _, n in segs)
out = {}
for k, v in M.items():
    v = np.asarray(v)
    if v.ndim >= 1 and v.shape[0] == L0:
        out[k] = np.zeros((N,) + v.shape[1:], v.dtype) if "vel" in k else np.repeat(v[:1], N, axis=0)
    else:
        out[k] = v
np.savez(os.path.join(args.out, "static_library.npz"), **out)
meta = {"fps_csv": 30, "fps_npz": 50, "segments": {}, "turn_bins": {}}
s = 0
for n, l in segs:
    meta["segments"][n] = [s, s + l]
    s += l
json.dump(meta, open(os.path.join(args.out, "static_library_meta.json"), "w"), indent=1)
print(f"{N} 프레임, 세그먼트 {len(segs)}개 -> {args.out}")
