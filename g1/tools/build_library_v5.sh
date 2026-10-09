#!/bin/bash
# 데모2 라이브러리 v5 재현 (2026-10-09): 대기 + 보행정책 회전 23개(15° 간격) + 손 흔들기 + 악수
#   회전: g1/motion/record_loco_turn.py 녹화본(도착 후 대기 0.1 s), 앞뒤 패딩 0.2 s
#   손 흔들기(BMLmovi 48_F_2): 팔이 움직이기 직전(0.25 s)부터, 상체만(다리·골반 = 대기자세), 앞 패딩 0·뒤 0.5 s
#   블렌드 전부 0.5 s. 보행 클립은 걸음 자체가 ~0.18 rad/프레임이라 --seam_joint 0.2
#   악수(KIT shake_hand04): 팔이 움직이기 직전 0.25 s ~ 4.7 s, 상체만, 앞 패딩 0·뒤 0.5 s
#   사용: ART=~/molbwa_g1 TURNS=$ART/loco_turns4 bash g1/tools/build_library_v5.sh
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ART=${ART:-$HOME/molbwa_g1}
TURNS=${TURNS:-$ART/loco_turns4}
L=${OUT:-$ART/lib5}
WAVE_NPZ=${WAVE_NPZ:-$ART/amass_g1_raw/BMLmovi_Subject_48_F_MoSh_Subject_48_F_2_poses_120_jpos.npz}
rm -rf "$L"; mkdir -p "$L/clips"
cp "$ART/idle_synth.csv" "$L/clips/idle.csv"
cp "$TURNS"/turn_*.csv "$L/clips/"
python3 -I "$REPO/g1/motion/amass_g1_to_csv.py" "$WAVE_NPZ" "$L/clips/wave.csv" --t0 0.25 --upper_only --pad_start 0 --pad_end 0.5
python3 -I "$REPO/g1/motion/amass_g1_to_csv.py" "${SHAKE_NPZ:-$ART/amass_g1_raw/KIT_291_shake_hand04_poses_100_jpos.npz}" "$L/clips/handshake.csv" \
  --t0 0.25 --t1 4.7 --upper_only --pad_start 0 --pad_end 0.5
python3 -I - "$REPO/g1/motion" "$L/clips" <<'EOF'
import glob, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from amass_g1_to_csv import pad
fs = sorted(glob.glob(sys.argv[2] + "/turn_*.csv"))
for f in fs:
    np.savetxt(f, pad(np.loadtxt(f, delimiter=","), 0.2), delimiter=",", fmt="%.9f")
print(len(fs), "turns padded 0.2 s")
EOF
sed -e 's|^blend_s: .*|blend_s: 0.5|' "$REPO/g1/motion/segments.yaml" > "$L/segments.yaml"
python3 -I "$REPO/g1/motion/build_motion_library.py" --segments "$L/segments.yaml" --out_csv "$L/molbwa_library5.csv" \
  --out_meta "$L/library_meta.json" --seam_joint 0.2 | grep -A5 '"seam"'
ls "$L"
