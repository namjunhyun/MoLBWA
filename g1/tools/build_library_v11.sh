#!/bin/bash
# 데모2 라이브러리 v11 (2026-10-16): v10 + 자연스러운 가리키기·박수 (IK 로 처음부터, 사용자 "모션 이상함" 반영)
#   point     = hand_ik point_fwd: 어깨에서 정면 30° 아래로 58 cm, 오른쪽 12 cm, 손바닥 위(85%), 손목 곧게(±0.2 rad)
#               올림 1.2 s · 유지 2.5 s · 내림 1.2 s — v10 은 팔이 몸을 가로지르고 손목이 꺾여 '좀비 팔'로 보였다
#   celebrate = hand_ik clap_stack: 몸 앞 38 cm·왼쪽 16 cm·6 cm 위, 팔꿈치 바깥(0.9 rad), 손목 ±0.35 rad
#               v10 은 52 cm 앞으로 팔을 쭉 뻗어 쟁반 든 모양이었다. 왼위팔 몸통 여유 +0.9 cm
#   사용: ART=~/molbwa_g1 TURNS=$ART/loco_turns4 bash g1/tools/build_library_v11.sh
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ART=${ART:-$HOME/molbwa_g1}
L=${OUT:-$ART/lib11}
PY=${PY:-$HOME/miniconda3/envs/g1deploy/bin/python}
OUT=$L bash "$REPO/g1/tools/build_library_v10.sh" > /dev/null
env -u PYTHONPATH $PY "$REPO/g1/motion/hand_ik.py" point_fwd "$L/clips/point.csv" --reach 0.58 --down 30 --out 0.12
env -u PYTHONPATH $PY "$REPO/g1/motion/hand_ik.py" clap_stack "$L/clips/celebrate.csv" --fwd 0.38 --left 0.16 --up 0.06 --gap 0.03 \
  --lift 0.10 --elbow_out 0.9 --wrist_lock 0.35
python3 -I "$REPO/g1/motion/build_motion_library.py" --segments "$L/segments.yaml" --out_csv "$L/molbwa_library11.csv" \
  --out_meta "$L/library_meta.json" --seam_joint 0.2 | grep -A5 '"seam"'
rm -f "$L/molbwa_library10.csv"
ls "$L"
