#!/bin/bash
# 데모2 라이브러리 v8 (2026-10-14): v7 에서 point·celebrate 를 손바닥 방향까지 맞춘 IK 클립으로 교체
#   point     = 64_F_4 (0.8배속, 관절 변화 상한 0.1 rad/프레임, 어깨 −55°) → hand_ik palm_up (오른손 손바닥 위)
#   celebrate = hand_ik clap_stack (왼손 손바닥 위 받침 + 오른손 손바닥 아래로 4회 내려치기, 1.6 Hz)
#   사용: ART=~/molbwa_g1 TURNS=$ART/loco_turns4 bash g1/tools/build_library_v8.sh
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ART=${ART:-$HOME/molbwa_g1}
L=${OUT:-$ART/lib8}
R=$ART/amass_g1_raw
PY=${PY:-$HOME/miniconda3/envs/g1deploy/bin/python}
OUT=$L POINT_NPZ=$R/BMLmovi_Subject_64_F_MoSh_Subject_64_F_4_poses_120_jpos.npz POINT_T0=0 POINT_T1=5.25 POINT_SPEED=0.8 \
  POINT_EXTRA="--vmax 0.1 --arm_scale right:0.65,-0.3" \
  CHEER_NPZ=$R/BMLmovi_Subject_59_F_MoSh_Subject_59_F_20_poses_120_jpos.npz CHEER_T0=0 CHEER_T1=4.7 CHEER_EXTRA="--arm_gap 8" \
  bash "$REPO/g1/tools/build_library_v7.sh" > /dev/null
env -u PYTHONPATH $PY "$REPO/g1/motion/hand_ik.py" palm_up "$L/clips/point.csv" "$L/clips/point.csv"
env -u PYTHONPATH $PY "$REPO/g1/motion/hand_ik.py" clap_stack "$L/clips/celebrate.csv"
python3 -I "$REPO/g1/motion/build_motion_library.py" --segments "$L/segments.yaml" --out_csv "$L/molbwa_library8.csv" \
  --out_meta "$L/library_meta.json" --seam_joint 0.2 | grep -A5 '"seam"'
rm -f "$L/molbwa_library6.csv" "$L/molbwa_library7.csv"
ls "$L"
