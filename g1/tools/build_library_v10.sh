#!/bin/bash
# 데모2 라이브러리 v10 (2026-10-15, v9 + 쉬운 가리키기): v7 에서 point·celebrate 를 손바닥 방향까지 맞춘 IK 클립으로 교체
#   point     = 64_F_4 (0.8배속, 관절 변화 상한 0.1 rad/프레임, 어깨 −55°) → hand_ik palm_up (오른손 손바닥 위)
#   celebrate = hand_ik clap_stack, 위치를 몸 앞 52 cm·왼쪽 5 cm·아래 5 cm·손 간격 3 cm 로 — v8 은 왼팔이 몸통을
#               10 cm 관통하는 자세라 학습 시뮬(자기 충돌 켜짐)에서 따라가지 못했다
#   사용: ART=~/molbwa_g1 TURNS=$ART/loco_turns4 bash g1/tools/build_library_v10.sh
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ART=${ART:-$HOME/molbwa_g1}
L=${OUT:-$ART/lib10}
R=$ART/amass_g1_raw
PY=${PY:-$HOME/miniconda3/envs/g1deploy/bin/python}
OUT=$L POINT_NPZ=$R/BMLmovi_Subject_64_F_MoSh_Subject_64_F_4_poses_120_jpos.npz POINT_T0=0 POINT_T1=5.25 POINT_SPEED=0.8 \
  POINT_EXTRA="--vmax 0.1 --arm_scale right:0.65,-0.3" \
  CHEER_NPZ=$R/BMLmovi_Subject_59_F_MoSh_Subject_59_F_20_poses_120_jpos.npz CHEER_T0=0 CHEER_T1=4.7 CHEER_EXTRA="--arm_gap 8" \
  bash "$REPO/g1/tools/build_library_v7.sh" > /dev/null
env -u PYTHONPATH $PY "$REPO/g1/motion/hand_ik.py" palm_up "$L/clips/point.csv" "$L/clips/point.csv" --yaw_dir 1 --inside --ramp 1.2 --up_frac 0.7
env -u PYTHONPATH $PY "$REPO/g1/motion/hand_ik.py" clap_stack "$L/clips/celebrate.csv" --fwd 0.52 --left 0.05 --up -0.05 --gap 0.03
python3 -I "$REPO/g1/motion/build_motion_library.py" --segments "$L/segments.yaml" --out_csv "$L/molbwa_library10.csv" \
  --out_meta "$L/library_meta.json" --seam_joint 0.2 | grep -A5 '"seam"'
rm -f "$L/molbwa_library6.csv" "$L/molbwa_library7.csv"
ls "$L"
