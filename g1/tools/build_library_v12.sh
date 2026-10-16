#!/bin/bash
# 데모2 라이브러리 v12 (2026-10-16): v11 + 느린 박수
#   v15(v11 학습) 결과 가리키기는 평균 3.9~6.4° 로 따라갔지만 박수는 16~19° — 손이 배꼽 아래로 처지고
#   아래 손 손목 roll(참조 −56°)이 거의 안 돌았다. 초당 1.6회·4회(2.5 s)는 너무 빨라 정책이 못 따라감
#   → 초당 1.0회·3회(3.0 s), 들어올림 10→7 cm. 위치·팔꿈치·손목 제한은 v11 그대로.
#   사용: ART=~/molbwa_g1 TURNS=$ART/loco_turns4 bash g1/tools/build_library_v12.sh
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ART=${ART:-$HOME/molbwa_g1}
L=${OUT:-$ART/lib12}
PY=${PY:-$HOME/miniconda3/envs/g1deploy/bin/python}
OUT=$L bash "$REPO/g1/tools/build_library_v11.sh" > /dev/null
env -u PYTHONPATH $PY "$REPO/g1/motion/hand_ik.py" clap_stack "$L/clips/celebrate.csv" --claps 3 --hz 1.0 --fwd 0.38 --left 0.16 \
  --up 0.06 --gap 0.03 --lift 0.07 --elbow_out 0.9 --wrist_lock 0.35
python3 -I "$REPO/g1/motion/build_motion_library.py" --segments "$L/segments.yaml" --out_csv "$L/molbwa_library12.csv" \
  --out_meta "$L/library_meta.json" --seam_joint 0.2 | grep -A5 '"seam"'
rm -f "$L/molbwa_library11.csv"
ls "$L"
