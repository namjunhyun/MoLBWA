#!/bin/bash
# G1 부위 YOLO 합성 데이터셋 (train 4×800 병렬 + val 400) → $OUT/{train,val} + data.yaml
#   사용: LIB=~/molbwa_g1/lib5/molbwa_library5.csv OUT=/mnt/data/molbwa_yolo bash g1/tools/make_yolo_dataset.sh
set -e
: "${LIB:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HOME/miniconda3/envs/g1deploy/bin/python}
N=${N:-800}
for s in 1 2 3 4; do
  nice $PY -I "$HERE/synth_parts.py" --lib "$LIB" --out "$OUT/train" --n $N --seed $s --bg random &
done
nice $PY -I "$HERE/synth_parts.py" --lib "$LIB" --out "$OUT/val" --n $((N / 2)) --seed 9 --bg random &
wait
cat > "$OUT/data.yaml" <<EOF
path: $OUT
train: train/images
val: val/images
names: {0: g1_face, 1: g1_hand, 2: g1_torso}
EOF
echo "train $(ls $OUT/train/images | wc -l) val $(ls $OUT/val/images | wc -l)"
