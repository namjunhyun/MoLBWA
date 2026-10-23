#!/bin/bash
# G1 부위 YOLO 학습 (합성 데이터). ~/yolo-env = /mnt/data/yolo-env (isaaclab torch 2.7 cu128 을 빌린 venv + ultralytics)
#   사용: DATA=/mnt/data/molbwa_yolo bash g1/tools/train_yolo_parts.sh
set -e
: "${DATA:?}"
cd "$DATA"
exec nice ~/yolo-env/bin/yolo detect train data="$DATA/data.yaml" model="${MODEL:-yolo11s.pt}" imgsz=640 \
  epochs="${EPOCHS:-40}" batch="${BATCH:-32}" device=0 workers=4 project="$DATA/runs" name="${NAME:-g1_parts_synth}" exist_ok=True
