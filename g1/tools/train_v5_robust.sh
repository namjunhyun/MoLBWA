#!/bin/bash
# 데모2 v5 (2026-10-09): 1단계 악수 추가 학습(밀기 ±0.8, 2000 iter) → 2단계 외부 힘 강화(밀기 ±1.2, 3000 iter)
#   처음부터 ±1.0 은 학습이 안 섰다(2026-10-08) — 서는 법을 익힌 정책에서 올리는 커리큘럼으로 간다.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$HOME/whole_body_tracking/logs/rsl_rl/g1_flat
echo "=== stage A $(date)"
MOTION=molbwa_library5.npz LOAD_RUN=${LOAD_RUN:-2026-10-09_22-00-50} LOAD_CKPT=${LOAD_CKPT:-model_11000.pt} ITERS=2000 PUSH=0.8 bash "$HERE/train_library.sh"
A=$(ls -t "$LOGS" | head -1); CK=$(ls -t "$LOGS/$A" | grep model_ | head -1)
echo "=== stage B from $A/$CK $(date)"
MOTION=molbwa_library5.npz LOAD_RUN=$A LOAD_CKPT=$CK ITERS=3000 PUSH=1.2 bash "$HERE/train_library.sh"
echo "=== done $(date)"
