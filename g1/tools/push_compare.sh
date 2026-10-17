#!/bin/bash
# 정책별 외부 힘 내성 비교 (MuJoCo, 대기 중 골반 수평 밀기 0.2 s, 기본 가드: 토크 0.6 + 넘어짐 50°)
#   단발 FORCES(기본 150 200 250) N (t=4 s 한 번) + 반복 150 N (4 s 마다, 25 s)
#   사용: LIB=$ART/lib5 MOTION=molbwa_library5.npz bash g1/tools/push_compare.sh policy_a.npz policy_b.npz ...
set -u
: "${LIB:?}" "${MOTION:?}"
ART=${ART:-$HOME/molbwa_g1}
cd "$(dirname "$0")/../deploy"
for P in "$@"; do
  tag=$(basename "$P" .npz)
  S="env -u PYTHONPATH -u LD_LIBRARY_PATH nice $HOME/miniconda3/envs/g1deploy/bin/python g1_motion_server.py --policy $P --motion $HOME/whole_body_tracking/motions/$MOTION --library_meta $LIB/library_meta.json --backend mujoco --hold_sec 1"
  for N in ${FORCES:-150 200 250}; do
    $S --run_sec 7.9 --sim_push $N --sim_push_every 4 > "$ART/pushs_${tag}_$N.log" 2>&1
    echo "$tag 단발 $N N: $(grep -E '밀기 [0-9]|\[중단\]' "$ART/pushs_${tag}_$N.log" | sed 's/.*→ //' | tr '\n' ' ')"
  done
  $S --run_sec 25 --sim_push 150 --sim_push_every 4 > "$ART/pushr_${tag}_150.log" 2>&1
  echo "$tag 반복 150 N: $(grep -E '밀기 [0-9]|\[중단\]|1초창' "$ART/pushr_${tag}_150.log" | sed 's/.*→ //;s/ *토크 사용률/토크/' | tr '\n' ' ')"
done
