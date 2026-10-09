#!/bin/bash
# 데모2 정책 체크포인트 평가 (MuJoCo, 로봇 없음): export → 시연 2개(착용자 시점 카메라) + 무작위 6개 + 대기 중 100 N 밀기
#   사용: IT=12000 RUN=<logs/rsl_rl/g1_flat/날짜> LIB=$ART/lib4 MOTION=molbwa_library4.npz TAG=v4 bash g1/tools/eval_demo.sh
set -u
: "${IT:?}" "${RUN:?}" "${LIB:?}" "${MOTION:?}"
TAG=${TAG:-eval}; ART=${ART:-$HOME/molbwa_g1}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CK="$HOME/whole_body_tracking/logs/rsl_rl/g1_flat/$RUN/model_$IT.pt"
until [ -e "$CK" ]; do sleep 20; done; sleep 10
cd "$REPO/g1/deploy"
P="$ART/policy_${TAG}_$IT.npz"
env -u PYTHONPATH nice ~/miniconda3/envs/isaaclab/bin/python export_policy_npz.py --ckpt "$CK" --out "$P" \
    --title "$TAG $IT" --note "데모2 $TAG" > "$ART/export_${TAG}_$IT.log" 2>&1
S="env -u PYTHONPATH -u LD_LIBRARY_PATH nice $HOME/miniconda3/envs/g1deploy/bin/python g1_motion_server.py --policy $P --motion $HOME/whole_body_tracking/motions/$MOTION --library_meta $LIB/library_meta.json --backend mujoco --hold_sec 1"
O="$ART/${TAG}_$IT"
$S --fake_events 1 --fake_bearing 37 --sim_cam_az 217 --sim_video ${O}_demo37.mp4 --log_csv ${O}_demo37.csv > ${O}_demo37.log 2>&1
$S --fake_events 1 --fake_bearing -110 --sim_cam_az 70 --sim_video ${O}_demo-110.mp4 --log_csv ${O}_demo-110.csv > ${O}_demo-110.log 2>&1
$S --fake_events 1 --fake_part g1_hand --fake_bearing -60 --sim_cam_az 120 --sim_video ${O}_shake-60.mp4 --log_csv ${O}_shake-60.csv > ${O}_shake-60.log 2>&1
$S --fake_events 6 --fake_seed 1 > ${O}_rand.log 2>&1
$S --run_sec 25 --sim_push 100 --sim_push_every 4 > ${O}_push100.log 2>&1
for k in demo37 demo-110 shake-60 rand push100; do
  echo "== $TAG $IT $k"; grep -E "판정|중단|\[act\]|heading 오차\||밀기 [0-9]|토크 사용률" ${O}_$k.log | head -10
done
