#!/bin/bash
# 정책 × 방위 6개 × (손 흔들기·악수) 흔들림. 사용: multi_angle.sh <policy.npz> <TAG> <LIBN>
P=$1; TAG=$2; N=$3; PORT=${4:-55070}; A=$HOME/molbwa_g1; R=/home/junhyun/창종설/.claude/worktrees/g1-emotional-demo; O=$A/ma; mkdir -p $O
cd $R/g1/deploy
for part in g1_face g1_hand; do for b in -150 -90 -30 20 75 135; do
  env -u PYTHONPATH -u LD_LIBRARY_PATH nice $HOME/miniconda3/envs/g1deploy/bin/python g1_motion_server.py --policy $P \
    --motion $HOME/whole_body_tracking/motions/molbwa_library$N.npz --library_meta $A/lib$N/library_meta.json --backend mujoco --hold_sec 1 --cmd_port $PORT --state_port $((PORT+100)) \
    --fake_events 1 --fake_part $part --fake_bearing $b --log_csv $O/${TAG}_${part}_$b.csv --sim_pose_log $O/${TAG}_${part}_$b.pose.npz > $O/${TAG}_${part}_$b.log 2>&1
done; done
echo "$TAG done"
