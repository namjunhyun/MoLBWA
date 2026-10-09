#!/bin/bash
# 데모2 라이브러리 정책 이어 학습 (Wo-State-Estimation + 지연 0~60 ms + action_rate -0.1 + 밀기 ±PUSH m/s, 기본 0.8)
#   사용: MOTION=molbwa_library4.npz LOAD_RUN=2026-10-09_13-47-22 LOAD_CKPT=model_10999.pt ITERS=4000 bash g1/tools/train_library.sh
#   다른 GPU 작업이 없을 때만 (RUNBOOK 0). env 2048 = RAM 16GB 에서 안전한 값.
set -e
: "${MOTION:?}" "${LOAD_RUN:?}" "${LOAD_CKPT:?}"
cd ~/whole_body_tracking
exec nice ~/bin/isaaclab_run.sh -p scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-Delay-v0 --motion_file "motions/$MOTION" --num_envs "${NENV:-2048}" --headless \
  --max_iterations "${ITERS:-3000}" \
  env.rewards.action_rate_l2.weight=-0.1 env.actions.joint_pos.max_delay_steps=3 \
  agent.resume=true agent.load_run="$LOAD_RUN" agent.load_checkpoint="$LOAD_CKPT" \
  "env.events.push_robot.params.velocity_range.x=[-${PUSH:-0.8},${PUSH:-0.8}]" \
  "env.events.push_robot.params.velocity_range.y=[-${PUSH:-0.8},${PUSH:-0.8}]"
