#!/bin/bash
# 흔들림 페널티 이어 학습 (train_library.sh + 골반 롤·피치 각속도² 페널티, 가중치 STEADY_W 기본 0.05). train_steady.py 참고
#   사용: MOTION=molbwa_library5.npz LOAD_RUN=2026-10-09_22-45-27 LOAD_CKPT=model_15998.pt ITERS=2000 PUSH=1.2 bash g1/tools/train_steady.sh
#   다른 GPU 작업이 없을 때만 (RUNBOOK 0). env 2048 = RAM 16GB 에서 안전한 값.
set -e
: "${MOTION:?}" "${LOAD_RUN:?}" "${LOAD_CKPT:?}"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd ~/whole_body_tracking
exec nice ~/bin/isaaclab_run.sh -p "$REPO/g1/tools/train_steady.py" \
  --task=Tracking-Flat-G1-Wo-State-Estimation-Delay-Steady-v0 --motion_file "motions/$MOTION" --num_envs "${NENV:-2048}" --headless \
  --max_iterations "${ITERS:-3000}" \
  env.rewards.action_rate_l2.weight=-0.1 env.actions.joint_pos.max_delay_steps=3 \
  agent.save_interval=${SAVE_EVERY:-500} agent.resume=true agent.load_run="$LOAD_RUN" agent.load_checkpoint="$LOAD_CKPT" \
  "env.events.push_robot.params.velocity_range.x=[-${PUSH:-0.8},${PUSH:-0.8}]" \
  "env.events.push_robot.params.velocity_range.y=[-${PUSH:-0.8},${PUSH:-0.8}]"
