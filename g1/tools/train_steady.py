"""whole_body_tracking train.py 를 고치지 않고 '흔들림 페널티' 태스크를 등록해 돌린다.

    Tracking-Flat-G1-Wo-State-Estimation-Delay-Steady-v0 = 원 태스크 + ang_vel_xy_l2(골반 롤·피치 각속도²) 페널티.
    2026-10-11: 밀기 ±1.2 강건성 학습 뒤 손 흔들기 중 몸통 롤·피치 각속도 RMS 6.4→9.1°/s.
    기존 motion_body_ang_vel 은 std=3.14 rad/s 라 0.16 rad/s 흔들림엔 보상이 거의 1 → 기울기가 없었다.

    사용: ~/bin/isaaclab_run.sh -p g1/tools/train_steady.py <train.py 인자 그대로, --task 는 위 이름>
    가중치: 환경변수 STEADY_W (기본 0.05)
"""
import os
import runpy
import sys

import gymnasium as gym

WBT = os.path.expanduser("~/whole_body_tracking")
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), f"{WBT}/scripts/rsl_rl"]
gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-Delay-Steady-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={   # 문자열 진입점 = 앱(AppLauncher)이 뜬 뒤 hydra 가 import 한다
        "env_cfg_entry_point": "train_steady_cfg:SteadyCfg",
        "rsl_rl_cfg_entry_point": "whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)
os.chdir(WBT)
sys.argv[0] = f"{WBT}/scripts/rsl_rl/train.py"
runpy.run_path(sys.argv[0], run_name="__main__")
