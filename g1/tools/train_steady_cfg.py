"""train_steady.py 가 앱을 띄운 뒤 import 하는 env cfg (isaaclab 은 AppLauncher 전엔 import 불가)."""
import os

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatWoStateEstimationDelayEnvCfg


@configclass
class SteadyCfg(G1FlatWoStateEstimationDelayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_ang_vel_xy = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-float(os.environ.get("STEADY_W", "0.05")))
        # 팔 관절각 직접 추종 (ARM_W > 0 일 때). 2026-10-15: 팔은 링크 위치(std 0.3 m)·자세(0.4 rad)로만 느슨하게
        # 따라가서 손목 roll(손바닥 위) 참조 55° 에 실제 14°, 박수 왼어깨 roll −42° 에 8° — 다리처럼 관절각을 보상한다.
        arm_w = float(os.environ.get("ARM_W", "0"))
        if arm_w > 0:
            self.rewards.motion_arm_joint = RewTerm(
                func=mdp.motion_joint_position_error_exp, weight=arm_w,
                params={"command_name": "motion", "std": float(os.environ.get("ARM_STD", "0.3")),
                        "joint_names": [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]})
