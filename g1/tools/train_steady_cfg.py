"""train_steady.py 가 앱을 띄운 뒤 import 하는 env cfg (isaaclab 은 AppLauncher 전엔 import 불가)."""
import os

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatWoStateEstimationDelayEnvCfg


@configclass
class SteadyCfg(G1FlatWoStateEstimationDelayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_ang_vel_xy = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-float(os.environ.get("STEADY_W", "0.05")))
