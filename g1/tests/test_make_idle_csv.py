import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "motion"))
from make_idle_csv import CSV_JOINTS, idle_rows  # noqa: E402

META = os.path.join(os.path.dirname(__file__), "..", "deploy", "g1_tracking_policy_meta.json")


def test_joints_mapped_by_name():
    meta = json.load(open(META))
    dflt = dict(zip(meta["joint_names"], meta["default_joint_pos"]))
    rows = idle_rows(dflt, 2.0)
    assert rows.shape == (61, 36)
    for n in ("left_knee_joint", "right_elbow_joint", "right_shoulder_roll_joint", "left_hip_pitch_joint"):
        assert np.allclose(rows[:, 7 + CSV_JOINTS.index(n)], dflt[n])
    # Isaac 순서 2번째는 right_hip_pitch 인데 csv 2번째는 left_hip_roll 이다 — 위치로 옮겼다면 여기서 걸린다
    assert np.isclose(rows[0, 7 + 1], dflt["left_hip_roll_joint"])


def test_upright_quat_xyzw():
    meta = json.load(open(META))
    rows = idle_rows(dict(zip(meta["joint_names"], meta["default_joint_pos"])), 1.0)
    assert np.allclose(rows[:, 3:7], [0, 0, 0, 1])
