import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "motion"))
from amass_g1_to_csv import convert, pad, rate_limit, retime, smooth_env  # noqa: E402
from make_idle_csv import CSV_JOINTS  # noqa: E402


def fake(n=90, order=None):
    names = list(order or CSV_JOINTS)
    q = np.tile(np.arange(29, dtype=np.float32) * 0.01, (n, 1))   # 값 = csv 위치×0.01 (이름 순서 기준)
    q = q[:, [CSV_JOINTS.index(nm) for nm in names]]               # 파일 안에서는 names 순서로 저장
    rot = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 30, 1))
    pos = np.zeros((n, 30, 3), np.float32); pos[:, 0, 2] = 0.75
    return {"fps": np.array([30.0], np.float32), "dof_names": np.array(names), "dof_positions": q,
            "body_positions": pos, "body_rotations": rot}


def test_maps_joints_by_name_not_position():
    shuffled = CSV_JOINTS[::-1]
    out = convert(fake(order=shuffled))
    assert np.allclose(out[0, 7:], np.arange(29) * 0.01)


def test_quat_wxyz_to_xyzw_and_root():
    out = convert(fake())
    assert np.allclose(out[:, 3:7], [0, 0, 0, 1]) and np.allclose(out[:, 2], 0.75)


def test_waist_offset_enveloped():
    out = convert(fake(n=151), waist_yaw_deg=60, ramp_s=1.0)
    j = 7 + CSV_JOINTS.index("waist_yaw_joint")
    base = CSV_JOINTS.index("waist_yaw_joint") * 0.01
    assert np.isclose(out[0, j], base) and np.isclose(out[-1, j], base)
    assert np.isclose(out[75, j], base + np.radians(60), atol=1e-6)


def test_slice_and_fps_guard():
    assert convert(fake(n=90), t0=1.0, t1=2.0).shape[0] == 31
    z = fake(); z["fps"] = np.array([120.0], np.float32)
    with pytest.raises(ValueError):
        convert(z)


def test_env_shape():
    e = smooth_env(61, 30, 1.0)
    assert e[0] == 0 and e[-1] == 0 and e[30] == 1


def test_retime_half_speed_doubles_length_keeps_ends():
    r = convert(fake(n=31))
    r[:, 0] = np.linspace(0, 1, 31)
    s = retime(r, 0.5)
    assert len(s) == 61 and np.isclose(s[0, 0], 0) and np.isclose(s[-1, 0], 1)
    assert np.allclose(np.linalg.norm(s[:, 3:7], axis=1), 1)


def test_rate_limit_turns_step_into_ramp():
    r = convert(fake(n=20)); j = 7 + CSV_JOINTS.index("right_shoulder_yaw_joint")
    r[10:, j] -= 0.9
    o = rate_limit(r, 0.1)
    assert np.abs(np.diff(o[:, j])).max() <= 0.1 + 1e-9 and np.isclose(o[-1, j], r[-1, j])


def test_pad_holds_ends():
    r = convert(fake(n=10)); r[:, 0] = np.arange(10)
    o = pad(r, 1.0)
    assert len(o) == 70 and (o[:30, 0] == 0).all() and (o[-30:, 0] == 9).all()
