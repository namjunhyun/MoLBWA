import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g1_protocol import decode, encode, plan  # noqa: E402


@pytest.mark.parametrize("b,expect", [
    (0, ["wave_y0"]), (14, ["wave_y0"]), (-14, ["wave_y0"]), (20, ["wave_y30"]), (59, ["wave_y60"]),
    (75, ["wave_y60"]), (-75, ["wave_y-60"]),
    (76, ["turn_l180", "wave_y-60"]),           # r = -104 -> -60 으로 잘림
    (180, ["turn_l180", "wave_y0"]), (-180, ["turn_l180", "wave_y0"]),
    (179, ["turn_l180", "wave_y0"]), (-179, ["turn_l180", "wave_y0"]),
    (150, ["turn_l180", "wave_y-30"]), (-150, ["turn_l180", "wave_y30"]),
    (360 + 20, ["wave_y30"]), (None, ["wave_y0"]),
])
def test_plan(b, expect):
    assert plan("g1_face", b) == expect


def test_plan_parts_and_bad_part():
    assert plan("g1_hand", 30) == ["handshake_y30"]
    assert plan("g1_torso", -30) == ["open_arms_y-30"]
    with pytest.raises(ValueError):
        plan("g1_foot", 0)
    with pytest.raises(ValueError):
        plan(None, 0)


def test_segments_yaml_matches_plan():
    """segments.yaml 이 plan() 이 낼 수 있는 이름 + idle 과 정확히 같아야 촬영 목록이 빠지지 않는다."""
    import yaml
    from g1_protocol import PART_TO_SEGMENT
    path = os.path.join(os.path.dirname(__file__), "..", "motion", "segments.yaml")
    names = {s["name"] for s in yaml.safe_load(open(path))["segments"]}
    need = {n for p in PART_TO_SEGMENT for b in [None] + list(range(-180, 181)) for n in plan(p, b)}
    assert names == need | {"idle"}


def test_roundtrip_and_garbage():
    m = {"seq": 3, "cmd": "act", "part": "g1_face", "bearing_deg": -90}
    assert decode(encode(m)) == m
    assert decode(b"\xff\x00{") is None
    assert decode(b"[1,2]") is None
