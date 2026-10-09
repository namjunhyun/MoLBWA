import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g1_protocol import decode, encode, plan  # noqa: E402


@pytest.mark.parametrize("b,expect", [
    (0, ["wave"]), (7, ["wave"]), (-7, ["wave"]), (8, ["turn_l15", "wave"]),
    (-100, ["turn_r105", "wave"]), (90, ["turn_l90", "wave"]), (-45, ["turn_r45", "wave"]),
    (179, ["turn_l180", "wave"]), (-179, ["turn_l180", "wave"]), (180, ["turn_l180", "wave"]),
    (-180, ["turn_l180", "wave"]), (-173, ["turn_l180", "wave"]), (-172, ["turn_r165", "wave"]),
    (152, ["turn_l150", "wave"]), (360 + 50, ["turn_l45", "wave"]), (None, ["wave"]),
])
def test_plan(b, expect):
    assert plan("g1_face", b) == expect


def test_plan_parts_and_bad_part():
    assert plan("g1_hand", 90) == ["turn_l90", "handshake"]
    from g1_protocol import TURN_BINS
    assert len(TURN_BINS) == 24 and len(set(TURN_BINS)) == 24
    assert plan("g1_torso", None) == ["open_arms"]
    with pytest.raises(ValueError):
        plan("g1_foot", 0)
    with pytest.raises(ValueError):
        plan(None, 0)


def test_segments_yaml_covers_face_and_hand_plan():
    """segments.yaml 이 g1_face(wave)·g1_hand(handshake) 의 plan() 출력 + idle 과 정확히 같아야 한다.
    open_arms(g1_torso) 는 보류 — 추가하면 그 부위도 여기 넣는다."""
    import yaml
    path = os.path.join(os.path.dirname(__file__), "..", "motion", "segments.yaml")
    names = {s["name"] for s in yaml.safe_load(open(path))["segments"]}
    need = {n for part in ("g1_face", "g1_hand") for b in [None] + list(range(-180, 181)) for n in plan(part, b)}
    assert names == need | {"idle"}


def test_roundtrip_and_garbage():
    m = {"seq": 3, "cmd": "act", "part": "g1_face", "bearing_deg": -90}
    assert decode(encode(m)) == m
    assert decode(b"\xff\x00{") is None
    assert decode(b"[1,2]") is None
