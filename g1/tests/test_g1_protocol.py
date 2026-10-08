import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g1_protocol import TURN_BINS, decode, encode, nearest_bin, turn_segment  # noqa: E402


def test_turn_segment():
    assert turn_segment(0) is None
    assert turn_segment(180) == "turn_l180"
    assert turn_segment(45) == "turn_l45"
    assert turn_segment(-45) == "turn_r45"
    assert turn_segment(-135) == "turn_r135"
    for b in TURN_BINS:
        turn_segment(b)


def test_nearest_bin_edges():
    assert nearest_bin(22.4) == 0
    assert nearest_bin(22.6) == 45
    assert nearest_bin(179) == 180
    assert nearest_bin(-179) == 180
    assert nearest_bin(-158) == 180
    assert nearest_bin(-157) == -135
    assert nearest_bin(-90) == -90
    assert nearest_bin(360 + 90) == 90


def test_roundtrip_and_garbage():
    m = {"seq": 3, "cmd": "act", "part": "g1_face", "turn_bin": -90}
    assert decode(encode(m)) == m
    assert decode(b"\xff\x00{") is None
    assert decode(b"[1,2]") is None
