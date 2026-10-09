import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))

from segment_player import SegmentPlayer  # noqa: E402

SEGS = {"idle": (0, 10), "turn_l180": (10, 15), "wave_y0": (15, 18), "wave_y-30": (25, 28),
        "handshake_y30": (18, 22), "open_arms_y0": (22, 25)}


def act(seq, part="g1_face", bearing_deg=150):
    return {"seq": seq, "cmd": "act", "part": part, "bearing_deg": bearing_deg}


def run(p, n):
    return [p.step() for _ in range(n)]


def test_idle_loops():
    p = SegmentPlayer(SEGS)
    out = run(p, 25)
    assert [k for k, _ in out] == list(range(10)) * 2 + list(range(5))
    assert all(f == (k == 0) for k, f in out)
    assert p.state == "idle"


def test_act_turn_then_gesture_then_idle():
    p = SegmentPlayer(SEGS)
    run(p, 3)                                   # idle 중간
    assert p.command(act(1))                    # 150° -> turn_l180 + wave_y-30
    states = []
    out = []
    for _ in range(5 + 3 + 1):
        out.append(p.step())
        states.append(p.state)
    assert out[0] == (10, True)                 # 다음 step 에서 즉시 turn 시작
    assert [k for k, _ in out[:5]] == list(range(10, 15))
    assert out[5] == (25, True)                 # wave_y-30 시작
    assert [k for k, _ in out[5:8]] == [25, 26, 27]
    assert out[8] == (0, True)                  # idle 복귀
    assert states == ["turning"] * 5 + ["acting"] * 3 + ["idle"]
    assert sum(f for _, f in out) == 3


def test_waist_only_skips_turn():
    p = SegmentPlayer(SEGS)
    p.step()
    assert p.command(act(1, "g1_hand", 30))
    assert p.step() == (18, True)
    assert p.state == "acting"


def test_no_bearing_is_y0():
    p = SegmentPlayer(SEGS)
    p.step()
    assert p.command(act(1, "g1_torso", None))
    assert p.step() == (22, True)


def test_busy_ignored():
    p = SegmentPlayer(SEGS)
    p.step()
    assert p.command(act(1))
    assert not p.command(act(2))                # 수락했지만 아직 시작 전 -> 이미 바쁨
    p.step()
    assert p.state == "turning"
    assert not p.command(act(3))
    assert p.last_seq == 1


def test_seq_duplicate_and_old_ignored():
    p = SegmentPlayer(SEGS)
    p.step()
    assert p.command(act(5, bearing_deg=0))
    run(p, 3 + 1)                               # wave_y0 3 + idle 복귀
    assert p.state == "idle"
    assert not p.command(act(5, bearing_deg=0))
    assert not p.command(act(3, bearing_deg=0))
    assert p.last_seq == 5
    assert p.command(act(6, bearing_deg=0))
    assert p.last_seq == 6


def test_ping_ignored():
    p = SegmentPlayer(SEGS)
    assert not p.command({"seq": 1, "cmd": "ping"})
    assert p.last_seq == -1
    assert p.step() == (0, True)


def test_unknown_part_rejected():
    p = SegmentPlayer(SEGS)
    assert not p.command(act(1, "g1_foot"))
    assert not p.command(act(1, ["g1_face"]))   # 해시 불가 타입도 예외 없이 거부
    assert p.last_seq == -1


def test_garbage_rejected():
    p = SegmentPlayer(SEGS)
    assert not p.command(None)                  # decode 실패
    assert not p.command(act(1, bearing_deg=60))    # wave_y60 이 라이브러리에 없음
    assert not p.command(act(1, bearing_deg=-150))  # turn 은 있어도 wave_y30 이 없으면 통째로 거부
    assert not p.command(act("7"))
    assert not p.command(act(True))
    assert not p.command(act(1, bearing_deg=0.0))   # float 거부 (규약은 int|null)
    assert not p.command(act(1, bearing_deg=False))
    assert not p.command(act(1, bearing_deg="0"))
    assert p.last_seq == -1
    assert p.command(act(1))
