import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))

from segment_player import SegmentPlayer  # noqa: E402

SEGS = {"idle": (0, 10), "turn_l90": (10, 15), "bow": (15, 18),
        "handshake": (18, 22), "open_arms": (22, 25)}


def act(seq, part="g1_face", turn_bin=90):
    return {"seq": seq, "cmd": "act", "part": part, "turn_bin": turn_bin}


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
    assert p.command(act(1))
    states = []
    out = []
    for _ in range(5 + 3 + 1):
        out.append(p.step())
        states.append(p.state)
    assert out[0] == (10, True)                 # 다음 step 에서 즉시 turn 시작
    assert [k for k, _ in out[:5]] == list(range(10, 15))
    assert out[5] == (15, True)                 # bow 시작
    assert [k for k, _ in out[5:8]] == [15, 16, 17]
    assert out[8] == (0, True)                  # idle 복귀
    assert states == ["turning"] * 5 + ["acting"] * 3 + ["idle"]
    assert sum(f for _, f in out) == 3


def test_bin0_skips_turn():
    p = SegmentPlayer(SEGS)
    p.step()
    assert p.command(act(1, "g1_hand", 0))
    assert p.step() == (18, True)
    assert p.state == "acting"


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
    assert p.command(act(5, turn_bin=0))
    run(p, 3 + 1)                               # bow 3 + idle 복귀
    assert p.state == "idle"
    assert not p.command(act(5, turn_bin=0))
    assert not p.command(act(3, turn_bin=0))
    assert p.last_seq == 5
    assert p.command(act(6, turn_bin=0))
    assert p.last_seq == 6


def test_ping_ignored():
    p = SegmentPlayer(SEGS)
    assert not p.command({"seq": 1, "cmd": "ping"})
    assert p.last_seq == -1
    assert p.step() == (0, True)


def test_unknown_part_rejected():
    p = SegmentPlayer(SEGS)
    assert not p.command(act(1, "g1_foot"))
    assert p.last_seq == -1


def test_garbage_rejected():
    p = SegmentPlayer(SEGS)
    assert not p.command(None)                  # decode 실패
    assert not p.command(act(1, turn_bin=30))   # bin 밖
    assert not p.command(act(1, turn_bin=-90))  # 라이브러리에 없는 세그먼트
    assert not p.command(act("7"))
    assert not p.command(act(True))
    assert not p.command(act(1, turn_bin=0.0))  # float 0 이 bin 0 으로 통과하면 서버 표 출력이 깨진다
    assert not p.command(act(1, turn_bin=False))
    assert p.last_seq == -1
    assert p.command(act(1))
