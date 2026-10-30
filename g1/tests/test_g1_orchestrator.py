import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g1_orchestrator import Orchestrator  # noqa: E402


def _ticks(a, b, dt):
    n = int(round((b - a) / dt))
    return [a + i * dt for i in range(n + 1)]


def _make(busy=False):
    sent = []
    orc = Orchestrator(sent.append, lambda k, b, after: sent.append(("dlg", k, b) + (("after_act",) if after else ())),
                       lambda: busy, dwell_time=1.0)
    return orc, sent


def _acts(sent):
    return [m for m in sent if isinstance(m, dict) and m.get("cmd") == "act"]


def test_face_dwell_does_nothing():
    orc, sent = _make()
    for t in _ticks(0, 3.0, 0.1):
        orc.feed_part(t, "g1_face", -30)
    assert sent == []                                          # 얼굴은 반응 없음 (시연은 화분·손만)


def test_hand_dwell_sends_wave_then_greet():
    orc, sent = _make()
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_part(t, "g1_hand", 10)
    a = _acts(sent)
    assert len(a) == 1 and a[0]["part"] == "g1_face" and a[0]["bearing_deg"] == 10   # 젯슨에서 g1_face = 회전 + wave
    assert sent[-1] == ("dlg", "greet", 10, "after_act")                            # 손 흔들기 끝난 뒤 안부


def test_object_dwell_starts_dialog():
    orc, sent = _make()
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_object(t, "plant", 45)
    assert sent == [("dlg", "plant", 45)]


def test_busy_blocks_all_triggers():
    orc, sent = _make(busy=True)
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_part(t, "g1_hand", -30)
        orc.feed_object(t, "plant", 45)
    assert sent == []


def test_part_priority_over_object():
    orc, sent = _make()
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_part(t, "g1_hand", -30)
        orc.feed_object(t, "plant", 45)
    assert len(_acts(sent)) == 1 and [m for m in sent if isinstance(m, tuple)] == [("dlg", "greet", -30, "after_act")]


def test_face_does_not_block_object():
    orc, sent = _make()
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_part(t, "g1_face", -30)
        orc.feed_object(t, "plant", 45)
    assert sent == [("dlg", "plant", 45)]


def test_label_switch_resets_dwell():
    orc, sent = _make()
    for t in _ticks(0, 0.6, 0.1):
        orc.feed_part(t, "g1_torso", 0)
    for t in _ticks(0.7, 1.2, 0.1):
        orc.feed_part(t, "g1_hand", 0)
    assert _acts(sent) == []


def test_invalid_frames_reset():
    orc, sent = _make()
    for t in _ticks(0, 0.4, 0.1):
        orc.feed_part(t, "g1_hand", 0)
    for t in _ticks(0.5, 0.9, 0.1):
        orc.feed_part(t, None, None)
    for t in _ticks(1.0, 1.2, 0.1):
        orc.feed_part(t, "g1_hand", 0)
    assert sent == []


def test_seq_monotonic():
    orc, sent = _make()
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_part(t, "g1_torso", 0)
    for t in _ticks(1.3, 1.7, 0.1):
        orc.feed_part(t, None, None)
    for t in _ticks(1.8, 5.0, 0.1):
        orc.feed_part(t, "g1_hand", 0)
    s = [m["seq"] for m in _acts(sent)]
    assert len(s) == 2 and s[0] < s[1]
