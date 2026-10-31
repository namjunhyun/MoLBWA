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


def test_face_dwell_sends_wave_then_greet():
    orc, sent = _make()
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_part(t, "g1_face", -30)
    a = _acts(sent)
    assert len(a) == 1 and a[0]["part"] == "g1_face" and a[0]["bearing_deg"] == -30   # 젯슨에서 g1_face = 회전 + wave
    assert sent[-1] == ("dlg", "greet", -30, "after_act")                            # 손 흔들기 끝난 뒤 안부


def test_hand_dwell_sends_handshake_only():
    orc, sent = _make()
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_part(t, "g1_hand", 10)
    a = _acts(sent)
    assert len(a) == 1 and a[0]["part"] == "g1_hand" and a[0]["bearing_deg"] == 10   # 젯슨에서 g1_hand = 회전 + handshake
    assert not [m for m in sent if isinstance(m, tuple)]   # 2026-10-31: 악수만 — 반가워요·안부 대화 없음


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
    assert len(_acts(sent)) == 1 and [m for m in sent if isinstance(m, tuple)] == []   # 악수만, 꽃 대화는 부위 우선으로 막힘


def test_face_priority_over_object():
    orc, sent = _make()
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_part(t, "g1_face", -30)
        orc.feed_object(t, "plant", 45)
    assert [m for m in sent if isinstance(m, tuple)] == [("dlg", "greet", -30, "after_act")]


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


def test_reset_allows_same_part_again():
    orc, sent = _make()
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_part(t, "g1_hand", 10)
    for t in _ticks(1.3, 2.6, 0.1):            # 계속 봐도 같은 부위는 떠나기 전 재확정 금지(래치)
        orc.feed_part(t, "g1_hand", 10)
    assert len(_acts(sent)) == 1
    orc.reset()                                 # 초기화 → 래치·누적이 비워져 다시 3초(여기선 1초) 응시하면 또 나간다
    for t in _ticks(2.7, 4.0, 0.1):
        orc.feed_part(t, "g1_hand", 10)
    assert len(_acts(sent)) == 2


def _make_preempt(can=True, ok=True):
    sent, pre = [], []
    orc = Orchestrator(sent.append, lambda k, b, after: sent.append(("dlg", k, b)), lambda: True, dwell_time=1.0,
                       can_preempt=lambda: can, preempt=lambda: pre.append(True) or ok)
    return orc, sent, pre


def test_flower_preempts_greet_dialog():
    # 2026-10-31: 인사 뒤 안부 대화 중이라도 꽃을 3초 보면 대화를 끝내고 꽃 시나리오로 (초기화 없이)
    orc, sent, pre = _make_preempt()
    orc._active_kind = "greet"
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_object(t, "plant", 45)
    assert pre == [True] and sent == [("dlg", "plant", 45)]


def test_no_preempt_while_robot_moving_or_same_dialog_or_parts():
    orc, sent, pre = _make_preempt(can=False)              # 로봇이 말하거나 움직이는 중
    orc._active_kind = "greet"
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_object(t, "plant", 45)
    assert pre == [] and sent == []
    orc, sent, pre = _make_preempt()
    orc._active_kind = "plant"                              # 이미 꽃 대화 중
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_object(t, "plant", 45)
    assert pre == [] and sent == []
    orc, sent, pre = _make_preempt()
    orc._active_kind = "plant"                              # 대화 중 로봇 손·머리는 끼어들지 않는다
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_part(t, "g1_hand", 10)
    assert pre == [] and sent == []


def test_preempt_failed_does_not_start():
    orc, sent, pre = _make_preempt(ok=False)                # 대화가 제때 안 끝나면 시작하지 않는다
    orc._active_kind = "greet"
    for t in _ticks(0, 1.2, 0.1):
        orc.feed_object(t, "plant", 45)
    assert pre and sent == []
