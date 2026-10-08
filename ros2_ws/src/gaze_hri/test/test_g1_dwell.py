"""LabelDwell: 라벨 유지 기반 응시 확정. ROS 없이 돈다."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gaze_hri.g1_dwell import LabelDwell  # noqa: E402

DT = 1.0 / 30.0


def run(d, segments, t0=0.0):
    """segments = [(label, 초), ...] 를 30 Hz 로 먹이고 확정된 (t, label) 목록을 돌려준다."""
    out, t = [], t0
    for label, dur in segments:
        for _ in range(int(round(dur * 30))):
            t += DT
            r = d.update(t, label)
            if r is not None:
                out.append((t, r))
    return out


def test_confirms_after_dwell():
    d = LabelDwell()
    fired = run(d, [("g1_face", 1.5)])
    assert len(fired) == 1
    assert fired[0][1] == "g1_face"
    assert 0.95 <= fired[0][0] <= 1.1


def test_progress_rises_then_zero_after_confirm():
    d = LabelDwell()
    run(d, [("g1_face", 0.5)])
    assert 0.3 < d.progress(0.5) < 0.7
    run(d, [("g1_face", 0.6)], t0=0.5)
    assert d.progress(1.1) == 0.0


def test_blink_tolerated():
    d = LabelDwell()
    fired, t = [], 0.0
    for i in range(60):            # 매 5번째 샘플이 None (20%)
        t += DT
        r = d.update(t, None if i % 5 == 4 else "g1_hand")
        if r:
            fired.append(r)
    assert fired == ["g1_hand"]


def test_switch_resets():
    d = LabelDwell()
    fired = run(d, [("g1_face", 0.6), ("g1_hand", 0.6)])
    assert all(lbl != "g1_face" for _, lbl in fired)


def test_no_retrigger_while_staring():
    d = LabelDwell()
    fired = run(d, [("g1_face", 3.0)])
    assert len(fired) == 1


def test_retrigger_after_leave():
    # 이탈 판정은 창 안 비율 < 0.3 이라 1 s 창에서 0.7 s 이상 떠나야 한다 (계획서 0.5 s 는 해제 불가).
    d = LabelDwell()
    fired = run(d, [("g1_face", 1.2), (None, 0.8), ("g1_face", 1.5)])
    assert [lbl for _, lbl in fired] == ["g1_face", "g1_face"]
    assert fired[1][0] >= fired[0][0] + 2.0          # cooldown 지난 뒤


def test_short_leave_does_not_release():
    d = LabelDwell()
    fired = run(d, [("g1_face", 1.2), (None, 0.5), ("g1_face", 3.0)])
    assert len(fired) == 1


def test_other_label_after_cooldown():
    d = LabelDwell()
    fired = run(d, [("g1_face", 1.2), ("g1_hand", 2.5)])
    assert [lbl for _, lbl in fired] == ["g1_face", "g1_hand"]
    assert fired[1][0] >= fired[0][0] + 2.0


def test_gap_resets_window():
    d = LabelDwell()
    run(d, [("g1_face", 0.9)])
    # 브리지가 1초 끊긴 뒤 한 샘플 -> 옛 샘플로 확정되면 안 된다
    assert d.update(2.0, "g1_face") is None


def test_next_seq_strictly_increasing_int():
    from gaze_hri.g1_dwell import next_seq
    a = next_seq(0, 1000.0)
    b = next_seq(a, 1000.0)                          # 같은 밀리초 -> +1
    c = next_seq(b, 999.0)                           # 시계가 뒤로 가도 증가
    d = next_seq(0, 1000.0, jetson_seq=10 ** 13)     # Jetson 이 더 큰 seq 를 봤으면 그 위로
    assert a == 1_000_000 and b == a + 1 and c == b + 1 and d == 10 ** 13 + 1
    assert all(type(x) is int for x in (a, b, c, d))
