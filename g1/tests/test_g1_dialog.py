import math
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import g1_dialog  # noqa: E402
from g1_dialog import (CLOSINGS_SILENT, CLOSINGS, MAX_TURNS, NOT_HEARD, RISK, SCENARIOS, Dialog, banned,  # noqa: E402
                       fixed_lines, good_question, is_thanks, risk, tidy)
from g1_say import pad_for_player  # noqa: E402


@pytest.fixture(autouse=True)
def no_keys(monkeypatch, tmp_path):
    """실제 키 파일·네트워크를 쓰지 않는다."""
    monkeypatch.setattr(g1_dialog, "ENV", str(tmp_path / "none.env"))
    for k in ("CLOVA_STUDIO_API_KEY", "NCP_CLIENT_ID", "NCP_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)


# ---- 금지어: 글자 포함이 아니라 앞뒤를 본다 ----

@pytest.mark.parametrize("text,bad", [
    ("어르신 마음이 담긴 용돈이라 지은이가 정말 기뻐하겠어요.", False),   # 프롬프트 예시 문장 — '돈' 글자로 걸리면 안 됨
    ("돌잔치 때 돈가스를 드셨다면서요.", False),
    ("손녀분과 약속하신 날이 다가오네요.", False), ("꽃이 약간 시들었네요.", False),
    ("손녀 생일 기억나세요?", True), ("손녀분 이름 기억하세요?", True), ("이분이 누구인지 아세요?", True),
    ("혈압약은 챙기셨어요?", True), ("약을 드실 시간이에요.", True), ("병원 다녀오셨어요?", True),
    ("대출 이야기는 그만해요.", True), ("선거가 다가오네요.", True),
])
def test_banned(text, bad):
    assert banned(text) == bad


def test_tidy():
    assert tidy("안녕하세요, 어르신! 🥰 오늘은 목요일이네요. 날씨도 좋아요.") == "안녕하세요, 어르신! 오늘은 목요일이네요."
    assert tidy("첫 문장이에요. 둘째 문장이에요.", max_sent=1) == "첫 문장이에요."


# ---- 끝맺는 말·위험 신호 ----

@pytest.mark.parametrize("ans,end", [
    ("이야기 들어줘서 고마워", True), ("말동무 해줘서 고맙다", True), ("들어줘서 고마워요.", True),
    ("고맙습니다", True), ("너랑 이야기하니까 좀 낫네. 고마워~", True), ("감사해요!", True),
    ("딸이 전화해 줘서 고마웠지, 그래서 기뻤어", False),      # 중간의 고마움은 이야기의 일부
    ("물 주는 게 낙이지", False), ("이제 됐어", False),
])
def test_is_thanks(ans, end):
    assert is_thanks(ans) == end


@pytest.mark.parametrize("ans,kind", [
    ("그냥 다 귀찮고 죽고 싶어", "life"), ("살기 싫어", "life"), ("이제 그만 살고 싶다", "life"),
    ("요즘 너무 아파", "pain"), ("숨이 차서 힘들어", "pain"), ("쓰러질 것 같아", "pain"),
    ("죽 끓여 먹었어", None), ("아들이 보고 싶어", None), ("조금 아파도 괜찮아", None),
])
def test_risk(ans, kind):
    assert risk(ans) == kind


@pytest.mark.parametrize("q,ok", [
    ("물 주실 때 어떤 기분이세요?", True), ("적적하실 때는 어떤 마음이 드세요?", False), ("적적할 때 뭘 하고 싶어요?", False), ("누가 제일 생각나세요?", True),
    ("요즘 자주 보는 프로그램이 있으신가요?", False), ("오늘 하루도 화분과 함께 잘 보내셨어요?", False),
    ("설레는 순간도 있지 않으신가요?", False), ("요즘 어떻게 지내시는지 궁금하지는 않으세요?", False),
    ("식물은 주로 어떻게 돌봐주시나요?", False), ("손녀 생일이 있지요?", False), ("공감만 하는 말이에요.", False),
    ("손녀분은 어르신께 어떤 의미인가요?", True), ("그때 어떤 마음이셨나요?", True), ("요즘 마음은 좀 어떠세요?", True),
    ("요즘 화분은 잘 자라고 있나요?", False), ("지금도 그때처럼 사랑스럽고 귀여우신가요?", False),
    ("누가 제일 보고 싶으세요? 다 보고 싶으시지요?", False),
])
def test_good_question(q, ok):
    assert good_question(q) == ok


def test_bad_question_is_replaced_keeping_empathy():
    r = Robot()
    run(r, ["TV 보지", "고마워"], gen=lambda m, last: "하루가 길게 느껴지시겠어요. 자주 보는 프로그램이 있으신가요?" if not last else "고마워요.")
    assert r.said[1] == f"하루가 길게 느껴지시겠어요. {SCENARIOS['plant']['follow'][0]}"


def test_closings_stay_with_elder():
    for line in CLOSINGS + CLOSINGS_SILENT + [x for _, x in RISK.values()]:
        assert line.endswith("저는 여기 있을게요.")


# ---- 대화 흐름 (가짜 말하기·가짜 듣기·가짜 생성) ----

class Robot:
    def __init__(self, fail=()):
        self.said, self.fail = [], set(fail)

    def say(self, text):
        if text in self.fail:
            raise RuntimeError("edge-tts 실패")
        self.said.append(text)
        return -1.0          # 기다림 없이 바로 다음 (_say 는 길이+1초를 기다린다)


def run(robot, answers, kind="plant", gen=None, attend=None, with_attend=False, ask=None):
    """answers 를 차례로 대답. gen(msgs, last) -> 생성 문장 | None(실패)."""
    d = Dialog(robot.say, attend=attend, ask=ask)
    q = list(answers)
    d._listen = lambda mode: q.pop(0) if q else ""
    calls = []
    if gen is not None:
        def fake(model, msgs, fallback, max_sent=2):
            calls.append(msgs)
            out = gen(msgs, max_sent == 1)
            d.fell = out is None
            return fallback if out is None else out
        d.gen = fake
    d._run("mic", with_attend, kind)
    d.calls = calls
    return d


def chat(msgs, last):
    return "마음이 느껴져요." if last else "그러셨군요. 그때는 어떤 마음이 드세요?"


def test_four_answers_continue_until_thanks():
    r = Robot()
    d = run(r, ["물 주는 게 낙이지", "덜 적적해", "TV 보지", "딸 사진 봐", "보고 싶지. 이야기 들어줘서 고마워"], gen=chat)
    sc = SCENARIOS["plant"]
    assert r.said[0] in sc["questions"]
    assert r.said[1:5] == ["그러셨군요. 그때는 어떤 마음이 드세요?"] * 4       # 4번은 공감 + 질문
    assert r.said[5] == "마음이 느껴져요."                                      # 고마움 → 공감만
    assert r.said[6] in CLOSINGS and len(r.said) == 7
    assert d.log[-1].endswith("(대화 끝)")


def test_thanks_before_four_ends_early():
    r = Robot()
    run(r, ["좋지, 들어줘서 고마워"], gen=chat)
    assert len(r.said) == 3 and r.said[-1] in CLOSINGS


def test_max_turns():
    r = Robot()
    run(r, ["응"] * 20, gen=chat)
    assert len(r.said) == 1 + MAX_TURNS + 1                                    # 질문 + 8번 답 + 마무리
    assert r.said[MAX_TURNS] == "마음이 느껴져요." and r.said[-1] in CLOSINGS


def test_history_goes_to_clova():
    r = Robot()
    d = run(r, ["물 주는 게 낙이지", "고마워"], gen=chat)
    msgs = d.calls[1]
    assert msgs[0]["role"] == "system" and "살피는 마음" in msgs[1]["content"]
    roles = [m["role"] for m in msgs[2:]]
    assert roles == ["assistant", "user", "assistant", "user"]                 # 첫 질문부터 대화 전체
    assert "고마워" in msgs[-1]["content"] and "질문은 하지 마세요" in msgs[-1]["content"]


def test_silence_once_waits_twice_closes():
    r = Robot()
    run(r, ["", "물 주는 게 낙이지", "", ""], gen=chat)
    assert r.said[1] in NOT_HEARD and r.said[2].endswith("?")
    assert r.said[3] in NOT_HEARD and r.said[4] in CLOSINGS_SILENT


def test_risk_skips_clova_and_alerts():
    r = Robot()
    d = run(r, ["그냥 다 귀찮고 죽고 싶어"], gen=chat)
    assert r.said[0] in SCENARIOS["plant"]["questions"] and r.said[1:] == [RISK["life"][1]]
    assert d.calls == [] and "위험 신호" in d.alert and "죽고 싶어" in d.alert


def test_no_key_uses_fallback_and_rotates_follow_questions():
    r = Robot()
    run(r, ["좋아", "그렇지", "고마워"], kind="greet")                          # 실제 gen, 키 없음
    sc = SCENARIOS["greet"]
    assert r.said[1] == f"{sc['empathy']} {sc['follow'][0]}"
    assert r.said[2] == f"{sc['empathy']} {sc['follow'][1]}"
    assert r.said[3] == sc["empathy"] and r.said[4] in CLOSINGS


def test_yes_no_question_is_regenerated_once():
    outs = iter(["아드님이 그리우시겠어요. 아드님 보고 싶으신가요?", "아드님이 그리우시겠어요. 아드님 생각하시면 어떤 마음이 드세요?"])
    r = Robot()
    d = run(r, ["아들이 명절에나 와", "고마워"], gen=lambda m, last: "마음이 느껴져요." if last else next(outs))
    assert r.said[1] == "아드님이 그리우시겠어요. 아드님 생각하시면 어떤 마음이 드세요?"
    assert len(d.calls) == 3 and "열린 질문" in d.calls[1][-1]["content"]          # 생성 → 다시 만들기 → 마지막 공감


def test_regenerated_still_bad_falls_back_to_follow():
    r = Robot()
    run(r, ["좋아", "고마워"], gen=lambda m, last: "저도 좋아요." if last else "좋으시겠어요. 좋으신가요?")
    assert r.said[1] == f"좋으시겠어요. {SCENARIOS['plant']['follow'][0]}"


def test_generated_without_question_gets_follow():
    r = Robot()
    run(r, ["좋아", "고마워"], gen=lambda m, last: "참 좋으시겠어요." if not last else "고마워요.")
    assert r.said[1] == f"참 좋으시겠어요. {SCENARIOS['plant']['follow'][0]}"


def test_last_reply_drops_question():
    r = Robot()
    run(r, ["고마워"], gen=lambda m, last: "저도 좋았어요. 또 어떤 이야기 해 주실래요?")
    assert r.said[1] == "저도 좋았어요."


def test_ask_called_for_each_question():
    n = []
    r = Robot()
    run(r, ["좋아", "그래", "고마워"], gen=chat, ask=lambda: n.append(1))
    assert len(n) == 3                                                         # 첫 질문 + 질문 2번 (마지막 답·마무리는 질문 아님)


def test_tts_failure_on_generated_line_falls_back_and_finishes():
    r = Robot(fail={"그러셨군요. 그때는 어떤 마음이 드세요?"})
    d = run(r, ["좋아", "고마워"], gen=chat)
    assert r.said[1] == SCENARIOS["plant"]["empathy"] and r.said[-1] in CLOSINGS
    assert not any("(오류" in x for x in d.log)


def test_attend_failure_still_speaks():
    def attend(text):
        raise RuntimeError("합성 실패")
    r = Robot()
    run(r, ["고마워"], gen=chat, attend=attend, with_attend=True)
    assert r.said[0] in SCENARIOS["plant"]["questions"] and r.said[-1] in CLOSINGS


def test_greet_starts_with_hello_and_tells_scene():
    r = Robot()
    d = run(r, ["밥 먹었지", "고마워"], kind="greet", gen=chat)
    assert r.said[0] in SCENARIOS["greet"]["questions"] and r.said[-1] in CLOSINGS
    assert "손" in d.calls[0][1]["content"] and "인사" in d.calls[0][1]["content"]


def test_first_line_varies():
    for kind, sc in SCENARIOS.items():
        assert len(set(sc["questions"])) >= 5
        firsts = set()
        for _ in range(30):
            r = Robot()
            run(r, ["고마워"], kind=kind, gen=chat)
            firsts.add(r.said[0])
        assert len(firsts) >= 3 and firsts <= set(sc["questions"])      # 매번 같은 첫마디가 아니다


def test_after_act_waits_for_motion_before_first_line():
    st = {"busy": True}
    d = Dialog(lambda t: 0, busy=lambda: st["busy"])
    said_at = []
    d.say_fn = lambda t: said_at.append(time.time()) or 0
    t0 = time.time()
    threading.Timer(2.5, lambda: st.update(busy=False)).start()     # 손 흔들기 2.5초
    d.start("keyboard", kind="greet", after_act=True)
    while not said_at and time.time() - t0 < 6:
        time.sleep(0.05)
    d.stop()
    assert said_at and said_at[0] - t0 >= 2.4


def test_plant_first_lines_ask_different_things():
    qs = SCENARIOS["plant"]["questions"]
    asks = [q.split(". ", 1)[1] for q in qs]                          # 앞 "화분을 보고 계셨네요." 다음 질문
    assert len(set(asks)) == len(asks) and len(asks) >= 5


def test_only_plant_and_greet():
    assert set(SCENARIOS) == {"plant", "greet"}
    for k in ("bottle", "photo", "phone"):
        assert "모르는" in Dialog(lambda t: 0).start(kind=k)


def test_unknown_kind():
    assert "모르는" in Dialog(lambda t: 0).start(kind="calendar")


def test_warm_synthesizes_fixed_lines_and_tolerates_failure():
    made = []
    d = Dialog(lambda t: 0, synth=made.append)
    assert d.warm() == len(fixed_lines()) and set(made) == set(fixed_lines())
    assert all(q in made for sc in SCENARIOS.values() for q in sc["questions"])

    def down(t):
        raise RuntimeError("offline")
    assert Dialog(lambda t: 0, synth=down).warm() == 0
    assert Dialog(lambda t: 0).warm() == 0


# ---- 직접 재생(g1_play_pcm) 끝 잘림 방지 ----

@pytest.mark.parametrize("sec", [0.6, 3.4, 5.0, 7.9, 11.8])
def test_pad_for_player(sec):
    raw = b"\x01\x00" * int(16000 * sec)
    m = len(pad_for_player(raw)) / 32000
    stop = 0.85 * math.ceil(m) + 0.7          # g1_play_pcm 이 PlayStop 하는 시각
    assert m >= sec and sec + 0.3 <= stop <= sec + 1.6


# ---- 말 감지(VAD): 짧은 잡음은 버리고, 인식 실패는 다시 듣는다 ----

class FakeMic:
    """0.1초 조각을 차례로 내주는 arecord 흉내. 소리 = (초, 크기) 목록."""
    def __init__(self, parts):
        import array
        self.chunks = []
        for sec, amp in parts:
            for _ in range(round(sec * 10)):
                self.chunks.append(array.array("h", [amp, -amp] * 800).tobytes())
        self.stdout = self

    def read(self, n):
        return self.chunks.pop(0) if self.chunks else b""

    def kill(self):
        pass


def vad(parts, wait=3.0):
    d = Dialog(lambda t: 0)
    d._arecord = lambda: FakeMic(parts)
    return d._record_vad(wait)


def test_vad_ignores_short_click_then_catches_speech():
    pcm = vad([(0.5, 0), (0.2, 5000), (1.5, 0), (1.2, 5000), (1.5, 0)])
    assert 1.4 <= len(pcm) / 32000 <= 2.9                     # 딸깍(0.2 s)은 버리고 말(1.2 s)+앞 0.3 s+끝 침묵 1.2 s


def test_vad_short_click_only_is_silence():
    assert vad([(0.5, 0), (0.2, 5000), (4.0, 0)], wait=3.0) == b""


def test_listen_retries_after_failed_recognition():
    d = Dialog(lambda t: 0, answer_sec=5)
    clips = [b"\x01\x00" * 16000, b"\x02\x00" * 16000]
    texts = ["", "지은이가 보고 싶어"]
    d._record_vad = lambda wait=None: clips.pop(0) if clips else b""
    d.stt = lambda pcm: texts.pop(0)
    assert d._listen("mic") == "지은이가 보고 싶어"
    assert any("다시 듣기" in x for x in d.log)


def test_closings_vary_and_last_reply_does_not_copy_example():
    assert len(set(CLOSINGS)) >= 8 and len(set(NOT_HEARD)) >= 3
    said = set()
    for _ in range(30):
        r = Robot()
        run(r, ["고마워"], gen=chat)
        said.add(r.said[-1])
    assert len(said) >= 4                                                      # 매번 같은 마무리가 아니다


def test_first_questions_do_not_describe_what_robot_cannot_see():
    for sc in SCENARIOS.values():
        for q in sc["questions"]:
            assert "화분" in q or "안녕하세요" in q
            assert good_question(q) and not any(w in q for w in ("표정", "싱그럽", "드시는 거 보니", "가까이 있네요"))
        assert not any(f.startswith("그분") for f in sc["follow"])


def test_on_end_called_when_dialog_finishes():
    # 2026-10-31: "대화해줘서 고마워"로 끝나면 패널이 초기화(응시 대기)로 — on_end 로 알린다
    ended = []
    r = Robot()
    d = Dialog(r.say, on_end=lambda: ended.append(True))
    q = ["물 주는 게 낙이지", "이야기 들어줘서 고마워"]
    d._listen = lambda mode: q.pop(0) if q else ""
    d.gen = lambda model, msgs, fallback, max_sent=2: fallback
    d._run("mic", False, "plant")
    assert ended == [True] and not d.running and d.thanked      # "고마워"로 맺음 → 패널이 다음 시나리오로


def test_thanked_only_when_user_says_thanks():
    # 2026-10-31: 다음 시나리오로 넘어가는 기준은 "대화해줘서 고마워"뿐 — 대답 없음·최대 횟수 마무리는 아님
    r = Robot()
    d = Dialog(r.say)
    d._listen = lambda mode: ""                                   # 두 번 대답 없음 → 마무리
    d._run("mic", False, "plant")
    assert not d.thanked
