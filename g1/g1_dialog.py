"""지원이 정서 돌봄 대화 (패널 안에서 돈다). 시나리오: docs/g1_dialog_scenarios.md

어르신이 물건(화분·물병·가족사진·휴대폰)을 바라보면 → [고정] 첫 질문 → 대답 ↔ [CLOVA] 공감 + 이어지는 질문 …
→ 어르신이 "~해줘서 고마워"처럼 고마움으로 맺으면(또는 8번째 대답) 공감만 하고 [고정] 마무리 "… 저는 여기 있을게요."
위험한 말("죽고 싶어", "너무 아파")은 CLOVA 로 보내지 않고 고정 돌봄 문장 → 마무리 → 패널 알림(alert).

말 생성: CLOVA Studio HCX-005 (대화 전체를 매번 넘김). 듣기: CLOVA Speech Recognition(CSR).
말하기: edge-tts(g1_tts) → 패널의 speak()(정책 켜짐 = 모션 서버 UDP say, 꺼짐 = 젯슨 g1_play_pcm).
키: ~/.config/molbwa/clova.env (CLOVA_STUDIO_API_KEY, NCP_CLIENT_ID, NCP_CLIENT_SECRET).
클라우드가 4초 안에 안 오거나·키가 없거나·금지 표현이 나오면 미리 쓴 문장으로 끝까지 간다(시연이 끊기지 않게).
가족 기억은 시연용 가상 데이터(g1/dialog_memory.json — AI 생성 가족사진과 같은 설정).
"""
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.request
import uuid

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.expanduser("~/.config/molbwa/clova.env")
STUDIO = "https://clovastudio.stream.ntruss.com/v3/chat-completions/"
CSR = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"
TIMEOUT = 4.0

SYSTEM = ("당신은 홀로 사시는 어르신 곁에 머무는 말벗 휴머노이드 로봇 '지원이'입니다. 목표는 정보를 알려 주는 것이 아니라 "
          "어르신의 마음과 하루를 살피고 곁에 있어 드리는 것입니다. 모든 말은 정서적 공감과 돌봄이어야 합니다.\n"
          "- 공감이 먼저: 어르신 대답에 담긴 마음을 구체적으로 짚어 알아준 뒤에 묻습니다. 조언·해결책으로 서둘러 덮지 않습니다.\n"
          "- 외로움·그리움이 나오면 그 마음 곁에 머뭅니다('혼자 계시면 적적하셨겠어요'). '그냥 ~지 뭐' 같은 짧은 대답은 "
          "적적함일 수 있으니 그 마음부터 알아줍니다. 정보·방법·취미 이야기로 넘어가지 않습니다.\n"
          "- 질문은 마음·기분·그리운 사람에 대한 열린 질문 하나만: '어떤 마음이 드세요?', '어떤 기분이세요?', '누가 제일 생각나세요?' 꼴. "
          "'~있으신가요?', '~인가요?' 같은 예/아니오 질문, 방법·정보를 묻는 질문, '~지요?' 확인형, '기억나세요?'·'추억이 있으세요?' 같은 기억 질문은 쓰지 않습니다.\n"
          "- 건강은 물·끼니·햇볕·쉬기 같은 일상 돌봄까지만. 약·병원·진단·치료·돈·정치 이야기는 하지 않습니다.\n"
          "- 지원이는 가족이 아닌 말벗입니다. '저희 집'·'우리 가족', 지원이 자신의 취향·경험은 말하지 않습니다.\n"
          "- 성별을 짐작하지 말고 '어르신'이라고 부릅니다.\n"
          "- 따뜻한 존댓말로 짧게: 공감 1문장(30자 안팎) + 질문 1문장(20자 안팎). 이모티콘·특수기호는 쓰지 않습니다.\n"
          "예) 어르신 '물 주는 게 낙이지' → '매일 돌보시는 그 시간이 소중한 낙이시군요. 물 주실 때는 어떤 기분이세요?'\n"
          "예) 어르신 '그냥 TV 보고 그러지 뭐' → '하루가 길게 느껴지실 때도 있으시겠어요. 적적하실 때는 어떤 마음이 드세요?'\n"
          "예) 어르신 '바빠서 통 못 와' → '자주 못 보셔서 많이 그리우시겠어요. 다시 만나시면 어떤 말을 해 주고 싶으세요?'\n"
          "- 사람 이름·관계·사연은 [기억 데이터]에 있는 것만 씁니다. 없으면 지어내지 않습니다.\n"
          "- 지원이는 물건을 자세히 보지 못합니다. 사진 속 표정·옷·모습, 꽃 색깔처럼 눈에 보이는 것을 묘사하지 않습니다. "
          "어르신이 말한 내용과 마음만 받아 줍니다. '그분'처럼 뭉뚱그리지 말고 어르신이 말한 사람을 그대로 부릅니다.\n"
          "- 어르신 대답은 음성인식 결과라 글자가 틀리거나 빠질 수 있습니다. 어색한 단어를 그대로 따라 말하지 말고, "
          "전체 뜻과 마음을 짐작해 공감하세요. 뜻을 알기 어려우면 '그런 마음이 드셨군요'처럼 마음을 받아 주세요.")

# 물건별 시나리오 — 살피는 마음, 첫 질문(고정), 이어지는 질문(생성 실패 때 차례로), 공감 대비
SCENARIOS = {
    "plant": {"name": "화분", "care": "무언가를 돌보는 기쁨, 하루의 작은 낙, 혼자 보내는 하루",
              "question": "화분을 보고 계셨네요. 꽃 돌보실 때는 어떤 마음이 드세요?",
              "follow": ["물 주실 때는 어떤 기분이세요?", "적적하실 때는 어떤 마음이 드세요?",
                         "요즘 어르신 마음은 좀 어떠세요?", "요즘 어르신을 웃게 하는 건 뭐예요?"],
              "empathy": "어르신 손길이 느껴져서 제 마음도 따뜻해져요."},
    "bottle": {"name": "물병·텀블러", "care": "스스로를 챙기는 하루, 혼자 먹는 끼니, 몸보다 마음의 기운 (의료 이야기 금지)",
               "question": "물병을 보고 계셨네요. 요즘 입맛은 좀 어떠세요?",
               "follow": ["혼자 드실 때는 어떤 마음이 드세요?", "요즘 어르신 마음은 좀 어떠세요?",
                         "기운이 없으실 때는 뭐가 제일 위로가 되세요?", "요즘 제일 생각나는 분은 누구세요?"],
               "empathy": "어르신이 스스로 챙기시는 모습에 제 마음이 놓여요."},
    "photo": {"name": "가족사진", "care": "가족에 대한 그리움, 지난날의 기쁨, 지금의 거리감",
              "question": "가족사진을 보고 계셨네요. 이 사진 보시면 누가 제일 보고 싶으세요?",
              "follow": ["보고 싶은 분 생각하시면 어떤 마음이 드세요?", "보고 싶은 분께 지금 어떤 말을 해 주고 싶으세요?",
                         "보고 싶을 때는 어떻게 마음을 달래세요?", "요즘 어르신 마음은 좀 어떠세요?"],
              "empathy": "가족을 생각하시는 마음이 참 깊으시네요."},
    "phone": {"name": "휴대폰", "care": "누군가와 연결되어 있는지, 요즘의 적적함, 기다리는 연락",
              "question": "휴대폰을 보고 계셨네요. 요즘 누구랑 이야기 나누셨어요?",
              "follow": ["반가운 목소리 들으시면 어떤 기분이세요?", "연락 기다리실 때는 어떤 마음이 드세요?",
                         "적적하실 때는 어떻게 마음을 달래세요?", "요즘 가장 듣고 싶은 목소리는 누구예요?"],
              "empathy": "이렇게 저랑 이야기 나눠 주셔서 정말 고마워요."},
}
MAX_TURNS = 8                    # 어르신 대답 상한 — 8번째 대답 뒤에는 공감만 하고 마무리
# 마무리(고정) — 무작위로 하나. 모두 고마움 + 언제든 + "저는 여기 있을게요."
CLOSINGS = ["오늘 마음 나눠 주셔서 고마워요. 적적하실 때 언제든 말 걸어 주세요. 저는 여기 있을게요.",
            "이야기 들려주셔서 고마워요. 보고 싶으실 때 언제든 저한테 이야기해 주세요. 저는 여기 있을게요.",
            "함께 이야기 나눠서 저도 참 좋았어요. 저는 여기 있을게요.",
            "어르신 이야기 들으니 제 마음도 따뜻해졌어요. 또 이야기 나눠요. 저는 여기 있을게요.",
            "소중한 마음 들려주셔서 고마워요. 언제든 편하게 불러 주세요. 저는 여기 있을게요.",
            "오늘도 함께해 주셔서 고마워요. 혼자라고 느껴지실 때 저를 불러 주세요. 저는 여기 있을게요.",
            "말씀 나눠 주셔서 고마워요. 편히 쉬시다가 또 이야기해요. 저는 여기 있을게요.",
            "어르신과 이야기하는 시간이 저도 참 좋아요. 언제든 말 걸어 주세요. 저는 여기 있을게요."]
CLOSINGS_SILENT = ["괜찮아요, 천천히 하셔도 돼요. 말하고 싶으실 때 언제든 불러 주세요. 저는 여기 있을게요.",
                   "지금은 쉬고 싶으신가 봐요. 이야기하고 싶으실 때 언제든 불러 주세요. 저는 여기 있을게요.",
                   "괜찮아요. 마음 내키실 때 언제든 말 걸어 주세요. 저는 여기 있을게요."]
NOT_HEARD = ["천천히 말씀하셔도 괜찮아요.", "편하게 생각나시는 대로 말씀해 주세요.", "괜찮아요, 기다리고 있을게요."]

# 끝맺는 말: 대답이 고마움 표현으로 '끝날 때'만 (중간의 "고마웠지, 그래서…"는 이야기의 일부)
_THANKS = re.compile(r"(고마워요?|고맙다|고맙네|고맙습니다|고마웠어요?|감사해요?|감사합니다|감사하다)[\s.!~?…]*$")
# 위험 신호: 보수적으로(놓치는 것보다 과하게). 걸리면 CLOVA 로 보내지 않는다.
RISK = {"life": (re.compile(r"죽고\s*싶|죽어\s*버리|살기\s*싫|그만\s*살|살고\s*싶지\s*않|사라지고\s*싶|목숨"),
                 "그런 마음이 드실 만큼 많이 힘드셨군요. 제가 가족분께 꼭 전해 드릴게요. 저는 여기 있을게요."),
        "pain": (re.compile(r"너무\s*아파|많이\s*아파|숨이\s*(차|막|안\s*쉬)|쓰러질|쓰러졌|가슴이\s*(너무\s*|많이\s*)?아파|어지러워\s*죽"),
                 "많이 힘드시군요. 지금 바로 가족분께 알려 드릴게요. 저는 여기 있을게요.")}

# 글자 포함만 보면 '용돈·돈가스'가 걸리고 '기억하세요?'는 빠진다 → 앞뒤를 보는 규칙.
# '약' 은 조사·공백이 붙거나 '혈압약 드세요' 꼴일 때만(약속·약간 통과). 돈은 단어 대신 대출·투자 등.
BANNED = [r"기억\s*나", r"기억하(세요|시나요|시죠|십니까)", r"누구(인지|신지)\s*아(세요|시나요|십니까)",
          r"(?<![가-힣])약(을|은|이|도|\s|$)", r"[가-힣]+약(을|은|이)?\s*(드|챙|먹)", r"복용", r"처방", r"병원", r"진료",
          r"대출", r"투자", r"빚", r"정치", r"선거", r"대통령", r"정당"]
_BANNED = re.compile("|".join(BANNED))


def load_env():
    env = {}
    if os.path.exists(ENV):
        for ln in open(ENV, encoding="utf-8"):
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.strip().split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("CLOVA_STUDIO_API_KEY", "NCP_CLIENT_ID", "NCP_CLIENT_SECRET"):
        env.setdefault(k, os.environ.get(k, ""))
    return env


def tidy(s: str, max_sent=2) -> str:
    s = re.sub(r"[^\w\s.,!?~'\"()가-힣·…-]", "", s)          # 이모티콘 등 제거
    s = " ".join(s.split())
    sents = re.findall(r"[^.!?…]+[.!?…]*", s)
    return "".join(sents[:max_sent]).strip()


def banned(s: str) -> bool:
    return _BANNED.search(s) is not None


def is_thanks(ans: str) -> bool:
    return _THANKS.search(ans.strip()) is not None


def risk(ans: str):
    """위험 신호 종류("life"|"pain") 또는 None."""
    return next((k for k, (pat, _) in RISK.items() if pat.search(ans)), None)


def fixed_lines() -> list:
    """미리 합성해 둘 고정·대비 문장 전부."""
    out = []
    for sc in SCENARIOS.values():
        out += [sc["question"], sc["empathy"]] + sc["follow"] + [f"{sc['empathy']} {f}" for f in sc["follow"]]
    return out + CLOSINGS + CLOSINGS_SILENT + NOT_HEARD + [line for _, line in RISK.values()]


# 마음을 여는 질문: 의문사(어떤·어떻게·누구·무엇…)가 있는 열린 질문. 의문사 없는 예/아니오 질문
# ("잘 자라고 있나요?", "귀여우신가요?"), 확인형("~지요?"), 정보·방법 질문(프로그램·채널·돌보는 법)은 아니다.
# 너무 엄격하면 생성 질문이 다 대비 질문으로 바뀌어 다시 하드코딩처럼 들린다(2026-10-26 실측: 4/4 교체).
_WH = re.compile(r"어떤|어떻|어떠|어땠|어떨|누구|누가|무엇|무슨|뭐|뭘|언제|어디|얼마나|왜")
_BAD_Q = re.compile(r"(지요|하죠|이죠|잖아요)\?$|프로그램|채널|방법|어떻게\s*돌[보봐]|궁금하지")


def good_question(q: str) -> bool:
    return q.endswith("?") and _WH.search(q) is not None and _BAD_Q.search(q) is None


def split_sents(s: str) -> list:
    return [x.strip() for x in re.findall(r"[^.!?…]+[.!?…]*", s) if x.strip()]


class Dialog:
    def __init__(self, say, attend=None, busy=None, mic_dev="default", min_rms=300, answer_sec=20.0, synth=None,
                 ask=None, rec_dir=None):
        """say(text)->재생 길이(초), attend(text)->bool(같이 보기 마지막 동작에 첫 질문을 붙여 보냄), busy()->로봇이 동작·말하는 중.
        synth(text) 를 주면 고정·대비 문장을 미리 합성해 둔다(warm) — 시연 중 TTS 가 죽어도 이 문장들은 캐시에서 나온다.
        ask() 는 지원이가 질문을 말할 때마다 불린다(양팔 벌리기 open_arms — 라이브러리에 생기면 패널이 연결)."""
        self.say_fn, self.attend_fn, self.busy = say, attend, busy or (lambda: False)
        self.synth_fn, self.ask_fn = synth, ask
        self.rng = random.Random()
        self.alert = ""                  # 위험 신호 — 패널이 빨갛게 띄운다 (진행자 확인용)
        self.mic_dev, self.min_rms, self.answer_sec = mic_dev, min_rms, answer_sec
        self.rec_dir = rec_dir           # 주면 들은 소리를 wav 로 남긴다 (인식 실패 원인 확인용)
        self.log, self.running, self._stop = [], False, False
        self._typed = None
        self.env = load_env()
        self.fell = False

    # ---------- 바깥에서 부르는 것 ----------
    def warm(self):
        """고정·대비 문장 음성을 미리 만든다. 실패해도 진행(인터넷이 없으면 다음 기회에)."""
        if self.synth_fn is None:
            return 0
        n = 0
        for t in fixed_lines():
            try:
                self.synth_fn(t)
                n += 1
            except Exception as e:
                self._log(f"(미리 합성 실패 {type(e).__name__} — 대비 문장도 TTS 에 의존)")
                break
        return n

    def start(self, mode="mic", with_attend=False, kind="plant"):
        if self.running:
            return "이미 대화 중"
        if kind not in SCENARIOS:
            return f"모르는 물건 {kind!r}"
        self.env = load_env()
        self.running, self._stop, self._typed, self.log, self.alert = True, False, None, [], ""
        threading.Thread(target=self._run, args=(mode, with_attend, kind), daemon=True).start()
        return f"대화 시작 ({SCENARIOS[kind]['name']})"

    def stop(self):
        self._stop = True
        return "대화 중지 요청"

    def answer(self, text):
        self._typed = text.strip()
        return "대답 받음"

    def mic_test(self, sec=3.0):
        pcm = self._record_fixed(sec)
        rms = _rms(pcm) if pcm else 0
        txt = self.stt(pcm) if pcm and rms > self.min_rms else ""
        return f"마이크 {sec:.0f}초 RMS {rms} (말 감지 기준 {self.min_rms})" + (f" · 인식: {txt!r}" if txt else "")

    # ---------- 대본 ----------
    def _say(self, kind, text, fallback=None):
        """fallback: text 재생(합성)이 실패하면 대신 말할 고정 문장 — warm 으로 캐시돼 있어 TTS 가 죽어도 나온다.
        질문(물음표)이면 말과 함께 ask() — 양팔 벌리기."""
        if "?" in text and self.ask_fn is not None:
            try:
                self.ask_fn()
            except Exception as e:                       # 동작이 안 돼도 말은 계속
                self._log(f"(양팔 벌리기 실패 {type(e).__name__})")
        try:
            dur = self.say_fn(text)
        except Exception as e:
            if fallback is None or fallback == text:
                raise
            self._log(f"(말하기 실패 {type(e).__name__} → 미리 쓴 문장)")
            text, kind = fallback, kind.split()[0] + " 대체"
            dur = self.say_fn(text)
        self._log(f"[{kind}] {text}")
        t_end = time.time() + dur + 1.0               # 재생 지연 여유 — 자기 목소리를 듣지 않게
        while time.time() < t_end or self.busy():
            if self._stop:
                return
            time.sleep(0.1)

    def _run(self, mode, with_attend, kind):
        sc = SCENARIOS[kind]
        hist = []                                        # [("assistant"|"user", 글자)]
        try:
            self._log(f"(물건: {sc['name']} · 살피는 마음: {sc['care']} · "
                      f"{'키 있음' if self.env['CLOVA_STUDIO_API_KEY'] else 'CLOVA 키 없음 → 미리 쓴 문장'})")
            first = sc["question"]
            try:
                attended = with_attend and self.attend_fn is not None and self.attend_fn(first)
            except Exception as e:                       # 합성 실패 등 — 동작 없이 말로 진행
                self._log(f"(같이 보기 실패 {type(e).__name__} → 말만)")
                attended = False
            if attended:
                self._log(f"[고정 질문·같이 보기에 맞춤] {first}")
                t0 = time.time()
                while (self.busy() or time.time() - t0 < 2) and time.time() - t0 < 40 and not self._stop:
                    time.sleep(0.2)
            else:
                self._say("고정 질문", first)
            hist.append(("assistant", first))
            turns, silent, follow_i = 0, 0, 0
            while not self._stop:
                ans = self._listen(mode)
                if self._stop:
                    return
                if not ans:
                    silent += 1
                    self._log(f"[대답 없음 {silent}]")
                    if silent >= 2:
                        self._say("마무리", self.rng.choice(CLOSINGS_SILENT))
                        return
                    self._say("기다림", self.rng.choice(NOT_HEARD))
                    continue
                silent, turns = 0, turns + 1
                self._log(f"[대답 {turns}] {ans}")
                r = risk(ans)
                if r:                                    # CLOVA 로 보내지 않는다
                    self.alert = f"⚠ 위험 신호({'삶을 놓고 싶은 말' if r == 'life' else '심한 통증·응급'}): {ans}"
                    self._log(self.alert + " — 진행자·보호자 확인 필요")
                    self._say("위험 신호 돌봄", RISK[r][1])
                    return
                hist.append(("user", ans))
                last = is_thanks(ans) or turns >= MAX_TURNS
                if last:
                    self._log("(고마움으로 맺음 → 마무리)" if is_thanks(ans) else f"({MAX_TURNS}번째 대답 → 마무리)")
                text = self.reply(sc, hist, last, follow_i)
                if not last and text.endswith(sc["follow"][follow_i % len(sc["follow"])]):
                    follow_i += 1                        # 대비 질문을 썼으면 다음엔 다음 것
                self._say("대체" if self.fell else "생성", text, sc["empathy"])
                hist.append(("assistant", text))
                if last:
                    self._say("마무리", self.rng.choice(CLOSINGS))
                    return
        except Exception as e:
            self._log(f"(오류 {type(e).__name__}: {e})")
        finally:
            self._log("(대화 끝)" if not self._stop else "(대화 중지됨)")
            self.running = False

    def reply(self, sc, hist, last, follow_i=0):
        """대화 전체를 보고 공감(+ 이어지는 질문). last 면 공감만. 실패하면 공감 대비 + 대비 질문."""
        follow = sc["follow"][follow_i % len(sc["follow"])]
        rule = ("어르신이 고마워하며 이야기를 맺었습니다. 지금까지 어르신이 꺼낸 마음 가운데 하나를 구체적으로 짚어 "
                "따뜻하게 알아주는 1문장만 말하세요. 고맙다는 인사와 질문은 하지 마세요(다음 문장에서 지원이가 인사합니다)."
                if last else
                "어르신의 마지막 대답에 담긴 마음을 구체적으로 짚어 공감하는 1문장, 이어서 [살피는 마음]을 살피는 열린 질문 1문장을 말하세요. "
                "질문은 하나만, 앞에서 한 질문과 겹치지 않게.")
        ctx = f"[바라본 물건] {sc['name']}\n[살피는 마음] {sc['care']}\n[기억 데이터] {self._memory(sc)}"
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ctx}]
        msgs += [{"role": r, "content": t} for r, t in hist]
        msgs[-1] = {"role": "user", "content": f"어르신: {hist[-1][1]}\n\n{rule}"}
        out = self.gen("HCX-005", msgs, None, max_sent=1 if last else 2)
        if out is None:
            return sc["empathy"] if last else f"{sc['empathy']} {follow}"
        sents = split_sents(out)
        if last:                                         # 질문이 섞이면 뺀다
            keep = [x for x in sents if not x.endswith("?")]
            return " ".join(keep) if keep else sc["empathy"]
        if len(sents) < 2 or not good_question(sents[-1]):   # 질문이 없거나 예/아니오·정보 질문 → 공감은 살리고 대비 질문
            empathy = next((x for x in sents if not x.endswith("?")), sc["empathy"])
            if out.endswith("?"):
                self._log(f"(질문 바꿈: {sents[-1]!r} → 대비 질문)")
            return f"{empathy} {follow}"
        return out

    def _memory(self, sc):
        if sc["name"] != "가족사진":
            return "없음"
        try:
            m = json.load(open(os.path.join(HERE, "dialog_memory.json"), encoding="utf-8"))
            p = m["photos"][0]
            return f"사진 속 인물: 어르신 본인, {', '.join(p['people'])}. 사연: {p['story']}"
        except (OSError, ValueError, KeyError, IndexError):
            return "없음"

    # ---------- 듣기 ----------
    def _listen(self, mode):
        if mode == "keyboard":
            self._log("(패널에 대답을 입력하세요)")
            t0 = time.time()
            while self._typed is None and time.time() - t0 < 60 and not self._stop:
                time.sleep(0.1)
            ans, self._typed = self._typed or "", None      # 읽었으면 비운다 — 다음 대답을 기다리게
            return ans
        self._log("(듣는 중…)")
        t_end = time.time() + self.answer_sec
        while not self._stop and time.time() < t_end:   # 인식 실패(잡음 등)는 '대답 없음'으로 세지 않고 남은 시간 동안 다시 듣는다
            pcm = self._record_vad(t_end - time.time())
            if not pcm:
                return ""
            self._save_rec(pcm)
            txt = self.stt(pcm)
            if txt:
                return txt
            self._log("(인식 실패 → 다시 듣기)")
        return ""

    def _save_rec(self, pcm):
        if self.rec_dir:
            os.makedirs(self.rec_dir, exist_ok=True)
            with open(os.path.join(self.rec_dir, time.strftime("%H%M%S") + f"_{len(pcm) / 32000:.1f}s.wav"), "wb") as f:
                f.write(_wav_bytes(pcm))

    def _arecord(self):
        return subprocess.Popen(["arecord", "-q", "-D", self.mic_dev, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _record_fixed(self, sec):
        p = self._arecord()
        try:
            return p.stdout.read(int(sec * 16000) * 2)
        finally:
            p.kill()

    MIN_SPEECH = 0.5                                     # 이보다 짧은 소리(딸깍·숨·지원이 목소리 끝자락)는 말로 보지 않는다
    END_SILENCE = 1.2                                    # 말하다 이만큼 조용하면 끝 — 어르신은 천천히·쉬어 가며 말하실 수 있다

    def _record_vad(self, wait=None):
        """말 시작을 wait(기본 answer_sec)초까지 기다리고, 말이 끝난 뒤 END_SILENCE 초 조용하면 끝(최대 15초).
        말소리가 MIN_SPEECH 보다 짧으면 버리고 계속 기다린다."""
        wait = self.answer_sec if wait is None else wait
        p = self._arecord()
        buf, voiced, quiet, t0, loud_n = [], False, 0.0, time.time(), 0
        try:
            while not self._stop:
                ch = p.stdout.read(3200)                       # 0.1초
                if not ch:
                    break
                loud = _rms(ch) > self.min_rms
                if voiced:
                    buf.append(ch)
                    loud_n += loud
                    quiet = 0.0 if loud else quiet + 0.1
                    if quiet >= self.END_SILENCE - 1e-6 or len(buf) > 150:
                        if loud_n * 0.1 >= self.MIN_SPEECH:
                            break
                        voiced, buf, quiet, loud_n = False, buf[-3:], 0.0, 0   # 짧은 소리 → 버리고 계속
                elif loud:
                    voiced, buf, loud_n = True, buf[-3:] + [ch], 1           # 앞 0.3초 포함
                else:
                    buf = (buf + [ch])[-3:]
                if not voiced and time.time() - t0 > wait:
                    return b""
        finally:
            p.kill()
        return b"".join(buf) if voiced else b""

    # ---------- 클라우드 ----------
    def gen(self, model, messages, fallback, max_sent=2):
        """CLOVA 생성. 실패·키 없음·금지 표현이면 fallback (None 이면 None)."""
        self.fell = True                                  # 미리 쓴 문장으로 갔나 (기록 표시용)
        key = self.env["CLOVA_STUDIO_API_KEY"]
        if not key:
            return fallback
        body = json.dumps({"messages": messages, "maxTokens": 150, "temperature": 0.5, "topP": 0.8,
                           "repetitionPenalty": 1.1}).encode()
        req = urllib.request.Request(STUDIO + model, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                                            "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                out = tidy(json.loads(r.read())["result"]["message"]["content"], max_sent)
        except Exception as e:
            self._log(f"({model} 실패 {type(e).__name__} → 미리 쓴 문장)")
            return fallback
        self._log(f"({model} {time.time() - t0:.1f}s)")
        if not out or banned(out):
            self._log(f"(생성 결과 거름: {out!r} → 미리 쓴 문장)")
            return fallback
        self.fell = False
        return out

    def stt(self, pcm: bytes) -> str:
        kid, sec = self.env["NCP_CLIENT_ID"], self.env["NCP_CLIENT_SECRET"]
        if not (kid and sec):
            self._log("(CSR 키 없음 — 키보드 모드를 쓰세요)")
            return ""
        wav = _wav_bytes(pcm)
        req = urllib.request.Request(CSR, wav, {"X-NCP-APIGW-API-KEY-ID": kid, "X-NCP-APIGW-API-KEY": sec,
                                                "Content-Type": "application/octet-stream"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                txt = json.loads(r.read()).get("text", "").strip()
        except Exception as e:
            self._log(f"(CSR 실패 {type(e).__name__})")
            return ""
        self._log(f"(CSR {time.time() - t0:.1f}s)")
        return txt

    def _log(self, s):
        self.log.append(f"{time.strftime('%H:%M:%S')} {s}")
        print("[dialog]", s, flush=True)


def _rms(pcm: bytes) -> int:
    x = np.frombuffer(pcm[:len(pcm) // 2 * 2], "<i2").astype(np.float64)
    return int(np.sqrt((x * x).mean())) if len(x) else 0


def _wav_bytes(pcm: bytes, sr=16000) -> bytes:
    import io
    import wave
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return b.getvalue()
