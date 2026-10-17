"""지원이 2턴 대화 (패널 안에서 돈다). 첫 마디(생성) → 고정 질문 → 대답 듣기 → 공감(생성) → 마무리(고정).

말 생성: CLOVA Studio HCX-005(첫 마디·공감 모두 — 한국어 자연스러움 우선, DASH 대비 +0.2 s). 듣기: CLOVA Speech Recognition(CSR).
말하기: edge-tts(g1_tts) → 패널의 speak()(정책 켜짐 = 모션 서버 UDP say, 꺼짐 = 젯슨 g1_play_pcm).
키: ~/.config/molbwa/clova.env (CLOVA_STUDIO_API_KEY, NCP_CLIENT_ID, NCP_CLIENT_SECRET) — 대화 팀과 같은 이름.
클라우드가 4초 안에 안 오거나·키가 없거나·금지 표현이 나오면 미리 쓴 문장으로 끝까지 간다(시연이 끊기지 않게).
가족 기억은 시연용 가상 데이터(g1/dialog_memory.json).
"""
import datetime as dt
import json
import os
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

SYSTEM = ("당신은 홀로 사시는 어르신 곁에서 이야기를 나누는 휴머노이드 로봇 '지원이'입니다. "
          "따뜻한 존댓말로, 50자 안팎·많아야 2문장으로 말하세요. 이모티콘·특수기호는 쓰지 마세요. "
          "약·병원·돈·정치 이야기와 기억력을 시험하는 질문(기억나세요? 등)은 하지 마세요. "
          "주어진 날짜와 남은 일수를 그대로 쓰고 다시 계산하지 마세요.")
QUESTION = "그날 뭘 해 주고 싶으세요?"
CLOSING = "오늘 이야기 나눠서 즐거웠어요."
FALLBACK = {"opening": "어르신, 곧 손녀 지은이 생일이에요. 벌써 기다려지시겠어요.",
            "empathy": "정말 좋은 생각이에요. 지은이가 많이 기뻐하겠어요.",
            "silence": "천천히 생각해 보셔도 괜찮아요."}
BANNED = ("기억나", "기억 나", "약 ", "약을", "약은", "병원", "돈", "정치", "선거")


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
    return any(b in s + " " for b in BANNED)


def upcoming(memory, today):
    """가장 가까운 기념일 (설명, 날짜, 남은 일수)."""
    best = None
    for ev in memory["events"]:
        m, d = ev["month"], ev["day"]
        day = dt.date(today.year, m, d)
        if day < today:
            day = dt.date(today.year + 1, m, d)
        n = (day - today).days
        if best is None or n < best[2]:
            best = (ev["what"], day, n)
    return best


class Dialog:
    def __init__(self, say, attend=None, busy=None, mic_dev="default", min_rms=300, answer_sec=10.0):
        """say(text)->재생 길이(초), attend(text)->bool(같이 보고 축하에 첫 마디를 붙여 보냄), busy()->로봇이 동작·말하는 중."""
        self.say_fn, self.attend_fn, self.busy = say, attend, busy or (lambda: False)
        self.mic_dev, self.min_rms, self.answer_sec = mic_dev, min_rms, answer_sec
        self.log, self.running, self._stop = [], False, False
        self._typed = None
        self.env = load_env()
        self.fell = False

    # ---------- 바깥에서 부르는 것 ----------
    def start(self, mode="mic", with_attend=False):
        if self.running:
            return "이미 대화 중"
        self.env = load_env()
        self.running, self._stop, self._typed, self.log = True, False, None, []
        threading.Thread(target=self._run, args=(mode, with_attend), daemon=True).start()
        return "대화 시작"

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
    def _say(self, kind, text):
        self._log(f"[{kind}] {text}")
        dur = self.say_fn(text)
        t_end = time.time() + dur + 1.0               # 재생 지연 여유 — 자기 목소리를 듣지 않게
        while time.time() < t_end or self.busy():
            if self._stop:
                return
            time.sleep(0.1)

    def _run(self, mode, with_attend):
        try:
            memory = json.load(open(os.path.join(HERE, "dialog_memory.json"), encoding="utf-8"))
            today = dt.date.today()
            what, day, n = upcoming(memory, today)
            self._log(f"(오늘 {today:%m월 %d일}, 다가오는 일: {what} {day.month}월 {day.day}일 · {n}일 뒤 · "
                      f"{'키 있음' if self.env['CLOVA_STUDIO_API_KEY'] else 'CLOVA 키 없음 → 미리 쓴 문장'})")
            opening = self.gen("HCX-005", f"오늘은 {today.year}년 {today.month}월 {today.day}일 "
                               f"{'월화수목금토일'[today.weekday()]}요일입니다. 다가오는 일정: {day.month}월 {day.day}일"
                               f"({n}일 뒤) {what}. 어르신께 이 일정을 반갑게 알려 주는 첫 마디를 하세요. 날짜({day.month}월 {day.day}일)와 "
                               f"남은 날({n}일)을 꼭 넣으세요. 지원이는 가족이 아니므로 '저희 집'·'우리 가족' 같은 말은 쓰지 마세요. "
                               f"예) '어르신, {day.month}월 {day.day}일이 {what}이에요. 이제 {n}일 남았네요.'", "opening")
            if with_attend and self.attend_fn is not None and self.attend_fn(opening):
                self._log(f"[1 {'대체' if self.fell else '생성'}·같이 보기 박수에 맞춤] {opening}")
                t0 = time.time()
                while (self.busy() or time.time() - t0 < 2) and time.time() - t0 < 40 and not self._stop:
                    time.sleep(0.2)
            else:
                self._say("1 대체" if self.fell else "1 생성", opening)
            if self._stop:
                return
            self._say("2 고정", QUESTION)
            if self._stop:
                return
            ans = self._listen(mode)
            if self._stop:
                return
            if ans:
                self._log(f"[3 대답] {ans}")
                emp = self.gen("HCX-005", f"상황: {what}({day.month}월 {day.day}일)을 앞두고 지원이가 '{QUESTION}' 하고 물었습니다. "
                               f"어르신 대답: '{ans}'. 어르신의 마음(손주·가족을 아끼는 마음)을 알아주는 공감을 한 문장으로 하세요. "
                               "로봇인 지원이 자신의 취향·경험은 말하지 말고, 대답 내용을 구체적으로 짚어 주세요. "
                               "대답이 '모르겠다'처럼 아직 정하지 못한 것이면 천천히 생각해도 괜찮다고 다정하게 말하세요. 할머니·할아버지처럼 성별을 짐작하지 말고 '어르신'이라고 부르세요. 예) 대답 '용돈 줘야지' → '어르신 마음이 담긴 용돈이라 지은이가 정말 기뻐하겠어요.'",
                               "empathy", max_sent=1)
                self._say("4 대체" if self.fell else "4 생성", emp)
            else:
                self._log("[3 대답] (없음)")
                self._say("4 고정", FALLBACK["silence"])
            if not self._stop:
                self._say("5 고정", CLOSING)
        except Exception as e:
            self._log(f"(오류 {type(e).__name__}: {e})")
        finally:
            self._log("(대화 끝)" if not self._stop else "(대화 중지됨)")
            self.running = False

    # ---------- 듣기 ----------
    def _listen(self, mode):
        if mode == "keyboard":
            self._log("(패널에 대답을 입력하세요)")
            t0 = time.time()
            while self._typed is None and time.time() - t0 < 60 and not self._stop:
                time.sleep(0.1)
            return self._typed or ""
        self._log("(듣는 중…)")
        pcm = self._record_vad()
        if not pcm:
            return ""
        txt = self.stt(pcm)
        if not txt:
            self._log("(인식 실패)")
        return txt

    def _arecord(self):
        return subprocess.Popen(["arecord", "-q", "-D", self.mic_dev, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _record_fixed(self, sec):
        p = self._arecord()
        try:
            return p.stdout.read(int(sec * 16000) * 2)
        finally:
            p.kill()

    def _record_vad(self):
        """말 시작을 answer_sec 까지 기다리고, 말이 끝난 뒤 0.8초 조용하면 끝(최대 15초)."""
        p = self._arecord()
        buf, voiced, quiet, t0 = [], False, 0.0, time.time()
        try:
            while not self._stop:
                ch = p.stdout.read(3200)                       # 0.1초
                if not ch:
                    break
                loud = _rms(ch) > self.min_rms
                if voiced:
                    buf.append(ch)
                    quiet = 0.0 if loud else quiet + 0.1
                    if quiet >= 0.8 or len(buf) > 150:
                        break
                elif loud:
                    voiced, buf = True, buf[-3:] + [ch]            # 앞 0.3초 포함
                else:
                    buf = (buf + [ch])[-3:]
                    if time.time() - t0 > self.answer_sec:
                        return b""
        finally:
            p.kill()
        return b"".join(buf) if voiced else b""

    # ---------- 클라우드 ----------
    def gen(self, model, prompt, kind, max_sent=2):
        self.fell = True                                  # 미리 쓴 문장으로 갔나 (기록 표시용)
        key = self.env["CLOVA_STUDIO_API_KEY"]
        if not key:
            return FALLBACK[kind]
        body = json.dumps({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                           "maxTokens": 150, "temperature": 0.5, "topP": 0.8, "repetitionPenalty": 1.1}).encode()
        req = urllib.request.Request(STUDIO + model, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                                            "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                out = tidy(json.loads(r.read())["result"]["message"]["content"], max_sent)
        except Exception as e:
            self._log(f"({model} 실패 {type(e).__name__} → 미리 쓴 문장)")
            return FALLBACK[kind]
        self._log(f"({model} {time.time() - t0:.1f}s)")
        if not out or banned(out):
            self._log(f"(생성 결과 거름: {out!r} → 미리 쓴 문장)")
            return FALLBACK[kind]
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
