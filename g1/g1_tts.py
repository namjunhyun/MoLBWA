"""문장 → 지원이 스피커용 16 kHz·mono·int16 WAV (데스크탑). 대화 모듈(g1/dialog/tts_edge.py)과 같은 설정.

    python3 g1/g1_tts.py "안녕하세요, 저는 지원이에요."      → 경로 출력

기본 edge-tts(Microsoft ko-KR-InJoonNeural, 속도 −10%) — 무료.
TTS_ENGINE=clova 면 CLOVA Voice Premium(목소리 TTS_VOICE, 기본 nara) — 가장 자연스럽지만 앱에서 'CLOVA Voice - Premium'
을 체크하는 순간 **월 기본요금 90,000원**(호출 안 해도). 키는 ~/.config/molbwa/clova.env 의 NCP_CLIENT_ID/SECRET.
edge-tts 는 가상환경에 있다: python3 -m venv ~/.venvs/g1_dialog && ~/.venvs/g1_dialog/bin/pip install edge-tts
정규화: 최대값을 −1 dBFS 로(증폭은 최대 +12 dB). 같은 문장·목소리·속도는 캐시에서 바로 돌려준다.
"""
import hashlib
import os
import subprocess
import sys
import wave

import numpy as np

EDGE = os.path.expanduser(os.environ.get("EDGE_TTS", "~/.venvs/g1_dialog/bin/edge-tts"))
CACHE = os.path.expanduser("~/molbwa_g1/tts")
VOICE = "ko-KR-InJoonNeural"
RATE = "-10%"
SR = 16000


def normalize(x: np.ndarray, peak_dbfs=-1.0, max_gain_db=12.0) -> np.ndarray:
    pk = float(np.abs(x).max()) if len(x) else 0.0
    if pk < 1:
        return x
    g = min(32767 * 10 ** (peak_dbfs / 20) / pk, 10 ** (max_gain_db / 20))
    return np.clip(np.round(x * g), -32768, 32767).astype(np.int16)


CLOVA_URL = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"


def _clova_mp3(text, mp3):
    import urllib.parse
    import urllib.request
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from g1_dialog import load_env
    env = load_env()
    if not (env["NCP_CLIENT_ID"] and env["NCP_CLIENT_SECRET"]):
        raise RuntimeError("CLOVA Voice 키 없음(~/.config/molbwa/clova.env)")
    body = urllib.parse.urlencode({"speaker": os.environ.get("TTS_VOICE", "nara"), "text": text, "format": "mp3",
                                   "speed": 0, "volume": 0, "pitch": 0}).encode()
    req = urllib.request.Request(CLOVA_URL, body, {"X-NCP-APIGW-API-KEY-ID": env["NCP_CLIENT_ID"],
                                                   "X-NCP-APIGW-API-KEY": env["NCP_CLIENT_SECRET"],
                                                   "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=10) as r, open(mp3, "wb") as f:
        f.write(r.read())


def synth(text: str, voice: str = VOICE, rate: str = RATE) -> str:
    text = " ".join(text.split())
    if not text:
        raise ValueError("빈 문장")
    os.makedirs(CACHE, exist_ok=True)
    engine = os.environ.get("TTS_ENGINE", "edge")        # 패널 --tts 가 import 뒤에 정하므로 매번 읽는다
    tag = f"clova:{os.environ.get('TTS_VOICE', 'nara')}" if engine == "clova" else f"{voice}|{rate}"
    key = hashlib.sha1(f"{tag}|{text}".encode()).hexdigest()[:12]
    out = os.path.join(CACHE, f"tts_{key}.wav")
    if os.path.exists(out):
        return out
    mp3 = out[:-4] + ".mp3"
    if engine == "clova":
        _clova_mp3(text, mp3)
    elif not os.path.exists(EDGE):
        raise RuntimeError(f"edge-tts 없음({EDGE}) — python3 -m venv ~/.venvs/g1_dialog && ~/.venvs/g1_dialog/bin/pip install edge-tts")
    else:
        subprocess.run([EDGE, "--voice", voice, f"--rate={rate}", "--text", text, "--write-media", mp3],
                       check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", mp3, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         check=True, timeout=30, stdout=subprocess.PIPE).stdout   # raw 로 받아 직접 감싼다(파이프 wav 헤더는 길이가 깨짐)
    os.remove(mp3)
    pcm = normalize(np.frombuffer(raw, "<i2").astype(np.float64))
    tmp = out + ".part"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype("<i2").tobytes())
    os.replace(tmp, out)
    return out


def duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


if __name__ == "__main__":
    p = synth(" ".join(sys.argv[1:]) or "안녕하세요, 저는 지원이에요.")
    print(p, f"{duration(p):.2f}s")
