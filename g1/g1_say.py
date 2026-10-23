"""데스크탑 → 지원이 스피커로 WAV 재생 (대화 모듈·패널이 TTS 결과를 넘길 때 쓰는 입구).

    python3 g1/g1_say.py hello.wav --jetson 192.168.50.107            # 정책 중: 모션 서버(UDP say)가 재생
    python3 g1/g1_say.py hello.wav --jetson 192.168.50.107 --direct   # 정책 없음: 젯슨 ~/g1_play_pcm 으로 바로
    python3 g1/g1_say.py hello.wav                                     # 시뮬(127.0.0.1): 경로만 보내고 서버가 로그만 찍음

코드에서:  from g1_say import say;  say("hello.wav", jetson="192.168.50.107")

두 길을 섞으면 안 된다:
  · 정책(g1_motion_server)이 도는 중에는 **반드시 UDP say** — 스피커는 서버 안에서 연다. 다른 프로세스가 젯슨에서
    DDS 를 새로 열면 제어주기가 깨져 정책이 스스로 멈춘다(2026-10-12 실기 p99 62.9 ms).
  · --direct(g1_play_pcm, 새 DDS 참여자)는 서버가 없을 때만. 젯슨에서 재생 직전에 서버가 떠 있으면 거부한다.
"""
import argparse
import os
import socket
import subprocess
import time
import wave

from g1_protocol import PORT_CMD, encode

REMOTE_DIR = "molbwa_g1/tts"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "-o", "ControlMaster=auto",
       "-o", "ControlPath=/tmp/g1_say_%r@%h", "-o", "ControlPersist=600"]
PLAY_PCM = "~/g1_play_pcm/build/g1_play_pcm"


def _local(jetson):
    return jetson in ("127.0.0.1", "localhost")


def upload(wav: str, jetson: str, user: str = "unitree") -> str:
    """WAV 를 젯슨에 올리고 젯슨 쪽 경로를 돌려준다(재생은 안 함). 로컬이면 절대경로 그대로."""
    wav = os.path.abspath(wav)
    if _local(jetson):
        return wav
    name = f"{int(time.time() * 1000)}_{os.path.basename(wav)}"
    dst = f"{user}@{jetson}"
    subprocess.run(SSH + [dst, f"mkdir -p ~/{REMOTE_DIR}"], check=True, timeout=10)
    subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ControlPath=/tmp/g1_say_%r@%h", "-q", wav,
                    f"{dst}:{REMOTE_DIR}/{name}"], check=True, timeout=20)
    return f"/home/{user}/{REMOTE_DIR}/{name}"


def say(wav: str, jetson: str = "127.0.0.1", user: str = "unitree", play: bool = True) -> str:
    """정책 중 경로: WAV 를 올리고 모션 서버에 UDP {cmd: say}. play=False 면 올리기만(동작에 붙일 때)."""
    path = upload(wav, jetson, user)
    if play:
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(encode({"cmd": "say", "path": path}), (jetson, PORT_CMD))
    return path


def say_direct(wav: str, jetson: str, user: str = "unitree", volume: int = 100, iface: str = "eth1") -> str:
    """정책 없을 때 경로: raw PCM 으로 올려 젯슨 g1_play_pcm(실기 확인된 C++ 재생기)으로 튼다. 끝날 때까지 블록."""
    if _local(jetson):
        return "시뮬: 직접 재생 없음"
    with wave.open(wav, "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 16000):
            raise ValueError("16 kHz mono int16 WAV 만 — g1_tts.synth 로 만든 파일을 쓰세요")
        raw = w.readframes(w.getnframes())
    dst = f"{user}@{jetson}"
    pcm = f"/tmp/molbwa_say_{int(time.time() * 1000)}.pcm"
    guard = (f"if pgrep -f \"[g]1_motion_server.py\" >/dev/null; then echo '거부: 모션 서버(정책)가 돌고 있음 — UDP say 를 쓰세요'; "
             f"rm -f {pcm}; exit 3; fi; {PLAY_PCM} {iface} {pcm} {int(volume)} | tail -1; rm -f {pcm}")
    r = subprocess.run(SSH + [dst, f"cat > {pcm} && {{ {guard}; }}"], input=raw, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=60)
    out = r.stdout.decode(errors="replace").strip()
    if r.returncode != 0:
        raise RuntimeError(out or f"직접 재생 실패(코드 {r.returncode})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--jetson", default="127.0.0.1")
    ap.add_argument("--direct", action="store_true", help="정책 없을 때만: 젯슨 g1_play_pcm 으로 바로 재생")
    ap.add_argument("--volume", type=int, default=100)
    a = ap.parse_args()
    print("보냄:", say_direct(a.wav, a.jetson, volume=a.volume) if a.direct else say(a.wav, a.jetson))
