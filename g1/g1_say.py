"""데스크탑 → 지원이 스피커로 WAV 재생 (대화 모듈이 TTS 결과를 넘길 때 쓰는 입구).

    python3 g1/g1_say.py hello.wav --jetson 192.168.50.107        # 실기: 젯슨에 복사 후 재생 명령
    python3 g1/g1_say.py hello.wav                                 # 시뮬(127.0.0.1): 경로만 보내고 서버가 로그만 찍음

또는 코드에서:  from g1_say import say;  say("hello.wav", jetson="192.168.50.107")
젯슨의 g1_motion_server 가 돌고 있어야 한다(스피커는 서버 안에서 연다 — 정책 중 DDS 를 새로 열면 제어주기가 깨짐).
"""
import argparse
import os
import socket
import subprocess
import time

from g1_protocol import PORT_CMD, encode

REMOTE_DIR = "molbwa_g1/tts"


def say(wav: str, jetson: str = "127.0.0.1", user: str = "unitree") -> str:
    """WAV 를 젯슨에 올리고 재생 명령. 반환 = 젯슨 쪽 경로. 로컬(127.0.0.1)이면 복사 없이 절대경로."""
    wav = os.path.abspath(wav)
    if jetson in ("127.0.0.1", "localhost"):
        path = wav
    else:
        name = f"{int(time.time() * 1000)}_{os.path.basename(wav)}"
        dst = f"{user}@{jetson}"
        subprocess.run(["ssh", "-o", "BatchMode=yes", dst, f"mkdir -p ~/{REMOTE_DIR}"], check=True, timeout=10)
        subprocess.run(["scp", "-o", "BatchMode=yes", "-q", wav, f"{dst}:{REMOTE_DIR}/{name}"], check=True, timeout=20)
        path = f"/home/{user}/{REMOTE_DIR}/{name}"
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(encode({"cmd": "say", "path": path}), (jetson, PORT_CMD))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--jetson", default="127.0.0.1")
    a = ap.parse_args()
    print("보냄:", say(a.wav, a.jetson))
