"""지원이 스피커로 WAV 재생 (G1 음성 서비스 PlayStream). Jetson·Python 3.8.

파이썬 SDK 에는 PlayStream 이 없어서 C++ SDK(g1_audio_client.hpp)와 같은 요청을 직접 만든다:
API 1003(START_PLAY), 파라미터 JSON {"app_name", "stream_id"}, 바이너리 = PCM. 형식은 16 kHz·mono·int16
(G1 플레이어 문서의 16 kHz WAV 기준, 실기 미검증).

**정책 서버 안에서만 쓴다** — 정책이 도는 중에 다른 프로세스가 DDS 참여자를 새로 만들면 제어주기가 깨진다
(2026-10-12 실기 p99 62.9 ms). 서버는 이미 ChannelFactoryInitialize 를 했으므로 여기서는 클라이언트만 만든다.
"""
import json
import queue
import threading
import time
import wave

import numpy as np

API_START_PLAY = 1003
API_STOP_PLAY = 1004
RATE = 16000


def load_pcm16k(path: str) -> bytes:
    """WAV → 16 kHz mono int16 little-endian 바이트 (채널 평균, 선형 보간 리샘플)."""
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float64)
    elif sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float64) - 128) * 256
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float64) / 65536
    else:
        raise ValueError(f"샘플 폭 {sw} 바이트는 지원 안 함")
    x = x.reshape(-1, ch).mean(1)
    if sr != RATE and len(x) > 1:
        m = int(round(len(x) * RATE / sr))
        x = np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x)
    return np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()


class Speaker:
    """서버 프로세스 안의 재생기. say(path) 는 큐에 넣고 바로 돌아온다(제어 루프를 막지 않음)."""

    def __init__(self, app_name: str = "molbwa", volume: int = None):
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_
        self._Request = Request_
        self.c = AudioClient()
        self.c.SetTimeout(5.0)
        self.c.Init()
        self.app = app_name
        if volume is not None:
            self.c.SetVolume(int(volume))
        self.q = queue.Queue()
        self.last = ""
        threading.Thread(target=self._run, daemon=True).start()

    def _play_stream(self, pcm: bytes, stream_id: str) -> int:
        c = self.c
        header = c._ClientBase__SetHeader(API_START_PLAY, 0, 0, False)   # SDK 내부 헬퍼 (이름 맹글링)
        req = self._Request(header, json.dumps({"app_name": self.app, "stream_id": stream_id}), list(pcm))
        fut = c._ClientBase__stub.SendRequest(req, 5.0)
        if fut is None:
            return -1
        res = fut.GetResult(5.0)
        return res.value.header.status.code if res.code == 0 else -2

    def stop(self):
        self.c._Call(API_STOP_PLAY, json.dumps({"app_name": self.app}))

    def say(self, path: str):
        self.q.put(path)

    def _run(self):
        while True:
            path = self.q.get()
            try:
                pcm = load_pcm16k(path)
                code = self._play_stream(pcm, str(int(time.time() * 1000)))
                self.last = f"재생 {path} ({len(pcm) / 2 / RATE:.1f}s) 코드 {code}"
            except Exception as e:     # 재생 실패가 제어를 멈추면 안 된다
                self.last = f"재생 실패 {path}: {type(e).__name__} {e}"
            print("  [say]", self.last, flush=True)
