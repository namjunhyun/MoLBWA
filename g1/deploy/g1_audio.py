"""지원이 스피커로 WAV 재생 (G1 음성 서비스 PlayStream). Jetson·Python 3.8.

파이썬 SDK 에는 PlayStream 이 없어서 C++ SDK(g1_audio_client.hpp)와 같은 요청을 직접 만든다:
API 1003(START_PLAY), 파라미터 JSON {"app_name", "stream_id"}, 바이너리 = PCM. 형식은 16 kHz·mono·int16.
보내는 법은 실기 확인된 젯슨 ~/g1_play_pcm(C++, 2026-10-15 대화 팀 확인)과 같다: 같은 stream_id 로 1초(32000 B)
조각을 0.85초 간격으로 보내고, 다 들린 뒤 PlayStop. 정지를 마지막 조각 + 0.7초에 하면 약 4.7초보다 긴 문장은
끝이 잘리므로(보내기가 재생보다 빠르다) 정지 시각을 "첫 조각 + 길이 + 0.3초" 이후로 둔다.

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
CHUNK = 32000           # 1초 = 16000 샘플 × 2 B


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
        self.until = 0.0                 # 지금 재생 중인 소리가 끝나는 시각(추정: 보낸 시각 + 길이)
        threading.Thread(target=self._run, daemon=True).start()

    def _play_stream(self, pcm: bytes, stream_id: str) -> int:
        c = self.c
        if hasattr(c, "PlayStream"):     # 젯슨 SDK(2026-10 확인)에는 있다 — 반환 (코드, 데이터)
            r = c.PlayStream(self.app, stream_id, pcm)
            return r[0] if isinstance(r, tuple) else int(r)
        header = c._ClientBase__SetHeader(API_START_PLAY, 0, 0, False)   # SDK 내부 헬퍼 (이름 맹글링)
        req = self._Request(header, json.dumps({"app_name": self.app, "stream_id": stream_id}), list(pcm))
        fut = c._ClientBase__stub.SendRequest(req, 5.0)
        if fut is None:
            return -1
        res = fut.GetResult(5.0)
        return res.value.header.status.code if res.code == 0 else -2

    def stop(self):
        if hasattr(self.c, "PlayStop"):
            self.c.PlayStop(self.app)
        else:
            self.c._Call(API_STOP_PLAY, json.dumps({"app_name": self.app}))

    def say(self, path: str):
        self.q.put(path)

    def speaking(self) -> bool:
        return not self.q.empty() or time.time() < self.until

    def play(self, pcm: bytes) -> int:
        """1초 조각 스트리밍(이 스레드에서 블록). 반환 = 첫 실패 코드 또는 0."""
        sid = f"molbwa_{int(time.time() * 1000)}"
        dur = len(pcm) / 2 / RATE
        t0 = time.time()
        self.until = t0 + dur
        code = 0
        for o in range(0, len(pcm), CHUNK):
            code = self._play_stream(pcm[o:o + CHUNK], sid)
            if code != 0:
                break
            time.sleep(0.85)
        time.sleep(max(0.7, t0 + dur + 0.3 - time.time()))
        try:
            self.stop()
        except Exception:
            pass
        self.until = 0.0
        return code

    def _run(self):
        while True:
            path = self.q.get()
            try:
                pcm = load_pcm16k(path)
                code = self.play(pcm)
                self.last = f"재생 {path} ({len(pcm) / 2 / RATE:.1f}s) 코드 {code}"
            except Exception as e:     # 재생 실패가 제어를 멈추면 안 된다
                self.until = 0.0
                self.last = f"재생 실패 {path}: {type(e).__name__} {e}"
            print("  [say]", self.last, flush=True)


if __name__ == "__main__":   # 정책이 안 돌 때만: python3 g1_audio.py eth1 hello.wav [음량]
    import sys
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0, sys.argv[1])
    sp = Speaker(volume=int(sys.argv[3]) if len(sys.argv) > 3 else None)
    print("코드", sp.play(load_pcm16k(sys.argv[2])))
