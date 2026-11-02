import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
from g1_audio import RATE, load_pcm16k  # noqa: E402


def _wav(path, sr, ch, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.asarray(x, "<i2").tobytes())


def test_resample_and_mono(tmp_path):
    sr = 44100
    t = np.arange(sr) / sr                                  # 1 초
    tone = (8000 * np.sin(2 * np.pi * 440 * t)).astype(int)
    st = np.stack([tone, tone], 1).reshape(-1)              # 스테레오 같은 신호
    p = tmp_path / "a.wav"; _wav(p, sr, 2, st)
    pcm = np.frombuffer(load_pcm16k(str(p)), "<i2")
    assert abs(len(pcm) - RATE) <= 1                        # 1 초 = 16000 샘플
    assert 7000 < np.abs(pcm).max() <= 8000                 # 진폭 보존(채널 평균)


def test_already_16k_unchanged(tmp_path):
    x = np.arange(-100, 100) * 50
    p = tmp_path / "b.wav"; _wav(p, RATE, 1, x)
    assert np.array_equal(np.frombuffer(load_pcm16k(str(p)), "<i2"), x)
