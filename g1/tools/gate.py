"""정책 채택 관문 한 줄 요약 (multi_angle.sh 산출, 방위 6 × 손 흔들기·악수).
    서있기 진동 = 동작 전 3 s / 동작 후 3~5 s 머리 가속 RMS (m/s²), 흔들림 = 머리 수평 RMS (cm).
    python gate.py <TAG> ..."""
import glob, os, sys
import numpy as np

O = os.path.expanduser("~/molbwa_g1/ma")


def sway(p):
    return np.sqrt(((p[:, 0, :2] - p[:, 0, :2].mean(0)) ** 2).sum(1).mean()) * 100


for tag in sys.argv[1:]:
    pre, post, mv, af, n = [], [], [], [], 0
    for part, ph in (("g1_face", "wave"), ("g1_hand", "handshake")):
        for f in sorted(glob.glob(f"{O}/{tag}_{part}_*.pose.npz")):
            z = np.load(f); s, p = z["segment"], z["pos"]
            if ph not in s:
                continue
            n += 1
            a = np.sqrt((np.diff(p[:, 0], 2, axis=0) ** 2).sum(1)) / 0.02 ** 2
            first = np.where(s != "idle")[0][0]; g = np.where(s == ph)[0]
            pre.append(a[max(0, first - 150):first - 2].mean()); post.append(a[g[-1] + 150:g[-1] + 248].mean())
            mv.append(sway(p[g])); af.append(sway(p[g[-1] + 1:g[-1] + 201]))
    fell = sum("서 있음" not in open(f).read() for f in glob.glob(f"{O}/{tag}_*.log"))
    print(f"{tag:10s} n={n:2d} 서있기진동 전 {np.mean(pre):.2f} 후 {np.mean(post):.2f}(최대 {np.max(post):.2f})  "
          f"동작중 머리 {np.mean(mv):.2f}cm  동작후 머리 {np.mean(af):.2f}cm  넘어짐/중단 {fell}")
