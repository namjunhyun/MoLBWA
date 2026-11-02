"""--sim_pose_log 결과로 '눈에 보이는' 흔들림: 세그먼트별 머리 수평 흔들림·머리 상하·골반 상하(무릎 출렁임)·발 끌림·머리 가속도.
    python pose_table.py <TAG> ...   (~/molbwa_g1/ma/<TAG>_<part>_<bearing>.pose.npz, multi_angle.sh 산출)"""
import glob, os, sys
import numpy as np

O = os.path.expanduser("~/molbwa_g1/ma")
DT = 0.02


def stats(seg, pos, name):
    if name.startswith("after_"):   # 동작(after_<동작>) 이 끝난 뒤 4 초
        g = np.where(seg == name[6:])[0]
        if len(g) == 0:
            return None
        m = np.zeros(len(seg), bool); m[g[-1] + 1:g[-1] + 1 + int(4.0 / DT)] = True
    else:
        m = seg == name
    if m.sum() < 10:
        return None
    p = pos[m]
    head, pel, feet = p[:, 0], p[:, 1], p[:, 2:]
    sway = np.sqrt(((head[:, :2] - head[:, :2].mean(0)) ** 2).sum(1).mean()) * 100          # cm
    hz = head[:, 2].std() * 100
    pz = pel[:, 2].std() * 100
    foot = np.abs(np.diff(feet[:, :, :2], axis=0)).sum() * 100                                  # cm, 두 발 합
    acc = np.sqrt((np.diff(head, 2, axis=0) ** 2).sum(1).mean()) / DT ** 2                     # m/s²
    return sway, hz, pz, foot, acc


print(f"{'정책':10s} {'동작':16s} {'n':>2s} {'머리수평cm':>10s} {'머리상하cm':>10s} {'골반상하cm':>10s} {'발이동cm':>8s} {'머리가속m/s²':>12s}")
for tag in sys.argv[1:]:
    for part, ph in (("g1_face", "wave"), ("g1_face", "after_wave"), ("g1_hand", "handshake"), ("g1_hand", "after_handshake")):
        rows = []
        for f in sorted(glob.glob(f"{O}/{tag}_{part}_*.pose.npz")):
            z = np.load(f)
            r = stats(z["segment"], z["pos"], ph)
            if r:
                rows.append(r)
        if rows:
            a = np.array(rows)
            print(f"{tag:10s} {ph:16s} {len(a):2d} " + " ".join(f"{m:6.2f}±{s:4.2f}".rjust(w) for m, s, w in
                                                                zip(a.mean(0), a.std(0), (10, 10, 10, 8, 12))))
