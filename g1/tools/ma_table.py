"""multi_angle.sh 결과 요약: 정책별 동작 구간 골반·몸통(골반+허리) 롤·피치 각속도 RMS, 방위 6개 평균±표준편차."""
import csv, glob, os, sys
import numpy as np
O = os.path.expanduser("~/molbwa_g1/ma")
for tag in sys.argv[1:]:
    for part, ph in (("g1_face", "wave"), ("g1_hand", "handshake")):
        pel, tor, fell = [], [], 0
        for f in sorted(glob.glob(f"{O}/{tag}_{part}_*.csv")):
            R = [r for r in csv.DictReader(open(f)) if r["phase"] == ph]
            if "서 있음" not in open(f[:-4] + ".log").read():
                fell += 1
            if not R:
                continue
            g = np.array([[float(r["gyro_x"]), float(r["gyro_y"])] for r in R])
            w = np.array([[float(r["act_waist_roll_joint"]), float(r["act_waist_pitch_joint"])] for r in R])
            t = g + np.gradient(w, 0.02, axis=0)
            pel.append(np.degrees(np.sqrt((g ** 2).sum(1).mean()))); tor.append(np.degrees(np.sqrt((t ** 2).sum(1).mean())))
        print(f"{tag:12s} {ph:9s} n={len(pel)} 골반 {np.mean(pel):5.1f}±{np.std(pel):3.1f}  몸통 {np.mean(tor):5.1f}±{np.std(tor):3.1f}  넘어짐/중단 {fell}")
