"""흔들림 비교: 세그먼트(phase)별 몸통 롤·피치 각속도 RMS, 기울기 최대, 관절 목표 떨림(2차 차분 RMS).
    python wobble.py a.csv b.csv ...   (g1_motion_server --log_csv 산출)"""
import csv, sys
import numpy as np


def tilt_deg(w, x, y, z):
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


for f in sys.argv[1:]:
    rows = list(csv.DictReader(open(f)))
    tg = [k for k in rows[0] if k.startswith("tgt_")]
    print(f"== {f.split('/')[-1]}")
    print(f"  {'phase':12s} {'n':>5s} {'롤피치각속도RMS°/s':>18s} {'기울기최대°':>10s} {'목표떨림°':>9s}")
    for ph in dict.fromkeys(r["phase"] for r in rows):
        R = [r for r in rows if r["phase"] == ph]
        g = np.array([[float(r["gyro_x"]), float(r["gyro_y"])] for r in R])
        q = np.array([[float(r[k]) for k in ("imu_w", "imu_x", "imu_y", "imu_z")] for r in R])
        T = np.array([[float(r[k]) for k in tg] for r in R])
        jit = np.degrees(np.sqrt((np.diff(T, 2, axis=0) ** 2).mean())) if len(T) > 2 else 0
        print(f"  {ph:12s} {len(R):5d} {np.degrees(np.sqrt((g ** 2).sum(1).mean())):18.1f} {tilt_deg(*q.T).max():10.1f} {jit:9.2f}")
