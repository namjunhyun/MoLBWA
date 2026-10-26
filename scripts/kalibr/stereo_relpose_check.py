#!/usr/bin/env python3
"""캘리브레이션 bag 에서 사진마다 두 카메라 위치를 따로 풀어 상대 위치(|T|·회전)가 일정한지 본다.
   회전이 사진마다 몇 도씩 흔들리면 카메라 고정이 휘는 것 — 스테레오 캘리브레이션이 발산한다(2026-10-26 3.3~8.7°).
   python3 scripts/kalibr/stereo_relpose_check.py   (경로는 아래 R·bag 이름을 고쳐 쓴다)
"""
import sys, cv2, numpy as np, yaml
sys.path.insert(0, "/home/junhyun/창종설/.claude/worktrees/ov9281-stereo/scripts/kalibr")
import stereo_extrinsic_opencv as S
R = "/home/junhyun/창종설/.claude/worktrees/ov9281-stereo/"
K0, D0, _ = S.cam(R + "calibration/ov9281/mono_left_camchain.yaml")
K1, D1, _ = S.cam(R + "calibration/ov9281/mono_right_camchain.yaml")
tg = yaml.safe_load(open(R + "calibration/target.yaml")); obj = S.obj_points(tg, False)
pairs = S.detections(R + "calibration/bags/ov9281_02", 4.0, 8)
rows = []
def pnp(D, K, Dd):
    ids = sorted(set(D) & set(obj)); o = np.concatenate([obj[i] for i in ids]); im = np.concatenate([D[i] for i in ids]).astype(np.float32)
    ok, rv, tv = cv2.solvePnP(o, im, K, Dd); pr, _ = cv2.projectPoints(o, rv, tv, K, Dd)
    return cv2.Rodrigues(rv)[0], tv.reshape(3), np.sqrt(((pr.reshape(-1, 2) - im) ** 2).sum(1)).mean(), len(ids)
for k, (L, Rr) in enumerate(pairs):
    R0, t0, e0, n0 = pnp(L, K0, D0); R1, t1, e1, n1 = pnp(Rr, K1, D1)
    Rrl = R1 @ R0.T; trl = t1 - Rrl @ t0          # p_right = Rrl p_left + trl
    ang = np.degrees(np.arccos(np.clip((np.trace(Rrl) - 1) / 2, -1, 1)))
    rows.append((k, np.linalg.norm(trl) * 100, *(trl * 100), ang, e0, e1, np.linalg.norm(t0) * 100, n0, n1))
a = np.array(rows)
print("쌍  |T|cm   Tx    Ty    Tz   회전°  오차L  오차R  판거리cm  태그L/R")
for r in a[::3]:
    print(f"{int(r[0]):3d} {r[1]:6.1f} {r[2]:6.1f} {r[3]:5.1f} {r[4]:5.1f} {r[5]:5.1f} {r[6]:5.2f} {r[7]:5.2f} {r[8]:7.1f}   {int(r[9])}/{int(r[10])}")
good = a[(a[:, 6] < 2) & (a[:, 7] < 2)]
print(f"\n전체 {len(a)}쌍 · |T| 중앙 {np.median(a[:,1]):.1f} cm (사분위 {np.percentile(a[:,1],25):.1f}~{np.percentile(a[:,1],75):.1f}) · 회전 중앙 {np.median(a[:,5]):.1f}°")
print(f"단독 오차 둘 다 <2px 인 {len(good)}쌍 · |T| 중앙 {np.median(good[:,1]):.1f} cm (사분위 {np.percentile(good[:,1],25):.1f}~{np.percentile(good[:,1],75):.1f})")
