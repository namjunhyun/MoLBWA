"""책상 위 버튼을 오른손 손바닥으로 위에서 꾹 누르는 팔 궤적 (2026-11-16, 고수준 모드 + arm_sdk 용).

    python3 button_press.py OUT.json [--desk 0.70 --btn_h 0.04 --btn_x 0.20 --btn_y -0.25 --depth 0.01]

경로(데카르트, 프레임마다 IK): 서기 자세의 손 → 몸 앞에서 위로(책상 위 여유 높이) → 버튼 위 above m → 아래로 눌러 depth 만큼 더.
관절 보간만 하면 손이 책상 모서리 밑을 지나며 부딪힐 수 있어서 이렇게 한다. 좌표는 골반 원점(+x 정면, +y 왼쪽), z 는 바닥 기준.
서기 자세 = 트래킹 정책 기본 자세(골반 0.76 m). Unitree 고수준 서기와 다를 수 있다 — 실기 첫 시험은 공중에서(책상 없이).
출력 JSON: 오른팔 7관절 이름, approach(접근, 30 fps 행렬), press(누르기), 검사 결과. 떼기는 젯슨이 역순으로 재생한다.
"""
import argparse
import json
import os
import sys

import mujoco as mj
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hand_ik import ARM, FPS, Arm, CSV_JOINTS, min_clearance, smooth_step  # noqa: E402

SIDE = "right"
NAMES = [f"{SIDE}_{a}_joint" for a in ARM]


def standby_row():
    meta = json.load(open(os.path.join(HERE, "..", "deploy", "g1_tracking_policy_meta.json")))
    sb = dict(zip(meta["joint_names"], meta["default_joint_pos"]))
    row = np.zeros(36); row[2] = 0.76; row[6] = 1.0
    row[7:] = [sb[n] for n in CSV_JOINTS]
    return row


def seg(a, b, n):
    return [a + smooth_step((k + 1) / n) * (b - a) for k in range(n)]


def build(desk=0.70, btn_h=0.04, btn_x=0.20, btn_y=-0.25, depth=0.01, above=0.08, clear=0.15, lift_x=0.12,
          t_move=2.0, t_down=1.0):
    """궤적은 '들어 올린 자세 L'(손 = 골반 앞 lift_x, 책상 위 clear, 손바닥 아래)에서 시작한다. 실기 시작 자세(Unitree 고수준
    서기)는 모르므로 젯슨이 실제 현재 자세 → L 을 관절 보간한다(운용 조건: 손이 책상 앞 모서리보다 뒤에 있게 세운다)."""
    arm = Arm()
    row = standby_row()
    arm.set(row)
    th = arm.hand[SIDE][3]                         # 손 두께(손바닥 법선 방향)
    top = desk + btn_h                             # 버튼 윗면
    press = np.array([btn_x, btn_y, top + th / 2 - depth])     # 손바닥 면이 depth 만큼 버튼을 누른 때의 손 중심
    abv = press + np.array([0, 0, depth + above])
    lift = np.array([lift_x, btn_y, desk + clear + th / 2])
    down_n = np.array([0, 0, -1.0])
    # 누르기 자세를 서기에서 바로 푼 뒤(정확·손목 안 비튼 해) 거꾸로 올라가며 이어 푼다 → 뒤집는다.
    # 2026-11-16: L 에서 출발해 내려오면 손목이 비틀린 해(yaw −1.19)에 갇혀 누르기 점에 12 mm 못 미쳤다.
    q_press = arm.solve(row, SIDE, press, down_n, w_ori=0.5, iters=300)
    cur = q_press.copy()
    up_d, up_a = [], []
    for p in seg(press, abv, int(t_down * FPS)):
        cur = arm.solve(cur, SIDE, p, down_n, w_ori=0.5, q_ref=q_press, lam=0.05, iters=80)
        up_d.append(cur.copy())
    for p in seg(abv, lift, int(t_move * FPS)):
        cur = arm.solve(cur, SIDE, p, down_n, w_ori=0.5, q_ref=q_press, lam=0.05, iters=80)
        up_a.append(cur.copy())
    rows_a = np.array(up_a[::-1] + [up_d[-1]])     # L → 버튼 위
    rows_d = np.array(up_d[::-1][1:] + [q_press])  # 버튼 위 → 누름
    q_lift = rows_a[0]
    allr = np.vstack([rows_a, rows_d])
    arm.set(q_lift); lift_err = float(np.linalg.norm(arm.palm(SIDE)[0] - lift) * 1000)

    # 검사 1: 목표 추종 오차 / 2: 팔-몸통 간격 / 3: 책상 위(모서리 = 몸통 앞 desk_edge 이후엔 팔 몸체가 윗면보다 위) / 4: 관절속도
    arm.set(rows_d[-1]); pe, ne = arm.palm(SIDE)
    err_mm = float(np.linalg.norm(pe - press) * 1000)
    tilt = float(np.degrees(np.arccos(np.clip(-ne[2], -1, 1))))
    clr = min_clearance(allr, arm)
    bodies = [mj.mj_name2id(arm.m, mj.mjtObj.mjOBJ_BODY, f"{SIDE}_{b}") for b in
              ("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")]
    desk_edge = 0.12
    low = 9.0
    for r in rows_a:                               # 접근 구간(누르기 전): 팔꿈치·손목·손바닥 아랫면이 책상 윗면보다 위
        arm.set(r)
        pts = [arm.d.xpos[b] for b in bodies] + [arm.palm(SIDE)[0] - np.array([0, 0, th / 2])]
        for x, _, z in pts:
            if x > desk_edge:
                low = min(low, z - desk)
    jmax = float(np.abs(np.diff(allr[:, 7:], axis=0)).max() * FPS)
    ci = [7 + CSV_JOINTS.index(n) for n in NAMES]
    out = {"joint_names": NAMES, "fps": FPS, "q_lift": rows_a[0, ci].round(5).tolist(),
           "approach": rows_a[:, ci].round(5).tolist(), "press": rows_d[:, ci].round(5).tolist(),
           "params": dict(desk=desk, btn_h=btn_h, btn_x=btn_x, btn_y=btn_y, depth=depth, above=above, lift_x=lift_x),
           "check": {"lift_err_mm": round(lift_err, 1), "press_err_mm": round(err_mm, 1), "palm_tilt_deg": round(tilt, 1),
                     "arm_torso_clear_m": round(clr["R-torso"], 3), "arm_above_desk_min_m": round(low, 3),
                     "joint_speed_max_rad_s": round(jmax, 2), "hand_thickness_m": round(float(th), 3)}}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    for k, v in dict(desk=0.70, btn_h=0.04, btn_x=0.20, btn_y=-0.25, depth=0.01, above=0.08, lift_x=0.12).items():
        ap.add_argument(f"--{k}", type=float, default=v)
    a = ap.parse_args()
    res = build(a.desk, a.btn_h, a.btn_x, a.btn_y, a.depth, a.above, lift_x=a.lift_x)
    json.dump(res, open(a.out, "w"))
    c = res["check"]
    print(f"{a.out}: 접근 {len(res['approach'])} + 누르기 {len(res['press'])} 프레임 | {c}")
    ok = c["lift_err_mm"] < 40 and c["press_err_mm"] < 10 and c["palm_tilt_deg"] < 15 and c["arm_torso_clear_m"] > 0.02 \
        and c["arm_above_desk_min_m"] > 0.02 and c["joint_speed_max_rad_s"] < 3.0
    print("검사 통과" if ok else "검사 실패 — 위치·높이 조정 필요")
    sys.exit(0 if ok else 1)
