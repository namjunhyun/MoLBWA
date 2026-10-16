"""손 위치 + 손바닥 방향을 맞추는 팔 IK (MuJoCo 운동학, 감쇠 최소제곱). 라이브러리 csv(36열)를 만들거나 고친다.

    # 가리키기: 기존 클립의 손 위치는 그대로, 오른손 손바닥을 위로 (팔을 뻗은 구간에만, 앞뒤는 부드럽게)
    python3 hand_ik.py palm_up IN.csv OUT.csv --side right
    # 박수: 아래 손(왼손) 손바닥 위로 가슴 앞에 받치고, 위 손(오른손) 손바닥 아래로 내려쳐 4번
    python3 hand_ik.py clap_stack OUT.csv [--claps 4] [--hz 1.6]

손바닥 방향 = rubber_hand 메시의 가장 얇은 축, 부호는 손가락이 굽는 쪽(=손바닥). 2026-10-14 사용자 요청:
"가리키기 손바닥 위로, 박수는 손바닥 마주보게 위아래로(한 손 위·한 손 아래)". 데이터셋 손목 관절은 전부 0 이라 새로 만든다.
"""
import argparse
import os
import sys

import numpy as np
import mujoco as mj

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_idle_csv import CSV_JOINTS  # noqa: E402

XML = os.environ.get("G1_MUJOCO_XML", os.path.expanduser(
    "~/unitree_g1_vibes/RL-shenanigans/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"))
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
FPS = 30


class Arm:
    def __init__(self):
        self.m = mj.MjModel.from_xml_path(XML)
        self.d = mj.MjData(self.m)
        self.adr = {n: self.m.jnt_qposadr[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, n)] for n in CSV_JOINTS}
        self.dof = {n: self.m.jnt_dofadr[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, n)] for n in CSV_JOINTS}
        self.lim = {n: self.m.jnt_range[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, n)] for n in CSV_JOINTS}
        self.hand = {}
        for s in ("left", "right"):
            g = next(g for g in range(self.m.ngeom) if self.m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH
                     and mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_MESH, self.m.geom_dataid[g]) == f"{s}_rubber_hand")
            mid = self.m.geom_dataid[g]
            v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]].astype(float)
            c = v - v.mean(0)
            _, U = np.linalg.eigh(c.T @ c)
            n, f = U[:, 0], U[:, 2]
            tip = c[(c @ f) > np.percentile(c @ f, 85)]
            if abs(np.percentile(c @ f, 85)) < abs(np.percentile(c @ f, 15)):
                tip = c[(c @ f) < np.percentile(c @ f, 15)]
            if (tip @ n).mean() < 0:      # 손끝이 굽는 쪽 = 손바닥
                n = -n
            th = (c @ n).max() - (c @ n).min()
            self.hand[s] = (g, n, v.mean(0), th)

    def set(self, row):
        d = self.d
        d.qpos[:] = 0
        d.qpos[:3] = row[:3]
        d.qpos[3:7] = row[[6, 3, 4, 5]]
        for j, n in enumerate(CSV_JOINTS):
            d.qpos[self.adr[n]] = row[7 + j]
        mj.mj_forward(self.m, d)

    def palm(self, s):
        """(손 중심 world, 손바닥 법선 world)"""
        g, n, ctr, _ = self.hand[s]
        R = self.d.geom_xmat[g].reshape(3, 3)
        return self.d.geom_xpos[g] + R @ ctr, R @ n

    def solve(self, row, side, p_tgt=None, n_tgt=None, w_ori=1.0, iters=60, q_ref=None, lam=0.15, bounds=None):
        """row 의 side 팔 7관절만 바꿔 손 중심 → p_tgt, 손바닥 법선 → n_tgt. 관절 한계 안. 다른 관절 불변.
        q_ref(36열 행)가 있으면 팔 관절을 그 자세 쪽으로 lam 만큼 당긴다 — 7관절이라 해가 여러 개라
        프레임마다 다른 해로 튀던 것(관절속도 40 rad/s)을 막는다."""
        row = row.copy()
        names = [f"{side}_{a}_joint" for a in ARM]
        g = self.hand[side][0]
        b = self.m.geom_bodyid[g]
        for _ in range(iters):
            self.set(row)
            pc, nc = self.palm(side)
            e, J = [], []
            jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
            mj.mj_jac(self.m, self.d, jp, jr, pc, b)
            cols = [self.dof[n] for n in names]
            if p_tgt is not None:
                e.append(p_tgt - pc); J.append(jp[:, cols])
            if n_tgt is not None:
                # 법선 회전 오차: nc → n_tgt 로 돌리는 축각, ṅ = ω × n
                e.append(w_ori * np.cross(nc, n_tgt)); J.append(w_ori * jr[:, cols])
            if q_ref is not None:
                ci = [7 + CSV_JOINTS.index(n) for n in names]
                e.append(lam * (q_ref[ci] - row[ci])); J.append(lam * np.eye(len(names)))
            e = np.concatenate(e); J = np.vstack(J)
            if np.linalg.norm(e[:6]) < 1e-4:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(len(e)), e)
            for k, n in enumerate(names):
                lo, hi = (bounds or {}).get(n, self.lim[n])
                row[7 + CSV_JOINTS.index(n)] = np.clip(row[7 + CSV_JOINTS.index(n)] + 0.5 * dq[k], lo + 0.02, hi - 0.02)
        return row


def smooth_step(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def palm_up(rows, side, arm: Arm, ramp_s=0.6, yaw_dir=0, inside=False, up_frac=1.0, wrist_lock=None):
    """팔을 뻗은 구간(어깨 pitch 가 최대 변위의 60% 이상)에서 손바닥을 위로. 손 위치는 원본 유지. 앞뒤 ramp 로 섞는다."""
    jp = 7 + CSV_JOINTS.index(f"{side}_shoulder_pitch_joint")
    dev = np.abs(rows[:, jp] - rows[0, jp])
    on = dev >= 0.6 * dev.max()
    idx = np.where(on)[0]
    a, b = idx[0], idx[-1]
    r = int(ramp_s * FPS)
    if inside:   # 팔을 다 든 뒤에 돌리기 시작해 내리기 전에 되돌림 (2026-10-15: 팔이 내려간 채 어깨가 0→46° 확 돌던 것)
        w = np.array([smooth_step((k - a) / r) * smooth_step((b - k) / r) for k in range(len(rows))])
    else:
        w = np.array([smooth_step((k - (a - r)) / r) * smooth_step(((b + r) - k) / r) for k in range(len(rows))])
    w = w * up_frac
    out = rows.copy()
    prev = None
    for k in range(len(rows)):
        if w[k] <= 0:
            continue
        arm.set(rows[k])
        p, n = arm.palm(side)
        tgt = n + w[k] * (np.array([0, 0, 1.0]) - n)        # 원래 법선 → 위 로 비율만큼
        tgt /= np.linalg.norm(tgt)
        seed = rows[k].copy()
        if prev is not None:                                   # 이전 프레임 해로 시작(연속성)
            for a_ in ARM:
                c = 7 + CSV_JOINTS.index(f"{side}_{a_}_joint")
                seed[c] = prev[c] + (rows[k][c] - rows[k - 1][c])
        bounds = None
        if yaw_dir:   # 어깨 yaw 를 원본에서 한쪽으로만 — 반대쪽은 위팔이 몸통을 파고든다(2026-10-15, −4 cm)
            yn = f"{side}_shoulder_yaw_joint"; y0 = rows[k][7 + CSV_JOINTS.index(yn)]; lo, hi = arm.lim[yn]
            bounds = {yn: (y0 - 0.02, hi) if yaw_dir > 0 else (lo, y0 + 0.02)}
        if wrist_lock is not None:   # 손목을 곧게 — 2026-10-16: 손목을 꺾어 손바닥을 돌려 '좀비 팔'처럼 보였다
            bounds = dict(bounds or {}, **wrist_bounds(side, wrist_lock))
        out[k] = arm.solve(seed, side, p_tgt=p, n_tgt=tgt, w_ori=0.5, q_ref=rows[k], bounds=bounds)
        prev = out[k]
    return out, w


def wrist_bounds(side, lock):
    """손목 pitch·yaw 를 ±lock(rad) 로 — 손바닥 방향은 팔뚝 회전(wrist_roll)·어깨로 만들게 한다."""
    return {f"{side}_wrist_pitch_joint": (-lock, lock), f"{side}_wrist_yaw_joint": (-lock, lock)}


def clap_stack(standby_row, arm: Arm, claps=4, hz=1.6, lift=0.12, fwd=0.30, left=0.0, gap_extra=0.01, up=0.12,
               elbow_out=0.0, wrist_lock=None):
    """아래 손(왼손) 손바닥 위로 가슴 앞, 위 손(오른손) 손바닥 아래로 lift 만큼 들었다 내려치기."""
    arm.set(standby_row)
    torso = arm.d.xpos[mj.mj_name2id(arm.m, mj.mjtObj.mjOBJ_BODY, "torso_link")].copy()
    yaw_R = arm.d.xmat[mj.mj_name2id(arm.m, mj.mjtObj.mjOBJ_BODY, "pelvis")].reshape(3, 3)
    fwdv, leftv = yaw_R[:, 0], yaw_R[:, 1]
    fwd_m, left_m = fwd, left
    base = torso + fwd_m * fwdv + left_m * leftv + np.array([0, 0, up])   # 몸통 원점 기준 앞·왼·위
    gap = 0.5 * (arm.hand["left"][3] + arm.hand["right"][3]) + gap_extra   # 손 두께 + 여유 (학습 시뮬은 자기 충돌 켜짐 → 닿으면 못 따라간다)
    up, down = np.array([0, 0, 1.0]), np.array([0, 0, -1.0])
    t_in, t_out = 1.0, 1.0
    t_clap = claps / hz
    T = t_in + t_clap + t_out
    N = int(T * FPS)
    rows = np.repeat(standby_row[None], N, 0)
    arm.set(standby_row)
    pL0, nL0 = arm.palm("left"); pR0, nR0 = arm.palm("right")
    cur = standby_row.copy()
    for k in range(N):
        t = k / FPS
        a_in = smooth_step(t / t_in); a_out = smooth_step((T - t) / t_out)
        a = min(a_in, a_out)
        if t_in <= t <= t_in + t_clap:
            ph = (t - t_in) * hz
            h = gap + lift * 0.5 * (1 + np.cos(2 * np.pi * ph))     # 위에서 시작 → 바닥에서 짝
        else:
            h = gap + lift
        pL = pL0 + a * (base - pL0)
        pR = pR0 + a * (base + np.array([0, 0, h]) - pR0)
        nL = nL0 + a * (up - nL0); nL /= np.linalg.norm(nL)
        nR = nR0 + a * (down - nR0); nR /= np.linalg.norm(nR)
        if elbow_out:   # 팔꿈치를 굽혀 바깥으로 벌린 자세 쪽으로 당긴다 — 손을 몸 가까이 둬도 팔이 몸통에 안 닿게(2026-10-16)
            ref = standby_row.copy()
            ref[7 + CSV_JOINTS.index("left_shoulder_roll_joint")] += a * elbow_out
            ref[7 + CSV_JOINTS.index("right_shoulder_roll_joint")] -= a * elbow_out
        else:
            ref = standby_row if a < 1e-3 else cur      # 밖으로 나갈 땐 대기자세 쪽으로, 박수 중엔 직전 프레임 쪽으로
        bl = None if wrist_lock is None else wrist_bounds("left", wrist_lock)
        br = None if wrist_lock is None else wrist_bounds("right", wrist_lock)
        cur = arm.solve(cur, "left", pL, nL, w_ori=0.5, q_ref=ref, bounds=bl)
        cur = arm.solve(cur, "right", pR, nR, w_ori=0.5, q_ref=ref, bounds=br)
        rows[k] = cur
    return rows


def point_forward(standby_row, arm: Arm, side="right", reach=0.55, down_deg=35.0, out=0.08, t_up=1.2, t_hold=2.5,
                  t_down=1.2, up_frac=0.85, wrist_lock=0.2, roll_out=0.25):
    """IK 로 처음부터 만든 가리키기: 어깨에서 정면 아래(down_deg) 로 reach 만큼 손을 뻗어 손바닥을 위로(up_frac) 들고
    t_hold 동안 멈췄다가 내린다. 2026-10-16: 모캡 가리키기를 고친 판은 팔이 몸을 가로지르고 손목이 꺾여 어색했다."""
    arm.set(standby_row)
    sh = arm.d.xpos[mj.mj_name2id(arm.m, mj.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_link")].copy()
    R = arm.d.xmat[mj.mj_name2id(arm.m, mj.mjtObj.mjOBJ_BODY, "pelvis")].reshape(3, 3)
    fwdv, leftv = R[:, 0], R[:, 1]
    sgn = -1.0 if side == "right" else 1.0
    a = np.radians(down_deg)
    tgt = sh + reach * (np.cos(a) * fwdv - np.sin(a) * np.array([0, 0, 1.0])) + sgn * out * leftv
    p0, n0 = arm.palm(side)
    up = np.array([0, 0, 1.0])
    T = t_up + t_hold + t_down
    N = int(T * FPS)
    rows = np.repeat(standby_row[None], N, 0)
    ref = standby_row.copy()
    ref[7 + CSV_JOINTS.index(f"{side}_shoulder_roll_joint")] += sgn * roll_out * -1.0   # 팔꿈치 살짝 바깥
    cur = standby_row.copy()
    for k in range(N):
        t = k / FPS
        w = min(smooth_step(t / t_up), smooth_step((T - t) / t_down))
        p = p0 + w * (tgt - p0)
        nt = n0 + w * up_frac * (up - n0); nt /= np.linalg.norm(nt)
        r_ = standby_row + w * (ref - standby_row)
        cur = arm.solve(cur, side, p, nt, w_ori=0.5, q_ref=r_, bounds=wrist_bounds(side, wrist_lock))
        rows[k] = cur
    return rows


def report(rows, arm: Arm, sides):
    out = []
    for s in sides:
        nz = []
        for r in rows:
            arm.set(r); nz.append(arm.palm(s)[1][2])
        out.append(f"{s} 손바닥 법선 z 최대 {max(nz):+.2f} 최소 {min(nz):+.2f}")
    v = np.abs(np.diff(rows[:, 7:], axis=0)).max() * FPS
    return " | ".join(out) + f" | 관절속도 최대 {v:.1f} rad/s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["palm_up", "clap_stack", "point_fwd"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--side", default="right")
    ap.add_argument("--claps", type=int, default=4)
    ap.add_argument("--hz", type=float, default=1.6)
    ap.add_argument("--lift", type=float, default=0.12)
    ap.add_argument("--fwd", type=float, default=0.30, help="박수 위치: 몸통 앞 거리(m)")
    ap.add_argument("--left", type=float, default=0.0, help="박수 위치: 왼쪽(+) 치우침(m) — 왼팔이 덜 가로지르게")
    ap.add_argument("--gap", type=float, default=0.01, help="손 두께 외 두 손 사이 여유(m)")
    ap.add_argument("--up", type=float, default=0.12, help="받침 손 높이: 몸통 원점 위(m)")
    ap.add_argument("--ramp", type=float, default=0.6, help="palm_up: 손바닥 돌리는 시간(s)")
    ap.add_argument("--inside", action="store_true", help="palm_up: 팔을 다 든 뒤에 돌리기 시작")
    ap.add_argument("--up_frac", type=float, default=1.0, help="palm_up: 손바닥을 위로 돌리는 비율(1 = 완전히 위)")
    ap.add_argument("--reach", type=float, default=0.58, help="point_fwd: 어깨→손 거리(m)")
    ap.add_argument("--down", type=float, default=30.0, help="point_fwd: 수평 아래 각(도)")
    ap.add_argument("--out", type=float, default=0.12, help="point_fwd: 바깥(오른손이면 오른쪽) 치우침(m)")
    ap.add_argument("--elbow_out", type=float, default=0.0, help="clap_stack: 팔꿈치 바깥 벌림 정규화(rad, 어깨 roll)")
    ap.add_argument("--wrist_lock", type=float, default=None, help="손목 pitch·yaw 를 ±값(rad)으로 제한 (곧은 손목)")
    ap.add_argument("--yaw_dir", type=int, default=0, help="palm_up: 어깨 yaw 를 원본에서 +1(증가만)/−1(감소만) 으로 제한")
    a = ap.parse_args()
    arm = Arm()
    if a.mode == "palm_up":
        rows = np.loadtxt(a.paths[0], delimiter=",")
        out, _ = palm_up(rows, a.side, arm, ramp_s=a.ramp, yaw_dir=a.yaw_dir, inside=a.inside, up_frac=a.up_frac, wrist_lock=a.wrist_lock)
        np.savetxt(a.paths[1], out, delimiter=",", fmt="%.9f")
        print(a.paths[1], report(out, arm, [a.side]))
    elif a.mode == "point_fwd":
        import json
        meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy", "g1_tracking_policy_meta.json")))
        sb = dict(zip(meta["joint_names"], meta["default_joint_pos"]))
        row = np.zeros(36); row[2] = 0.76; row[6] = 1.0
        row[7:] = [sb[n] for n in CSV_JOINTS]
        out = point_forward(row, arm, a.side, a.reach, a.down, a.out, wrist_lock=0.2 if a.wrist_lock is None else a.wrist_lock)
        np.savetxt(a.paths[0], out, delimiter=",", fmt="%.9f")
        print(a.paths[0], f"{len(out)} 프레임 {len(out)/FPS:.2f}s", report(out, arm, [a.side]))
    else:
        idle = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy", "g1_tracking_policy_meta.json")
        import json
        meta = json.load(open(idle))
        sb = dict(zip(meta["joint_names"], meta["default_joint_pos"]))
        row = np.zeros(36); row[2] = 0.76; row[6] = 1.0
        row[7:] = [sb[n] for n in CSV_JOINTS]
        out = clap_stack(row, arm, a.claps, a.hz, a.lift, a.fwd, a.left, a.gap, a.up, a.elbow_out, a.wrist_lock)
        np.savetxt(a.paths[0], out, delimiter=",", fmt="%.9f")
        print(a.paths[0], f"{len(out)} 프레임 {len(out)/FPS:.2f}s", report(out, arm, ["left", "right"]))


def min_clearance(rows, arm: Arm):
    """충돌 geom(contype≠0) 기준 최소 거리: 왼팔↔오른팔, 왼팔↔몸통, 오른팔↔몸통 (m, 음수 = 관통)."""
    m, d = arm.m, arm.d
    def geoms(pred):
        return [g for g in range(m.ngeom) if m.geom_contype[g] and pred(mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")]
    armL = geoms(lambda b: b.startswith("left_") and any(k in b for k in ("elbow", "wrist", "hand", "shoulder_yaw")))
    armR = geoms(lambda b: b.startswith("right_") and any(k in b for k in ("elbow", "wrist", "hand", "shoulder_yaw")))
    torso = geoms(lambda b: b in ("torso_link", "pelvis"))
    best = {"L-R": 9.0, "L-torso": 9.0, "R-torso": 9.0}
    ft = np.zeros(6)
    for r in rows:
        arm.set(r)
        for key, A, B in (("L-R", armL, armR), ("L-torso", armL, torso), ("R-torso", armR, torso)):
            for ga in A:
                for gb in B:
                    best[key] = min(best[key], mj.mj_geomDistance(m, d, ga, gb, 0.3, ft))
    return best
