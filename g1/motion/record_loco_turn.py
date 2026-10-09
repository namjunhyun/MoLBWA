"""Unitree 사전학습 G1 보행 정책(unitree_rl_gym motion.pt, 다리 12관절)으로 제자리 회전을 MuJoCo 에서 녹화 → 라이브러리 csv.

기본 SDK 보행처럼 발을 디디며 도는 회전 참조를 만든다. 허리·팔은 트래킹 대기 자세로 채운다(관절은 이름 매핑).
    PYTHONPATH=~/unitree_rl_gym python record_loco_turn.py OUT_DIR
"""
import json, os, sys
import mujoco, numpy as np, torch, yaml
from legged_gym import LEGGED_GYM_ROOT_DIR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_idle_csv import CSV_JOINTS  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
cfg = yaml.safe_load(open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/g1.yaml"))
R = lambda s: s.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)  # noqa: E731
kps, kds = np.array(cfg["kps"], np.float32), np.array(cfg["kds"], np.float32)
da = np.array(cfg["default_angles"], np.float32); cs = np.array(cfg["cmd_scale"], np.float32)
na, no, sdt, dec = cfg["num_actions"], cfg["num_obs"], cfg["simulation_dt"], cfg["control_decimation"]
meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy", "g1_tracking_policy_meta.json")))
STANDBY = dict(zip(meta["joint_names"], meta["default_joint_pos"]))
# 보행정책 기본자세(무릎 0.3)는 트래킹 대기자세(무릎 0.669)보다 펴져 있어, 그대로 쓰면 회전 앞뒤로 키가 출렁인다.
# 다리 관절에 두 기본자세 차이를 더해 같은 높이로 웅크린 채 돌게 하고, 발이 바닥에 닿도록 골반을 낮춘다
# (MuJoCo FK: 대기자세에서 발목이 골반 기준 2.75 cm 높다).
RLGYM_DEFAULT = {"hip_pitch": -0.1, "knee": 0.3, "ankle_pitch": -0.2}
LEG_OFFSET = {f"{s}_{j}_joint": STANDBY[f"{s}_{j}_joint"] - v for s in ("left", "right") for j, v in RLGYM_DEFAULT.items()}
PELVIS_DROP = 0.0275
PRE_STAND_S, POST_STAND_S = 0.3, 0.4    # 회전 전후 대기(예전 1.0/1.5 s 는 '멈췄다 홱 도는' 인상)


def gvec(q):
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])


def yaw_of(q):
    qw, qx, qy, qz = q
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def record(target_deg, kp=1.5, wmax=0.6):
    m = mujoco.MjModel.from_xml_path(R(cfg["xml_path"])); d = mujoco.MjData(m); m.opt.timestep = sdt
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[6 + k]) for k in range(na)]
    pol = torch.jit.load(R(cfg["policy_path"]))
    act = np.zeros(na, np.float32); tgt = da.copy(); obs = np.zeros(no, np.float32); cmd = np.zeros(3, np.float32)
    tgt_yaw = np.radians(target_deg); rec = []; t_done = None; ph = 0.0
    for i in range(int(20 / sdt)):
        d.ctrl[:] = (tgt - d.qpos[7:]) * kps - d.qvel[6:] * kds
        mujoco.mj_step(m, d)
        if i % dec:
            continue
        t = i * sdt
        yaw = yaw_of(d.qpos[3:7])
        err = (tgt_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        cmd[2] = 0.0 if t < PRE_STAND_S else np.clip(kp * err, -wmax, wmax)
        if t_done is None and t > PRE_STAND_S and abs(err) < np.radians(3):
            t_done = t
        qj = (d.qpos[7:] - da) * cfg["dof_pos_scale"]; dqj = d.qvel[6:] * cfg["dof_vel_scale"]
        ph = (t % 0.8) / 0.8
        obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]; obs[3:6] = gvec(d.qpos[3:7]); obs[6:9] = cmd * cs
        obs[9:9 + na] = qj; obs[9 + na:9 + 2 * na] = dqj; obs[9 + 2 * na:9 + 3 * na] = act
        obs[9 + 3 * na:9 + 3 * na + 2] = [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]
        act = pol(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
        tgt = act * cfg["action_scale"] + da
        rec.append((t, d.qpos[:3].copy(), d.qpos[3:7].copy(), d.qpos[7:].copy()))
        if d.qpos[2] < 0.4:
            raise RuntimeError(f"turn {target_deg}: 넘어짐 t={t:.1f}")
        # 도착 후 1.5 s 서 있고, 보행 위상이 두 발 지지(위상 0 근처)일 때 끝낸다
        if t_done is not None and t > t_done + POST_STAND_S and ph < 0.03:
            break
    ts = np.array([r[0] for r in rec]); pos = np.array([r[1] for r in rec]); quat = np.array([r[2] for r in rec]); leg = np.array([r[3] for r in rec])
    t30 = np.arange(0, ts[-1] - ts[0] + 1e-9, 1 / 30) + ts[0]
    rows = np.zeros((len(t30), 36))
    for c in range(3):
        rows[:, c] = np.interp(t30, ts, pos[:, c])
    for k in range(1, len(quat)):
        if np.dot(quat[k], quat[k - 1]) < 0:
            quat[k] = -quat[k]
    qi = np.stack([np.interp(t30, ts, quat[:, c]) for c in range(4)], 1); qi /= np.linalg.norm(qi, axis=1, keepdims=True)
    rows[:, 3:7] = qi[:, [1, 2, 3, 0]]
    for j, n in enumerate(CSV_JOINTS):
        rows[:, 7 + j] = (np.interp(t30, ts, leg[:, names.index(n)]) + LEG_OFFSET.get(n, 0.0)) if n in names else STANDBY[n]
    rows[:, 2] -= PELVIS_DROP
    fin = np.degrees(yaw_of(quat[-1]))
    drift = np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1).max()
    return rows, fin, drift, ts[-1], t_done


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from g1_protocol import TURN_BINS  # noqa: E402
for deg in [b for b in TURN_BINS if b != 0]:
    name = f"turn_l{deg}" if deg > 0 else f"turn_r{-deg}"
    rows, fin, drift, T, td = record(deg)
    np.savetxt(f"{OUT}/{name}.csv", rows, delimiter=",", fmt="%.9f")
    print(f"{name}: 목표 {deg:+4d}° 최종 {fin:+7.1f}° · 도착 {td:.1f}s · 길이 {T:.1f}s · 이동 최대 {drift*100:.1f} cm")
