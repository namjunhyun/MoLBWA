#!/usr/bin/env python3
"""G1 모션 서버 — 모션 라이브러리 + 트래킹 정책 1개 (Wo-State-Estimation, 관측 154차원).

~/g1_dance_deploy/deploy_g1_tracking.py(실기 검증본)를 복사해 확장했다. 바뀐 것:
  * 고정 모션 1회 재생 대신 SegmentPlayer 커서로 idle 반복 -> act 명령 -> turn_<bin> -> 동작 -> idle.
  * 세그먼트 첫 프레임마다 yaw_off 를 재정렬한다(원본은 시작 시 1회).
  * UDP 명령 수신(--cmd_port, 비차단) / 5 Hz 상태 송신(--state_host:--state_port).
    명령 채널이 끊기면 idle 을 계속 재생한다 — 댐핑하지 않는다(정책을 끄면 넘어진다).
  * --gain_ramp / --motion_scale / --pre_stand / --squat_exit 제거(플레이북 5·8장: 약하게 시작·
    쪼그림 종료는 성립하지 않는다, pre_stand 는 세그먼트 인덱스를 밀어낸다).

**기본은 dry-run 이다.** 실제 DDS 발행은 --arm 을 명시해야 한다.
로봇은 하네스(느슨하게)·매트·비상정지 담당이 있을 때만 시험한다.

설계 근거 (원본과 같음, 모두 측정으로 확인된 것):
  * 관절 순서: 정책은 Isaac Lab 의 트리 BFS 순서로 말하고, SDK 모터 인덱스는 URDF 순서다.
    양방향 재정렬이 필수다 (생략하면 sim2sim 에서 1초 안에 넘어졌다).
  * 관측 154차원 = command(58) + motion_anchor_ori_b(6) + base_ang_vel(3)
                   + joint_pos(29) + joint_vel(29) + actions(29)
    전부 IMU + 엔코더로 얻어진다. world 위치·선속도는 쓰지 않는다.
  * 이 스크립트는 로봇 온보드(Jetson)에서 돌아야 한다. 데스크탑 원격은 불가.
  * 보간 진입은 하지 않는다(--enter_sec 0) — 정책이 균형을 잡는 주체다.
"""
from __future__ import annotations

import argparse
import itertools
import socket
import subprocess, json, os, signal, sys, time
import numpy as np

# segment_player 가 g1_protocol 을 같은 폴더(Jetson) 또는 부모 폴더(저장소)에서 찾는다.
from segment_player import SegmentPlayer
from g1_protocol import PART_TO_SEGMENT, PORT_CMD, PORT_STATE, TURN_BINS, decode, encode, turn_segment


class _SafeOut:
    """콘솔 SSH 가 먼저 끊기면 print 가 BrokenPipeError 를 던져 비상 댐핑 전에 죽는다(재현함).
    출력 실패는 삼키고 제어는 계속한다."""
    def __init__(self, f):
        self.f = f
    def write(self, s):
        try:
            return self.f.write(s)
        except (OSError, ValueError):
            return len(s)
    def flush(self):
        try:
            self.f.flush()
        except (OSError, ValueError):
            pass
    def __getattr__(self, k):
        return getattr(self.f, k)
sys.stdout, sys.stderr = _SafeOut(sys.stdout), _SafeOut(sys.stderr)

# ---------------------------------------------------------------- 인자
AP = argparse.ArgumentParser()
AP.add_argument("--policy", required=True, help="Wo-State-Estimation 으로 학습한 체크포인트(.pt)")
AP.add_argument("--motion", required=True,
                help="모션 라이브러리 npz (csv_to_npz.py 산출, Isaac 관절 순서, 50 fps)")
AP.add_argument("--library_meta", required=True,
                help="build_motion_library.py 가 쓴 library_meta.json — segments 는 npz 프레임 [start, end)")
AP.add_argument("--cmd_port", type=int, default=PORT_CMD, help="act/ping 명령 UDP 수신 포트")
AP.add_argument("--state_host", default="127.0.0.1", help="상태 {state, seq} 를 보낼 곳(데스크탑)")
AP.add_argument("--state_port", type=int, default=PORT_STATE)
AP.add_argument("--run_sec", type=float, default=0.0,
                help="세그먼트 재생 시간. 0 이면 정지 신호까지 무한. 끝나면 유지(--hold_sec) 단계로 간다")
AP.add_argument("--fake_events", type=int, default=0,
                help="mujoco 백엔드 전용: 6 초 간격으로 무작위 part/bin act 를 N 번 내부 주입하고, "
                     "마지막 동작이 끝나 idle 로 돌아오면 재생을 끝낸다. 요약에 이벤트별 heading 오차·지연 표")
AP.add_argument("--fake_seed", type=int, default=0, help="--fake_events 의 난수 씨앗")
_HERE = os.path.dirname(os.path.abspath(__file__))
AP.add_argument("--meta", default=os.path.join(_HERE, "g1_tracking_policy_meta.json"),
                help="dump_policy_meta.py 로 뽑은 Isaac 실측값. 손으로 쓰지 말 것")
AP.add_argument("--xml", default=os.environ.get("G1_MUJOCO_XML", os.path.expanduser(
                "~/unitree_g1_vibes/RL-shenanigans/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")),
                help="MuJoCo 백엔드(--backend mujoco)에서만 필요한 G1 모델 xml. "
                     "환경변수 G1_MUJOCO_XML 로도 준다. 실기 발행에는 쓰지 않는다 — "
                     "torso 회전은 허리 3관절로 직접 계산한다")
AP.add_argument("--iface", default="auto",
                help="DDS 네트워크 인터페이스. auto 면 로봇 내부망(192.168.123.x) IPv4 를 가진 "
                     "인터페이스를 직접 찾는다 — **이름을 믿으면 안 된다.** 2026-09-20 실측: "
                     "재부팅 후 내부망이 eth0 에서 eth1 로 옮겨 갔고 eth0 은 DOWN 이었다. "
                     "잘못된 인터페이스에 붙으면 lowstate 가 오지 않는다.")
AP.add_argument("--domain", type=int, default=0)
AP.add_argument("--imu_frame", choices=["pelvis", "torso"], default="pelvis",
                help="lowstate.imu_state 가 어느 링크의 것인가. 실기에서 확인 필요: "
                     "waist_yaw 만 돌렸을 때 IMU yaw 가 변하면 torso, 안 변하면 pelvis")
AP.add_argument("--enter_sec", type=float, default=0.0,
                help="참조 첫 자세까지 보간 진입 시간. **0 을 권장한다** — 이 구간은 정책이 꺼져 있고 "
                     "PD 만으로는 이족 로봇의 균형이 유지되지 않아 오히려 넘어진다(실측: 4초 유지 시 "
                     "참조 자세 0.866→0.089 m, 기본 자세 0.866→0.061 m). 정책이 균형을 잡는 주체다.")
AP.add_argument("--hold_sec", type=float, default=300.0,
                help="모션이 끝난 뒤 **정책을 끈 채로 두지 않고** 마지막 프레임을 계속 먹여 "
                     "서 있게 하는 시간. 정책이 꺼지면 무조건 주저앉는다(실측: 어떤 마무리 방식도 "
                     "골반 0.78 → 0.09~0.17 m). 유지 중 토크는 19%% 뿐이라 사실상 무한히 버틴다. "
                     "0 이면 유지하지 않는다. 시간이 만료되면 카운트다운 뒤 관절을 푸는데, "
                     "로봇은 그때 주저앉으므로(골반 0.73 → 0.17 m) 받칠 준비가 돼 있어야 한다.")
AP.add_argument("--exit_sec", type=float, default=3.0, help="종료 시 기본 자세로 복귀 시간(유지를 끄면 사용)")
AP.add_argument("--hold_pose", choices=["default", "last"], default="default",
                help="유지 모드에서 어떤 자세로 서 있을까. last 는 춤의 마지막 프레임을 그대로 "
                     "들고 있어(제로투는 손을 든 자세라 보기 무섭다), default 는 팔을 내린 "
                     "기본 자세로 옮겨 선다. 기본값은 default 다.")
AP.add_argument("--hold_frame", type=int, default=-1,
                help="유지 자세로 쓸 **참조 프레임 번호**. -1 이면 --hold_pose 규칙을 따른다. "
                     "춤 마지막 프레임이 발이 뜬 자세면 그 자세로 정지해 있으라는 명령이 되어 "
                     "넘어진다(파라파라 실측: 춤 21.7초는 완주하는데 유지 진입에서 무너졌다). "
                     "양발이 접지하고 직립·정지에 가까운 프레임을 골라 쓴다.")
AP.add_argument("--settle_sec", type=float, default=1.5,
                help="춤이 끝난 뒤 유지 자세로 옮겨가는 시간. 정책은 그동안 계속 켜져 있다.")
AP.add_argument("--start_pose", choices=["ref", "default", "slack"], default="ref",
                help="mujoco 백엔드에서 시작 자세. 실기는 기본 자세에서 시작하므로 default 로 검증한다. "
                     "slack 은 하네스에 매달려 중력으로 늘어진 자세를 물리로 만든다 "
                     "(공중 고정 + 토크 0 으로 --slack_sec 동안 방치)")
AP.add_argument("--slack_sec", type=float, default=2.0,
                help="start_pose slack 에서 늘어지게 방치하는 시간")
AP.add_argument("--harness", type=float, default=0.0,
                help="하네스가 골반을 위로 당기는 힘의 비율(0~1, 로봇 무게 대비). **지속적으로** "
                     "가한다 — --start_lift 는 들어서 시작만 하고 곧 착지하므로 '하네스를 조인' "
                     "상태를 재현하지 못한다. 1.0 이면 발이 완전히 뜬다.")
AP.add_argument("--sim_push", type=float, default=0.0,
                help="mujoco 전용 '밀어도 버티는가' 시험: idle 중에만 --sim_push_every 초마다 골반을 이 힘(N)으로 "
                     "--sim_push_dur 초 민다. 방향은 앞→왼→뒤→오른 순환. 요약에 자리 이탈 최대(m)를 낸다")
AP.add_argument("--sim_push_every", type=float, default=4.0)
AP.add_argument("--sim_push_dur", type=float, default=0.2)
AP.add_argument("--start_lift", type=float, default=0.0,
                help="mujoco 백엔드에서 골반을 이만큼(m) 들어올려 시작한다 — 하네스에 매달려 "
                     "발이 땅에서 뜬 상태를 재현한다")
AP.add_argument("--backend", choices=["dds", "mujoco"], default="dds",
                help="mujoco 면 DDS 대신 MuJoCo 물리에 붙는다 — 로봇 없이 이 스크립트 전체를 검증한다")
AP.add_argument("--log_csv", default="",
                help="매 스텝의 목표·실제 관절·토크·IMU 를 CSV 로 남긴다. 실기에서 무슨 일이 "
                     "있었는지는 요약만으로는 알 수 없다 — 사후 분석용이다.")
AP.add_argument("--sim_video", default="")
AP.add_argument("--sim_delay", type=int, default=0,
                help="mujoco 전용 강건성 시험: 목표 관절을 이 스텝 수(×20ms)만큼 늦게 적용한다")
AP.add_argument("--sim_kp", type=float, default=1.0,
                help="mujoco 전용 강건성 시험: 모터 P 이득 배율 (실기 모터와의 차이를 흉내)")
AP.add_argument("--viewer", action="store_true",
                help="mujoco 백엔드에서 실시간 뷰어 창을 띄운다 (화면으로 보며 확인)")
AP.add_argument("--arm", action="store_true", help="실제로 DDS 에 발행한다 (없으면 dry-run)")
AP.add_argument("--probe", type=float, default=0.0,
                help="DDS 를 열어 lowstate 만 이 시간(초) 받아보고 끝낸다. **발행하지 않는다.** "
                     "dry-run 은 DDS 를 아예 열지 않으므로 인터페이스가 맞는지·lowstate 가 오는지를 "
                     "검증하지 못한다 — 실기 발행 직전의 마지막 관문이 이것이다.")
AP.add_argument("--max_cycles_late", type=int, default=5, help="주기를 이만큼 연속 놓치면 중단")
AP.add_argument("--hot_window", type=int, default=50,
                help="토크 이동창 길이(스텝). 50 = 1초.")
AP.add_argument("--hot_frac", type=float, default=0.50,
                help="이동창 평균 토크가 한계의 이 비율을 넘으면 중단하고 관절을 푼다. "
                     "문턱은 분포를 재서 정했다 — 1초창 평균 최대가 정상(하네스 지지 0~50%%)에서 "
                     "31~41%%, 넘어지는 90%% 에서 58%%, 완전 매달림에서 74%% 였다. "
                     "0.50 은 정상에서 오탐 0 이고 위험 두 조건을 잡는다(여유 9%%p). "
                     "하네스 지지 70%%(창 47%%)는 통과하는데, 그 조건은 실제로 완주한다.")
AP.add_argument("--hot_grace", type=float, default=1.5,
                help="시작 후 이 시간(초)까지는 토크 가드를 적용하지 않는다 — 첫 1초는 원래 급격하다.")
AP.add_argument("--max_hot_steps", type=int, default=25,
                help="토크가 한계 근처(95%%)에 이만큼 연속 붙어 있으면 중단하고 관절을 푼다. "
                     "25 스텝 = 0.5초. 하네스를 너무 조여 발이 뜨면 정책이 자세를 고칠 수 없어 "
                     "토크 한계까지 다리를 휘두른다(실측: 완전 매달림에서 토크 100%%, "
                     "관절속도 16.8 rad/s — 정상은 64%%, 14.3). 발이 땅을 딛는 정상 실행의 "
                     "최대는 64~84%% 라 95%% 문턱은 오탐하지 않는다.")
args = AP.parse_args()

if args.fake_events and args.backend != "mujoco":
    sys.exit("--fake_events 는 mujoco 백엔드 전용이다 — 실기에 가짜 명령을 넣지 않는다")
if args.sim_push and args.backend != "mujoco":
    sys.exit("--sim_push 는 mujoco 백엔드 전용이다")

# ---------------------------------------------------------------- 메타 / 정책 / 모션

meta = json.load(open(args.meta))
NJ = 29
I_JOINTS = meta["joint_names"]                 # Isaac BFS 순서
KP = np.asarray(meta["joint_stiffness"]); KD = np.asarray(meta["joint_damping"])
QDEF = np.asarray(meta["default_joint_pos"])   # nominal (action_offset 은 reset 랜덤화가 섞여 있어 쓰지 않는다)
ASCALE = np.asarray(meta["action_scale"])
QLIM = np.asarray(meta["joint_pos_limits"])
CTRL_DT = meta["control_dt"]
DEC = int(meta["decimation"])
ANCHOR = meta["anchor_body_name"]

# .npz 면 torch 없이 돈다 (Jetson aarch64 에 torch 를 올리지 않기 위해).
# export_policy_npz.py 로 .pt → .npz 변환하고, 그 스크립트가 torch 경로와 일치를 확인한다.
if args.policy.endswith(".npz"):
    _z = np.load(args.policy)
    W = [_z[f"w{i}"].astype(np.float64) for i in range(4)]
    B = [_z[f"b{i}"].astype(np.float64) for i in range(4)]
    OM = _z["obs_mean"].astype(np.float64); OS = _z["obs_std"].astype(np.float64)
else:
    import torch
    sd = torch.load(args.policy, map_location="cpu", weights_only=False)["actor_state_dict"]
    W = [sd[f"mlp.{i}.weight"].numpy().astype(np.float64) for i in (0, 2, 4, 6)]
    B = [sd[f"mlp.{i}.bias"].numpy().astype(np.float64) for i in (0, 2, 4, 6)]
    OM = sd["obs_normalizer._mean"].numpy().reshape(-1).astype(np.float64)
    OS = sd["obs_normalizer._std"].numpy().reshape(-1).astype(np.float64)
OBS_DIM = OM.shape[0]
if OBS_DIM != 154:
    sys.exit(f"관측 {OBS_DIM}차원 — 이 스크립트는 154차원(Wo-State-Estimation) 전용이다.\n"
             f"160차원 정책은 world 위치·선속도를 요구하므로 실기에 올릴 수 없다.")

def elu(x): return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
def policy(o):
    x = (o - OM) / OS
    for k in range(3):
        x = elu(W[k] @ x + B[k])
    return W[3] @ x + B[3]

M = np.load(args.motion)
REF_Q, REF_QD = M["joint_pos"].copy(), M["joint_vel"].copy()
REF_BQ = M["body_quat_w"].copy()
FPS = int(np.atleast_1d(M["fps"])[0])
T_ALL = REF_Q.shape[0]
if abs(1.0 / CTRL_DT - FPS) > 1e-6:
    sys.exit(f"제어 {1/CTRL_DT} Hz vs 모션 {FPS} fps 불일치")
B_ANCHOR = meta["body_names"].index(ANCHOR)

# 세그먼트 경계. 라이브러리와 meta 가 짝이 맞는지 여기서 확인한다 — 어긋나면 엉뚱한 구간을 먹인다.
LIB = json.load(open(args.library_meta))
if int(LIB.get("fps_npz", FPS)) != FPS:
    sys.exit(f"library_meta fps_npz {LIB.get('fps_npz')} vs 모션 npz {FPS} fps 불일치")
SEGS = {n: tuple(v) for n, v in LIB["segments"].items()}
_bad = [n for n, (s0, e0) in SEGS.items() if not 0 <= s0 < e0 <= T_ALL]
if _bad:
    sys.exit(f"세그먼트 {_bad} 가 라이브러리 npz({T_ALL} 프레임) 밖이거나 비었다 — meta 와 npz 짝 확인")
_missing = [n for n in ["idle"] + [turn_segment(b) for b in TURN_BINS if b]
            + sorted(PART_TO_SEGMENT.values()) if n not in SEGS]
if "idle" in _missing:
    sys.exit("library_meta 에 idle 세그먼트가 없다")
if _missing:
    print(f"[경고] 라이브러리에 없는 세그먼트 {_missing} — 그 명령은 거부한다")
player = SegmentPlayer(SEGS)

# 원격(Jetson)에서 돌 때 콘솔이 비상정지로 이 PID 에 SIGTERM 을 보낸다.
# SIGTERM 핸들러는 즉시 kp=0·kd=2 로 관절을 풀고 나간다.
try:
    with open("/tmp/g1_deploy.pid", "w") as _f:
        _f.write(str(os.getpid()))
except Exception:
    pass

def slerp(qa, qb, t):
    """wxyz 쿼터니언 구면 보간. 반구를 맞춰 최단 경로로 간다."""
    a = np.asarray(qa, float); b = np.asarray(qb, float)
    d = float(a @ b)
    if d < 0.0:
        b = -b; d = -d
    if d > 0.9995:                      # 거의 같으면 선형 보간 후 정규화
        r = a + t * (b - a)
        return r / np.linalg.norm(r)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)

# ---------------------------------------------------------------- FK (torso 회전 전용)
# MuJoCo 를 쓰지 않는다. Jetson 은 python 3.8 이라 mujoco 3.x 설치가 막히고,
# 어차피 필요한 건 pelvis → torso 회전 하나뿐이다.
# URDF 확인 결과 체인이 단순하다 — origin 회전이 전부 0 이고 축이 정확히 z, x, y 다:
#   pelvis --[waist_yaw  axis (0,0,1)]--> --[waist_roll axis (1,0,0)]--> --[waist_pitch axis (0,1,0)]--> torso
# 따라서  R_torso = R_pelvis · Rz(yaw) · Rx(roll) · Ry(pitch)
# (origin 의 xyz 는 위치라서 회전에 영향이 없다)
_IW = [I_JOINTS.index(n) for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]

def _rz(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

def _rx(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])

def _ry(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])

def waist_chain(q_isaac):
    """pelvis 기준 torso 의 상대 회전. 허리 3관절만 쓴다."""
    y, r, pi = (float(q_isaac[i]) for i in _IW)
    return _rz(y) @ _rx(r) @ _ry(pi)

# SDK 모터 인덱스 (URDF 순서) ↔ Isaac BFS 순서
SDK_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
assert sorted(SDK_ORDER) == sorted(I_JOINTS), "관절 이름 집합 불일치"
I2S = np.array([SDK_ORDER.index(n) for n in I_JOINTS])   # Isaac i → SDK 모터 번호
S2I = np.array([I_JOINTS.index(n) for n in SDK_ORDER])   # SDK i → Isaac 인덱스

# 두 배열은 서로 다른 순열이다(27/29 자리가 다르다). 어느 쪽을 쓸지 손으로 고르면
# 반드시 틀린다 — 실제로 틀렸고, 실기에서 로봇이 엉뚱한 관절을 휘둘렀다
# (왼쪽 무릎 모터에 오른쪽 hip_pitch 값이 들어갔다).
# 그래서 방향을 함수 이름에 박고, 이름으로 자체 검증한다.
#   x_sdk[j]   = x_isaac[S2I[j]]     → 배열 색인은 x_isaac[S2I]
#   x_isaac[i] = x_sdk[I2S[i]]       → 배열 색인은 x_sdk[I2S]
def isaac_to_sdk(x):
    """Isaac(BFS) 순서 배열 → SDK 모터 순서 배열."""
    return np.asarray(x)[S2I]

def sdk_to_isaac(x):
    """SDK 모터 순서 배열 → Isaac(BFS) 순서 배열."""
    return np.asarray(x)[I2S]

# 왕복 검사만으로는 부족하다 — 두 방향을 바꿔 써도 왕복은 성립한다(서로 역순열이므로).
# **이름**으로 검증해야 한다.
_probe = np.arange(NJ)
_sdk = isaac_to_sdk(_probe)
for _j in range(NJ):
    assert I_JOINTS[int(_sdk[_j])] == SDK_ORDER[_j], (
        f"isaac_to_sdk 가 틀렸다: SDK 모터 {_j}({SDK_ORDER[_j]}) 자리에 "
        f"{I_JOINTS[int(_sdk[_j])]} 가 들어간다")
_isc = sdk_to_isaac(np.arange(NJ))
for _i in range(NJ):
    assert SDK_ORDER[int(_isc[_i])] == I_JOINTS[_i], (
        f"sdk_to_isaac 가 틀렸다: Isaac {_i}({I_JOINTS[_i]}) 자리에 "
        f"{SDK_ORDER[int(_isc[_i])]} 가 들어간다")

def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def yaw_of_mat(R):
    return np.arctan2(R[1, 0], R[0, 0])

def quat_from_yaw(a):
    return np.array([np.cos(a/2), 0.0, 0.0, np.sin(a/2)])

def quat_mul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def torso_rot_world(imu_quat, q_isaac):
    """IMU 자세 + 관절각으로 torso 의 world 회전을 구한다.
    imu_frame=torso 면 IMU 값이 곧 torso 회전이므로 체인을 건너뛴다.
    (실기 실측으로 IMU 는 pelvis 로 확정됐다 — 허리를 돌릴 때 IMU yaw 가 반작용으로
     반대 방향으로 2.26° 움직였고 상관이 −0.474 였다. torso 라면 같은 방향 3.4° 였을 것)"""
    if args.imu_frame == "torso":
        return quat_to_mat(imu_quat)
    return quat_to_mat(imu_quat) @ waist_chain(q_isaac)

# ---------------------------------------------------------------- DDS
def find_robot_iface(prefix="192.168.123."):
    """로봇 내부망 IPv4 를 가진 인터페이스 이름. 없으면 None.
    인터페이스 이름은 재부팅 때 바뀔 수 있으므로 주소로 찾는다."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True,
                             text=True, timeout=5).stdout
        for ln in out.splitlines():
            f = ln.split()
            if len(f) >= 4 and f[2] == "inet" and f[3].startswith(prefix):
                return f[1]
    except Exception:
        pass
    return None


def resolve_iface():
    """--iface 를 실제 이름으로 바꾼다."""
    if args.iface != "auto":
        return args.iface
    nm = find_robot_iface()
    if nm is None:
        sys.exit("로봇 내부망(192.168.123.x) 인터페이스를 찾지 못했다. "
                 "`ip -4 -o addr` 로 확인하고 --iface 로 직접 지정하라.")
    return nm


class Bridge:
    """rt/lowcmd 발행 + rt/lowstate 구독. dry-run 이면 아무것도 발행하지 않는다."""
    def __init__(self):
        self.ok = False
        self.state = None
        self.mode_machine = None
        self.n_state = 0
        self.t_state = []
        self.probe_only = (not args.arm) and args.probe > 0
        if not args.arm and not self.probe_only:
            print("[dry-run] DDS 를 열지 않는다. 관측·명령을 계산해 로그만 찍는다.")
            return
        # SDK 위치는 기기마다 다르다 — 데스크탑과 Jetson 이 서로 다른 경로에 두고 있다.
        # 경로를 하나로 박아두면 실기 발행에서만 터지고 dry-run 은 이 줄을 타지 않아 잡히지 않는다.
        for _cand in ("/home/unitree/unitree_sdk2_python",
                      "/home/junhyun/unitree_g1_vibes/unitree_sdk2_python",
                      os.path.expanduser("~/unitree_sdk2_python")):
            if os.path.isdir(_cand) and _cand not in sys.path:
                sys.path.insert(0, _cand)
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        except ImportError as e:
            sys.exit(f"unitree_sdk2py 를 찾지 못했다: {e}\n"
                     f"찾아본 경로: /home/unitree/unitree_sdk2_python, "
                     f"/home/junhyun/unitree_g1_vibes/unitree_sdk2_python, ~/unitree_sdk2_python")
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        iface = resolve_iface()
        if iface != args.iface:
            print(f"내부망 인터페이스 자동 탐색: {iface} "
                  f"(이름은 재부팅 때 바뀔 수 있어 주소로 찾는다)")
        ChannelFactoryInitialize(args.domain, iface)
        self.sub = ChannelSubscriber("rt/lowstate", LowState_); self.sub.Init(self._on_state, 10)
        if self.probe_only:
            # 수신만 한다. 발행자를 아예 만들지 않아 실수로 쏠 수 없게 한다.
            print("[probe] lowstate 수신만 한다 — 발행자를 만들지 않았다.")
            self.ok = True
            return
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_); self.pub.Init()
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.ok = True

    def _on_state(self, msg):
        self.state = msg
        self.mode_machine = msg.mode_machine
        self.n_state += 1
        if args.probe > 0:
            self.t_state.append(time.perf_counter())

    def send(self, q_sdk, kp_sdk, kd_sdk):
        """kp=0, kd=작은값으로 부르면 댐핑이 된다 (비상정지 경로에서 그렇게 쓴다)."""
        if not self.ok:
            return
        self.cmd.mode_pr = 0          # Mode.PR — 발목을 pitch/roll 관절로 명령한다.
                                      # 정책이 그 공간에서 학습했으므로 AB 모드면 발목이 뒤바뀐다.
        self.cmd.mode_machine = self.mode_machine
        for i in range(NJ):
            mc = self.cmd.motor_cmd[i]
            mc.mode = 1
            mc.q = float(q_sdk[i]); mc.dq = 0.0; mc.tau = 0.0
            mc.kp = float(kp_sdk[i]); mc.kd = float(kd_sdk[i])
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

class MujocoBackend:
    # 이 클래스만 mujoco 를 쓴다. 실기 경로(--backend dds)에서는 import 하지 않는다.
    """DDS 대신 MuJoCo 를 물리 로봇 자리에 놓는다. 관측 조립·재정렬·PD 는 실기와 같은 코드를 탄다."""
    def __init__(self):
        global mj
        import mujoco as mj
        if not args.xml:
            sys.exit("MuJoCo 백엔드에는 --xml (또는 환경변수 G1_MUJOCO_XML)로 G1 모델이 필요하다.\n"
                     "바닥이 있는 scene 쪽을 쓴다 — g1_29dof.xml 에는 plane 이 없어 로봇이 낙하한다.")
        scene = args.xml.replace("g1_29dof.xml", "scene_29dof.xml")
        self.m = mj.MjModel.from_xml_path(scene)
        self.m.opt.timestep = 0.005
        self.m.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
        self.d = mj.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, n)] for n in I_JOINTS])
        self.vadr = np.array([self.m.jnt_dofadr[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, n)] for n in I_JOINTS])
        for i in range(NJ):
            self.m.dof_armature[self.vadr[i]] = meta["joint_armature"][i]
            self.m.dof_damping[self.vadr[i]] = KD[i]      # D 항은 MuJoCo 가 암시적으로 적분한다
            self.m.dof_frictionloss[self.vadr[i]] = 0.0
        self.tau_last = np.zeros(NJ)
        self.m_total = float(self.m.body_mass.sum())
        self.b_feet = [mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, n)
                       for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.act = np.full(NJ, -1)
        for a in range(self.m.nu):
            nm = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_JOINT, self.m.actuator_trnid[a, 0])
            if nm in I_JOINTS:
                self.act[I_JOINTS.index(nm)] = a
        self.b_pel = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "pelvis")
        # 참조 프레임 0 자세로 시작 (sim2sim 과 동일 조건)
        bp, bq = M["body_pos_w"], M["body_quat_w"]
        pel = meta["body_names"].index("pelvis")
        self.d.qpos[:3] = bp[0, pel]; self.d.qpos[3:7] = bq[0, pel]
        self.d.qpos[self.qadr] = REF_Q[0] if args.start_pose == "ref" else QDEF.copy()
        self.d.qvel[:] = 0.0
        mj.mj_forward(self.m, self.d)
        if args.start_pose == "slack":
            # 하네스에 매달린 상태: 골반을 공중에 고정하고 토크 0 으로 두면 중력이 관절을 늘어뜨린다.
            pel_hold = self.d.qpos[:7].copy()
            pel_hold[2] += 0.30                      # 발이 확실히 뜨도록 들어올린다
            # 위치만 되돌리면 MuJoCo 는 로봇이 강체로 자유낙하한다고 계산해 관절에 중력
            # 토크가 걸리지 않는다(실측: qacc 가 골반 z 만 -9.81, 관절 전부 0).
            # 하네스는 힘이므로 free joint 의 bias 를 상쇄해 준다.
            for _ in range(int(args.slack_sec / self.m.opt.timestep)):
                self.d.qpos[:7] = pel_hold           # 하네스가 골반을 붙잡는다
                self.d.qvel[:6] = 0.0
                self.d.ctrl[:] = 0.0                 # 모터 무력 = 늘어짐
                mj.mj_forward(self.m, self.d)
                self.d.qfrc_applied[:6] = self.d.qfrc_bias[:6]   # 하네스 지지력
                mj.mj_step(self.m, self.d)
            self.d.qfrc_applied[:] = 0.0
            q_slack = self.d.qpos[self.qadr].copy()
            print(f"늘어진 자세 생성: 기본 자세와 평균 "
                  f"{np.degrees(np.abs(q_slack - QDEF).mean()):.1f}° "
                  f"최대 {np.degrees(np.abs(q_slack - QDEF).max()):.1f}° 차이")
            # 다시 착지 높이로 되돌린다 (자세는 늘어진 채로 유지)
            self.d.qpos[:7] = pel_hold; self.d.qpos[2] -= 0.30
            self.d.qvel[:] = 0.0
            mj.mj_forward(self.m, self.d)
        if args.start_lift != 0.0:
            self.d.qpos[2] += args.start_lift
            self.d.qvel[:] = 0.0
            mj.mj_forward(self.m, self.d)
            print(f"골반 {args.start_lift*100:.0f} cm 들어올려 시작 "
                  f"(발이 뜬 상태 = 하네스에 매달림)")
        self.viewer = None
        if args.viewer:
            import mujoco.viewer as mjv
            self.viewer = mjv.launch_passive(self.m, self.d, show_left_ui=False,
                                             show_right_ui=False)
            self.viewer.cam.distance, self.viewer.cam.elevation = 2.2, -10
            self.viewer.cam.azimuth = 135
        self.renderer = None
        if args.sim_video:
            self.m.vis.global_.offwidth, self.m.vis.global_.offheight = 960, 720
            self.renderer = mj.Renderer(self.m, 720, 960)
            self.cam = mj.MjvCamera(); mj.mjv_defaultCamera(self.cam)
            self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.9, -8, 135
            # 쌓지 않고 흘려보낸다. 3분 유지면 9000프레임이고, 그걸 메모리에 들면
            # 11.5GB 가 된다(실측). 2프레임마다 한 장씩 기록해 절반 fps 로 남긴다.
            import imageio
            self.vw = imageio.get_writer(args.sim_video, fps=max(1, FPS // 2),
                                         codec="libx264", quality=7,
                                         macro_block_size=None)
            self.nrender = 0

    def foot_force(self):
        """양 발에 걸린 수직 접촉력 합 [N]. 0 이면 발이 떠 있다."""
        tot = 0.0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = self.m.geom_bodyid[c.geom1]; b2 = self.m.geom_bodyid[c.geom2]
            if b1 in self.b_feet or b2 in self.b_feet:
                f = np.zeros(6); mj.mj_contactForce(self.m, self.d, i, f)
                tot += abs(float(f[0]))
        return tot

    def read(self):
        q = self.d.qpos[self.qadr].copy(); dq = self.d.qvel[self.vadr].copy()
        quat = np.empty(4); mj.mju_mat2Quat(quat, self.d.xmat[self.b_pel])   # pelvis IMU 를 흉내
        gyro = self.d.qvel[3:6].copy()                                        # free joint 각속도는 body local
        return q, dq, quat, gyro

    def apply(self, q_target_isaac):
        # 하네스는 위치가 아니라 힘이다. 골반 z 에 무게의 일정 비율을 계속 걸어
        # "너무 조여 매달린" 상태를 재현한다.
        if args.harness > 0:
            self.d.qfrc_applied[2] = args.harness * self.m_total * 9.81
        self.q_hist = getattr(self, "q_hist", []) + [q_target_isaac]
        q_target_isaac = self.q_hist[max(0, len(self.q_hist) - 1 - args.sim_delay)]
        self.q_hist = self.q_hist[-(args.sim_delay + 1):]
        for _ in range(DEC):
            if args.sim_push > 0:
                self._push_step()
            # 실기 모터의 토크 한계를 그대로 건다 — 없으면 실기보다 관대한 조건이 된다
            tau = np.clip(args.sim_kp * KP * (q_target_isaac - self.d.qpos[self.qadr]), -ELIM_A, ELIM_A)
            self.tau_last = tau
            self.d.ctrl[self.act] = tau
            mj.mj_step(self.m, self.d)
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt("뷰어 창이 닫혔다")
            self.viewer.cam.lookat[:] = self.d.qpos[:3]
            self.viewer.cam.lookat[2] = 0.75
            self.viewer.sync()
            time.sleep(0.02)                 # 실시간 속도로 보여준다 (50 Hz)
        if self.renderer is not None:
            self.cam.lookat[:] = self.d.qpos[:3]; self.cam.lookat[2] = 0.75
            self.nrender += 1
            if self.nrender % 2 == 0:          # 2프레임마다 한 장
                self.renderer.update_scene(self.d, self.cam)
                self.vw.append_data(self.renderer.render())

    def height(self):
        return float(self.d.qpos[2])

    push_idle = False      # 메인 루프가 매 스텝 갱신 — idle 일 때만 민다
    _DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def _push_step(self):
        """--sim_push: 골반에 수평 외력. 자리 이탈 = 첫 위치에서 골반 xy 최대 거리."""
        xy = self.d.qpos[:2].copy()
        if not hasattr(self, "xy0"):
            self.xy0, self.max_drift, self.n_push, self._was = xy, 0.0, 0, False
        self.max_drift = max(self.max_drift, float(np.linalg.norm(xy - self.xy0)))
        t = self.d.time
        on = self.push_idle and t > args.sim_push_every and (t % args.sim_push_every) < args.sim_push_dur
        if on and not self._was:
            self.n_push += 1
        self._was = on
        dx, dy = self._DIRS[int(t // args.sim_push_every) % 4]
        self.d.xfrc_applied[self.b_pel, :3] = (args.sim_push * dx, args.sim_push * dy, 0.0) if on else (0.0, 0.0, 0.0)

# 명령 수신(비차단) / 상태 송신. DDS·MuJoCo 를 열기 전에 만든다 — 포트가 막혔으면 로봇을 건드리기 전에 끝낸다.
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    cmd_sock.bind(("0.0.0.0", args.cmd_port))
except OSError as e:
    sys.exit(f"명령 포트 {args.cmd_port} 를 열지 못했다: {e} (다른 서버가 떠 있지 않은지 확인)")
cmd_sock.setblocking(False)
state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sim = MujocoBackend() if args.backend == "mujoco" else None
bridge = Bridge() if args.backend == "dds" else type("N", (), {"ok": False, "state": None, "mode_machine": 0})()
KP_SDK = isaac_to_sdk(KP); KD_SDK = isaac_to_sdk(KD)
_EFF = {"hip_yaw":88.,"hip_roll":139.,"hip_pitch":88.,"knee":139.,"ankle_pitch":50.,"ankle_roll":50.,
        "waist_yaw":88.,"waist_roll":50.,"waist_pitch":50.,"shoulder_pitch":25.,"shoulder_roll":25.,
        "shoulder_yaw":25.,"elbow":25.,"wrist_roll":25.,"wrist_pitch":5.,"wrist_yaw":5.}
ELIM_A = np.array([next(v for k, v in sorted(_EFF.items(), key=lambda kv: -len(kv[0])) if k in n)
                   for n in I_JOINTS])
QLO = QLIM[:, 0]; QHI = QLIM[:, 1]

def send_damp(frames=150, kd=2.0):  # noqa: C901
    """관절을 부드럽게 풀어준다 (kp=0, 약한 kd). 비상정지의 실질적 안전 조치다.
    SDK 에 lowcmd 워치독이 없어서 '발행을 멈추는 것' 만으로는 로봇이 어떻게 되는지 알 수 없다
    (펌웨어 의존, 미확인). 그래서 끊기 전에 능동적으로 토크를 빼고 나간다."""
    if sim is not None:
        # MuJoCo 백엔드에서도 실제로 풀어본다 — 종료 거동을 시뮬로 확인할 수 있게
        for _ in range(frames * DEC):
            sim.d.ctrl[sim.act] = -kd * sim.d.qvel[sim.vadr]
            mj.mj_step(sim.m, sim.d)
        print(f"[댐핑] MuJoCo 에서 kd={kd} 로 풀었다 → 골반 {sim.height():.3f} m")
        return
    if not bridge.ok:
        print(f"[댐핑] dry-run — 발행하지 않음 (kp=0, kd={kd}, {frames}프레임)")
        return
    z = np.zeros(NJ)
    for _ in range(frames):
        bridge.send(isaac_to_sdk(z), isaac_to_sdk(z), isaac_to_sdk(np.full(NJ, kd)))
        time.sleep(CTRL_DT)
    print(f"[댐핑] kp=0 kd={kd} 로 {frames}프레임 발행했다")


stop = {"v": False, "hard": False}
def _sig_int(*_):
    stop["v"] = True
    print("\n[중단] 기본 자세로 복귀한다.")
def _sig_term(*_):
    """비상정지. 복귀 보간을 건너뛰고 즉시 댐핑으로 풀고 나간다."""
    stop["v"] = True
    stop["hard"] = True
    print("\n[비상] 즉시 댐핑으로 풀고 종료한다.")
signal.signal(signal.SIGINT, _sig_int)
signal.signal(signal.SIGTERM, _sig_term)

def read_state():
    """(Isaac 순서 관절각, 관절속도, IMU 쿼터니언, 자이로)"""
    if sim is not None:
        return sim.read()
    if not bridge.ok:
        # dry-run: 첫 호출은 기본 자세, 이후는 직전 명령이 그대로 실현됐다고 가정
        q = read_state.fake_q.copy()
        return q, np.zeros(NJ), np.array([1.0, 0, 0, 0]), np.zeros(3)
    st = bridge.state
    if st is None:
        return None
    q_sdk = np.array([st.motor_state[i].q for i in range(NJ)])
    d_sdk = np.array([st.motor_state[i].dq for i in range(NJ)])
    imu = np.array(st.imu_state.quaternion, dtype=np.float64)   # wxyz
    gyro = np.array(st.imu_state.gyroscope, dtype=np.float64)
    return sdk_to_isaac(q_sdk), sdk_to_isaac(d_sdk), imu, gyro
read_state.fake_q = QDEF.copy()

def torque_now():
    """모터 토크 (Isaac 순서). 시뮬은 실제 낸 값, 실기는 lowstate 추정값. 없으면 None."""
    if sim is not None:
        return sim.tau_last.copy()
    if not bridge.ok or bridge.state is None:
        return None
    return sdk_to_isaac([bridge.state.motor_state[i].tau_est for i in range(NJ)])

def publish_target(q_isaac_target):
    q = np.clip(q_isaac_target, QLO, QHI)
    if sim is not None:
        sim.apply(q)
        return q
    bridge.send(isaac_to_sdk(q), KP_SDK, KD_SDK)   # ← 여기가 q[I2S] 였다 (버그)
    if not bridge.ok:
        read_state.fake_q = q.copy()
    return q

# ---------------------------------------------------------------- 1) 상태 수신 대기
if bridge.ok:
    print("lowstate 수신 대기...")
    t0 = time.time()
    while bridge.state is None:
        if time.time() - t0 > 5.0:
            sys.exit("lowstate 가 오지 않는다. 인터페이스/도메인, 그리고 고수준 컨트롤러가 "
                     "내려갔는지(MotionSwitcher.ReleaseMode) 확인하라.")
        time.sleep(0.02)
    print(f"  mode_machine = {bridge.mode_machine}, mode_pr = {bridge.state.mode_pr}")
    # 고수준(loco)이 살아 있으면 lowcmd 와 둘이 같은 모터에 명령을 낸다. 2026-10-05 실기에서
    # 관절이 목표를 무시하고 고정된 run(실토크/기대토크 0.00)과 0.4~0.6 인 run 이 나왔다.
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        _msc = MotionSwitcherClient(); _msc.SetTimeout(3.0); _msc.Init()
        _res = _msc.CheckMode()[1]
        _mode = _res.get("name", "") if isinstance(_res, dict) else "?"
    except Exception as e:
        _mode = f"?({type(e).__name__})"
    print(f"  고수준 모드 = {_mode!r} ({'R&D — 발행 가능' if _mode == '' else '살아 있음'})")
    if args.arm and _mode != "":
        sys.exit("[거부] 고수준 컨트롤러가 살아 있다(또는 확인 불가). 'R&D 진입' 후 다시 발행하라 — "
                 "그대로 쏘면 두 컨트롤러가 싸워 정책이 발산한다.")

if args.probe > 0:
    # 발행 없이 수신만 — 인터페이스가 맞는지, lowstate 가 규칙적으로 오는지,
    # 로봇이 어떤 모드·자세인지를 확인하는 마지막 관문.
    print(f"\n[probe] {args.probe:.0f}초 동안 lowstate 만 받는다 (발행하지 않는다)")
    bridge.t_state.clear()
    t0 = time.time()
    while time.time() - t0 < args.probe:
        time.sleep(0.05)
    ts = bridge.t_state[:]
    n = len(ts)
    print(f"  수신 {n}개 / {args.probe:.0f}s = {n/args.probe:.0f} Hz")
    if n > 20:
        gaps = sorted((ts[i+1] - ts[i]) * 1000.0 for i in range(n - 1))
        p50, p99 = gaps[len(gaps)//2], gaps[int(len(gaps)*0.99)]
        print(f"  수신 간격 중앙 {p50:.2f}  p99 {p99:.2f}  최대 {gaps[-1]:.2f} ms "
              f"({'여유 있음' if p99 < 60 else '⚠ 지연 여유 60ms 초과'})")
    r = read_state()
    if r is None:
        sys.exit("[probe] 상태를 읽지 못했다.")
    q, dq, imu, gyro = r
    tq = torque_now()
    print(f"  현재 자세: 기본 자세와 평균 {np.degrees(np.abs(q - QDEF).mean()):.1f}° "
          f"최대 {np.degrees(np.abs(q - QDEF).max()):.1f}° 차이")
    print(f"  참조 첫 자세와는 평균 {np.degrees(np.abs(REF_Q[0] - q).mean()):.1f}° "
          f"최대 {np.degrees(np.abs(REF_Q[0] - q).max()):.1f}°")
    print(f"  관절속도 최대 {np.abs(dq).max():.3f} rad/s")
    if tq is not None:
        print(f"  추정 토크 합 {np.abs(tq).sum():.2f} Nm "
              f"({'늘어진 상태' if np.abs(tq).sum() < 20 else '힘이 들어가 있다'})")
    print(f"  IMU quat {np.round(imu, 3)}  gyro {np.round(gyro, 3)}")
    print(f"  mode_machine {bridge.mode_machine}  mode_pr {bridge.state.mode_pr} "
          f"({'PR — 우리 정책이 학습한 발목 공간' if bridge.state.mode_pr == 0 else '⚠ AB'})")
    ok = (n / args.probe > 100) and (bridge.state.mode_pr == 0)
    print(f"\n[probe] {'통과 — 발행 준비됨' if ok else '⚠ 확인 필요'}. 발행하지 않고 끝낸다.")
    sys.exit(0 if ok else 1)

def motor_health():
    """실기 모터 상태 한 줄. 최고 온도·최저 전압·에러 플래그(motorstate≠0). 연속 시도 발열·
    배터리 저하·모터 보호가 걸리면 실토크가 줄어 정책이 발산한다 — 시뮬로는 안 보이는 원인들."""
    if not bridge.ok or bridge.state is None:
        return None
    ms = bridge.state.motor_state
    t = [max(ms[i].temperature) for i in range(NJ)]; v = [ms[i].vol for i in range(NJ)]
    nm = SDK_ORDER
    err = [f"{nm[i]}={ms[i].motorstate:#x}" for i in range(NJ) if ms[i].motorstate]
    it = int(np.argmax(t))
    return (f"모터 최고 {t[it]}°C({nm[it]}) · 최저 전압 {min(v):.1f} V · 에러 "
            + (", ".join(err) if err else "없음"))
if motor_health():
    print(motor_health())

r = read_state()
q_now = r[0].copy()
print(f"\n현재 자세와 참조 첫 자세의 차이: 평균 {np.degrees(np.abs(REF_Q[0]-q_now).mean()):.1f}° "
      f"최대 {np.degrees(np.abs(REF_Q[0]-q_now).max()):.1f}°")

# ---------------------------------------------------------------- 2) 보간 진입
n_enter = int(args.enter_sec / CTRL_DT)
if n_enter > 0:
    print(f"[경고] 보간 진입 {args.enter_sec:.1f}s — 이 구간은 정책이 꺼져 있어 균형이 유지되지 않는다. "
          f"실측상 1초만 넘어도 넘어진다. 0 을 권장한다.")
    print(f"참조 첫 자세까지 보간 진입 ({n_enter} step)")
else:
    print("보간 진입 없음 — 정책이 처음부터 균형을 잡는다 (권장)")
q_start = q_now.copy()
for k in range(n_enter):
    if stop["v"]:
        break
    a = 0.5 - 0.5*np.cos(np.pi * (k+1)/n_enter)        # cosine ease-in-out
    publish_target(q_start + a * (REF_Q[0] - q_start))
    if sim is None:
        time.sleep(CTRL_DT)

# ---------------------------------------------------------------- 3) 정책 실행
print(f"정책 실행 — 라이브러리 {T_ALL} 프레임 / 세그먼트 {len(SEGS)}개, "
      f"{'정지 신호까지' if args.run_sec <= 0 else f'{args.run_sec:.0f}s'} 재생. "
      f"명령 UDP :{args.cmd_port}, 상태 -> {args.state_host}:{args.state_port}")
last_action = np.zeros(NJ)
yaw_off = None
late = 0
hot = 0              # 토크가 한계 근처에 연속으로 붙어 있는 스텝 수
csvf = None
if args.log_csv:
    csvf = open(args.log_csv, "w", buffering=1)
    csvf.write("phase,t,step," +
               ",".join(f"tgt_{n}" for n in I_JOINTS) + "," +
               ",".join(f"act_{n}" for n in I_JOINTS) + "," +
               ",".join(f"tau_{n}" for n in I_JOINTS) + "," +
               "imu_w,imu_x,imu_y,imu_z,gyro_x,gyro_y,gyro_z\n")

def csv_row(phase, t_s, k, tgt, act, tau, imu, gyro):
    if csvf is None:
        return
    def j(a):
        return ",".join(f"{float(v):.5f}" for v in a)
    csvf.write(f"{phase},{t_s:.3f},{k}," + j(tgt) + "," + j(act) + "," +
               j(tau if tau is not None else np.zeros(NJ)) + "," +
               j(imu) + "," + j(gyro) + "\n")
log = []
tq_log = []          # 스텝별 최대 토크 사용률 (전류 한계 대비)
tq_hold = []         # 유지·쪼그림 구간만
ff_log = []          # 발 수직 접촉력 [N] — 0 이면 떠 있다
cycles = []          # 실제 제어 주기 (ms) — 지연 여유 60ms 안에 드는지 확인용
t_prev = None
t_next = time.perf_counter()
n_run = int(args.run_sec / CTRL_DT) if args.run_sec > 0 else None
events = []          # 수락한 act 마다 heading 오차·지연 기록 (요약 표)
fake_rng = np.random.default_rng(args.fake_seed)
fake_parts = sorted(PART_TO_SEGMENT)
fake_next = int(6.0 / CTRL_DT)       # 다음 가짜 이벤트를 넣을 수 있는 가장 이른 스텝
k = 0
def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0
for t in itertools.count():
    if stop["v"] or (n_run is not None and t >= n_run):
        break
    if args.fake_events and len(events) >= args.fake_events and player.state == "idle":
        print(f"가짜 이벤트 {len(events)}개 완료 — 재생을 끝낸다")
        break
    r = read_state()
    if r is None:
        print("[중단] lowstate 끊김"); break
    q, dq, imu, gyro = r
    yaw_now = float(np.degrees(yaw_of_mat(torso_rot_world(imu, q))))   # 정렬 전 실제 heading

    # 명령: 쌓인 패킷을 전부 비운다. 끊겨도 아무 일 없음 = idle 계속(댐핑하지 않는다).
    msgs = []
    while True:
        try:
            msgs.append(decode(cmd_sock.recvfrom(4096)[0]))
        except OSError:              # BlockingIOError = 비었다. 그 밖의 소켓 오류도 제어를 멈추지 않는다
            break
    if args.fake_events and len(events) < args.fake_events and t >= fake_next and player.state == "idle":
        msgs.append({"seq": player.last_seq + 1, "cmd": "act",
                     "part": str(fake_rng.choice(fake_parts)), "turn_bin": int(fake_rng.choice(TURN_BINS))})
        fake_next = t + int(6.0 / CTRL_DT)
    for m in msgs:
        if player.command(m):
            events.append({"part": m["part"], "bin": m["turn_bin"], "t_acc": t,
                           "target": wrap_deg(yaw_now + m["turn_bin"]), "lat_ms": None, "err": None})
            print(f"  [act] seq {m['seq']} {m['part']} bin {m['turn_bin']:+d}")

    k, first = player.step()
    if sim is not None:
        sim.push_idle = player.state == "idle"
    # 참조 모션의 world 프레임과 로봇 yaw 를 **세그먼트 첫 프레임마다** 다시 정렬한다
    # (원본은 시작 시 1회). 회전 클립이 끝난 뒤의 heading 잔차를 다음 세그먼트로 끌고 가지 않는다.
    # IMU 는 pelvis 자세인데 anchor 는 torso 다 — 두 링크의 yaw 는 waist_yaw 만큼 다르므로
    # 반드시 FK 로 torso 를 구한 뒤 torso 끼리 비교해야 한다(이걸 틀려서 로봇이 넘어졌다).
    if first:
        yaw_off = yaw_of(REF_BQ[k, B_ANCHOR]) - yaw_of_mat(torso_rot_world(imu, q))
        ev = events[-1] if events else None
        if ev is not None and ev["lat_ms"] is None and player.segment != "idle":
            ev["lat_ms"] = (t - ev["t_acc"]) * CTRL_DT * 1000.0
        if ev is not None and ev["err"] is None and player.segment == PART_TO_SEGMENT[ev["part"]]:
            ev["err"] = wrap_deg(yaw_now - ev["target"])     # 회전이 끝나고 동작을 시작하는 순간
    imu_aligned = quat_mul(quat_from_yaw(yaw_off), imu)

    R_rob = torso_rot_world(imu_aligned, q)
    ori_b = (R_rob.T @ quat_to_mat(REF_BQ[k, B_ANCHOR]))[:, :2].reshape(-1)

    obs = np.concatenate([REF_Q[k], REF_QD[k], ori_b, gyro, q - QDEF, dq, last_action])
    assert obs.shape[0] == 154, obs.shape

    action = policy(obs)
    last_action = action.copy()
    target = QDEF + ASCALE * action
    sent = publish_target(target)
    _tq = torque_now()
    log.append((t/FPS, float(np.abs(sent - q).max()), float(np.abs(dq).max())))
    tq_log.append(float(np.abs(_tq / ELIM_A).max()) if _tq is not None else 0.0)
    csv_row(player.segment, t/FPS, k, sent, q, _tq, imu, gyro)
    if t % 10 == 0:                  # 5 Hz 상태 보고. 데스크탑이 없어도 제어는 계속한다
        try:
            state_sock.sendto(encode({"state": player.state, "seq": player.last_seq}),
                              (args.state_host, args.state_port))
        except OSError:
            pass
    # 토크가 한계에 계속 붙어 있으면 물리적으로 뭔가 막고 있다 —
    # 하네스 과조임이 1순위다. 정책은 발 접촉을 관측하지 않으므로 스스로 못 알아챈다.
    # 순간 최대가 아니라 **이동창 평균**을 본다. 매달려 발버둥칠 때의 특징은
    # 순간 첨두가 아니라 토크가 계속 높게 유지되는 것이다(중앙 17% → 39%, 실측).
    if len(tq_log) >= args.hot_window:
        w = float(np.mean(tq_log[-args.hot_window:]))
        if w > args.hot_frac and t * CTRL_DT > args.hot_grace:
            print(f"[중단] 최근 {args.hot_window*CTRL_DT:.1f}초 평균 토크가 "
                  f"한계의 {w*100:.0f}% 다 (문턱 {args.hot_frac*100:.0f}%). 관절을 푼다.\n"
                  f"        하네스를 너무 조여 발이 떠 있지 않은지 확인하라 — 정책은 발 접촉을 "
                  f"관측하지 않아 스스로 알아채지 못한다. 발이 로봇 무게의 절반 이상을 받아야 한다.")
            stop["hard"] = True
            break
    _now = time.perf_counter()
    if t_prev is not None:
        cycles.append((_now - t_prev) * 1000.0)
    t_prev = _now

    if sim is not None:
        log[-1] = log[-1] + (sim.height(),)
        ff_log.append(sim.foot_force())
        continue
    t_next += CTRL_DT
    slack = t_next - time.perf_counter()
    if slack > 0:
        time.sleep(slack)
    else:
        late += 1
        t_next = time.perf_counter()
        if late >= args.max_cycles_late:
            print(f"[중단] 제어 주기를 {late}회 연속 놓쳤다. 지연 여유가 20 ms 뿐이므로 계속하지 않는다.")
            break
    if late and slack > 0:
        late = 0

# ---------------------------------------------------------------- 4) 복귀
if stop["hard"]:
    send_damp()
    print("[비상] 종료")
    sys.exit(0)

def print_summary():
    """어느 종료 경로(유지·복귀·비상)에서도 같은 요약을 찍는다."""
    if motor_health():
        print("  끝 " + motor_health())
    # dry-run 은 물리가 없다. read_state 가 직전 명령을 그대로 돌려주므로
    # 관측 → 액션 → 관측 이 자기참조 루프가 되어 추종 오차·속도·토크는 아무 의미가 없다
    # (관절속도가 항상 0.00 인 것이 그 증거다). 제어주기만 유효하다.
    dry = (sim is None) and (not bridge.ok)
    if log and not dry:
        t_last = log[-1][0]
        gap = [r[1] for r in log]
        dqm = [r[2] for r in log]
        print(f"\n실행 요약: {len(log)} step / {t_last:.2f}s")
        print(f"  목표-실제 관절차 최대 {np.degrees(max(gap)):.1f}°  "
              f"(중앙 {np.degrees(float(np.median(gap))):.1f}°)")
        print(f"  관절속도 최대 {max(dqm):.2f} rad/s = {np.degrees(max(dqm)):.0f}°/s")
    if tq_log and max(tq_log) > 0 and not dry:
        _w = 50
        _mv = max((float(np.mean(tq_log[i:i+_w])) for i in range(max(1, len(tq_log)-_w+1))),
                  default=0.0)
        print(f"  토크 사용률 최대 {max(tq_log)*100:.0f}%  중앙 {np.median(tq_log)*100:.0f}%  "
              f"1초창 평균 최대 {_mv*100:.0f}% (전류 한계 대비)")
        if tq_hold:
            print(f"    유지 구간만: 최대 {max(tq_hold)*100:.0f}%  "
                  f"중앙 {np.median(tq_hold)*100:.0f}%  "
                  f"— 5분 유지 시 과열 위험은 중앙값으로 판단한다")
    if len(cycles) > 20:
        c = sorted(cycles)
        p50 = c[len(c)//2]; p99 = c[int(len(c)*0.99)]
        over = sum(1 for x in cycles if x > CTRL_DT*1000*1.5)
        print(f"  제어주기 목표 {CTRL_DT*1000:.0f} ms → 중앙 {p50:.1f}  p99 {p99:.1f}  "
              f"최대 {c[-1]:.1f} ms")
        print(f"  주기 1.5배 초과 {over}회 / {len(cycles)}  "
              f"({'여유 있음' if p99 < 60 else '⚠ 지연 여유 60ms 초과'})")
    if events:
        print(f"\n이벤트 {len(events)}개 (heading 오차 = 동작 첫 프레임의 torso yaw - (수락 시 yaw + bin), "
              f"지연 = 수락 -> 세그먼트 첫 프레임, 패킷이 소켓에서 기다린 최대 1주기는 빠짐)")
        print(f"  {'#':>2} {'part':9s} {'bin':>5} {'목표°':>7} {'오차°':>7} {'지연ms':>7}")
        for i_, ev in enumerate(events):
            err = "-" if ev["err"] is None else f"{ev['err']:+.1f}"
            lat = "-" if ev["lat_ms"] is None else f"{ev['lat_ms']:.0f}"
            print(f"  {i_:2d} {ev['part']:9s} {ev['bin']:+5d} {ev['target']:+7.1f} {err:>7} {lat:>7}")
        errs = [abs(ev["err"]) for ev in events if ev["err"] is not None]
        if errs:
            print(f"  |heading 오차| 중앙 {np.median(errs):.1f}°  최대 {max(errs):.1f}°  ({len(errs)}/{len(events)}개 측정)")
    if ff_log:
        air = sum(1 for f in ff_log if f < 30.0) / len(ff_log)
        print(f"  발 수직반력 중앙 {np.median(ff_log):.0f} N  최소 {min(ff_log):.0f} N  "
              f"— 떠 있던 시간 {air*100:.0f}%")
    if csvf is not None:
        csvf.flush()
        print(f"  시계열 로그: {args.log_csv}")
    if sim is not None:
        hs = [r[3] for r in log if len(r) > 3]
        if hs:
            print(f"  골반 높이 최소 {min(hs):.3f} m → 판정: "
                  f"{'서 있음' if min(hs) > 0.4 else '넘어짐'}")
        if args.sim_push > 0 and hasattr(sim, "xy0"):
            print(f"  밀기 {sim.n_push}회 ({args.sim_push:.0f} N × {args.sim_push_dur:.1f}s, idle 중) → "
                  f"자리 이탈 최대 {sim.max_drift*100:.1f} cm (회전 세그먼트의 디딤도 포함)")
        if sim.renderer is not None and args.sim_video:
            try:
                sim.vw.close()
            except Exception:
                pass
            print(f"  영상: {args.sim_video} ({sim.nrender//2}프레임)")
    elif not args.arm:
        print(f"\ndry-run 이었다 ({len(log)} step). 물리가 없으므로 추종 오차·관절속도·토크는 "
              f"찍지 않았다 — 자기참조 루프라 의미가 없다.")
        print("  유효한 것: 관측 조립·재정렬·정책 추론 경로가 끝까지 돌았다는 것과 위의 제어주기.")
        print("  실제 발행은 --arm 이 필요하다.")
    else:
        print("발행 종료.")



# ── 3-b) 유지: 모션이 끝나도 정책을 끄지 않는다 ──────────────────────────────
# 정책이 균형을 잡는 주체다. 끄면 무조건 주저앉는다(실측으로 네 방식 모두 확인).
# 마지막 프레임을 계속 먹이면 서 있고, 그동안 하네스를 확인하거나 사람이 받칠 수 있다.
def policy_hold(n_steps, rq_at, rbq_at, tag, rqd_at=None):
    """정책을 켠 채로 주어진 참조 자세를 먹인다. 참조 속도는 0 — 멈춰 있으라는 뜻이다.
    rq_at(k) 가 k 번째 스텝의 참조 관절을 준다. 끊김·비상이면 False."""
    global last_action
    t_next2 = time.perf_counter()
    for k in range(n_steps):
        if stop["v"]:
            return False
        r = read_state()
        if r is None:
            print(f"[중단] lowstate 끊김 ({tag})"); return False
        q, dq, imu, gyro = r
        imu_aligned = quat_mul(quat_from_yaw(yaw_off or 0.0), imu)
        R_rob = torso_rot_world(imu_aligned, q)
        rbq = rbq_at(k) if callable(rbq_at) else rbq_at
        ori_b = (R_rob.T @ quat_to_mat(rbq))[:, :2].reshape(-1)
        rq = rq_at(k)
        rqd = rqd_at(k) if rqd_at is not None else np.zeros(NJ)
        obs = np.concatenate([rq, rqd, ori_b, gyro, q - QDEF, dq, last_action])
        action = policy(obs)
        last_action = action.copy()
        sent = publish_target(QDEF + ASCALE * action)
        log.append((log[-1][0] + CTRL_DT if log else 0.0,
                    float(np.abs(sent - q).max()), float(np.abs(dq).max()))
                   + ((sim.height(),) if sim is not None else ()))
        _tq = torque_now()
        tq_log.append(float(np.abs(_tq / ELIM_A).max()) if _tq is not None else 0.0)
        tq_hold.append(tq_log[-1])
        csv_row(tag, log[-1][0], k, sent, q, _tq, imu, gyro)
        if sim is not None:
            continue
        t_next2 += CTRL_DT
        slack = t_next2 - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            t_next2 = time.perf_counter()
    return True

if args.hold_sec > 0 and not stop["hard"]:
    n_hold = int(args.hold_sec / CTRL_DT)
    print(f"유지 — 마지막 자세로 서 있는다 (최대 {args.hold_sec:.0f}s). "
          f"정지를 누르면 관절을 풀고 끝낸다")
    rq_last, rbq_last = REF_Q[k], REF_BQ[k, B_ANCHOR]     # 마지막으로 실행한 라이브러리 프레임
    # 유지 자세. default 면 팔을 내린 기본 자세로 옮긴다 — 춤 마지막 프레임을 그대로
    # 들고 있으면(제로투는 손을 든 자세) 사람이 다가가기 무섭다.
    # 앵커 회전은 yaw 만 남겨 직립으로 만든다.
    if args.hold_frame >= 0:
        # 참조의 특정 프레임을 유지 자세로 쓴다. 방향(yaw)은 춤이 끝난 방향을 유지하고
        # 자세만 그 프레임에서 가져온다 — 로봇이 돌지 않게.
        _k = min(max(0, args.hold_frame), T_ALL - 1)
        rq_hold = REF_Q[_k].copy()
        rbq_hold = quat_from_yaw(yaw_of(rbq_last))
        print(f"유지 자세를 참조 {_k}번 프레임({_k/FPS:.2f}s)에서 가져온다")
    elif args.hold_pose == "default":
        rq_hold = QDEF.copy()
        rbq_hold = quat_from_yaw(yaw_of(rbq_last))
    else:
        rq_hold, rbq_hold = rq_last, rbq_last
    n_settle = int(args.settle_sec / CTRL_DT) if (args.hold_pose == "default" or args.hold_frame >= 0) else 0
    if n_settle > 0:
        d_arm = np.degrees(np.abs(rq_hold - rq_last).max())
        print(f"유지 자세로 {args.settle_sec:.1f}s 정착 — 팔을 내린 기본 자세로 옮긴다 "
              f"(참조 최대 이동 {d_arm:.0f}°)")
        _qd_last = REF_QD[k].copy()
        policy_hold(n_settle,
                    lambda k: rq_last + (0.5 - 0.5*np.cos(np.pi*(k+1)/n_settle)) * (rq_hold - rq_last),
                    lambda k: slerp(rbq_last, rbq_hold, 0.5 - 0.5*np.cos(np.pi*(k+1)/n_settle)),
                    "정착",
                    rqd_at=lambda k: _qd_last * (0.5 + 0.5*np.cos(np.pi*(k+1)/n_settle)))
        # 정책이 분포 밖 참조를 무시할 수 있다(쪼그려 앉히기가 그렇게 실패했다).
        # 실제로 팔이 내려왔는지 읽어서 확인한다.
        _r = read_state()
        if _r is not None:
            _q = _r[0]
            print(f"  정착 결과: 기본 자세와 평균 {np.degrees(np.abs(_q - rq_hold).mean()):.1f}° "
                  f"최대 {np.degrees(np.abs(_q - rq_hold).max()):.1f}°")
            for _nm in ("right_shoulder_pitch", "left_shoulder_pitch",
                        "right_elbow", "left_elbow"):
                _i = next(j for j, n in enumerate(I_JOINTS) if _nm in n)
                print(f"    {_nm:22s} 춤끝 {np.degrees(rq_last[_i]):+7.1f}° → "
                      f"목표 {np.degrees(rq_hold[_i]):+7.1f}° → "
                      f"실제 {np.degrees(_q[_i]):+7.1f}°")
    # 30 초마다 경과를 찍는다 — 사용자가 얼마나 남았는지 알아야 받칠 준비를 한다.
    n_chunk = int(30.0 / CTRL_DT)
    done = 0
    expired = True
    while done < n_hold:
        step = min(n_chunk, n_hold - done)
        if not policy_hold(step, lambda k: rq_hold, rbq_hold, "유지"):
            expired = False
            break
        done += step
        if done < n_hold:
            print(f"  유지 중 — {done*CTRL_DT:.0f}s 경과 / {args.hold_sec:.0f}s "
                  f"(정지를 누르면 관절을 푼다)")
    if expired and not stop["hard"]:
        # 시간 만료로 스스로 푸는 경우다. 경고 없이 풀면 로봇이 갑자기 주저앉는다.
        print(f"\n[경고] 유지 시간 {args.hold_sec:.0f}s 가 끝났다. 5초 뒤 관절을 푼다 — "
              f"로봇이 주저앉는다. 지금 받쳐라.")
        for c in (5, 4, 3, 2, 1):
            if not policy_hold(int(1.0 / CTRL_DT), lambda k: rq_hold, rbq_hold, f"카운트 {c}"):
                break
            print(f"  {c-1}...", flush=True)
    if stop["hard"]:
        send_damp()
        print_summary()
        print("[비상] 종료")
        sys.exit(0)
    # 유지를 정상적으로 끝냈다 = 사용자가 정지를 눌렀다 → 관절을 풀고 끝낸다.
    # 기본자세로 보간해도 정책이 없으니 결국 주저앉는다(실측 0.096 m). 댐핑이 더 부드럽다(0.170 m).
    send_damp()
    print_summary()
    print("관절을 풀었다. 로봇이 주저앉으므로 하네스·받침을 확인하라.")
    sys.exit(0)

n_exit = int(args.exit_sec / CTRL_DT)
print(f"기본 자세로 {args.exit_sec:.1f}s 복귀")
r = read_state()
q_end = r[0].copy() if r else QDEF.copy()
for k in range(n_exit):
    a = 0.5 - 0.5*np.cos(np.pi * (k+1)/n_exit)
    sent = publish_target(q_end + a * (QDEF - q_end))
    # 이 구간은 정책이 꺼져 있다. 기록하지 않으면 여기서 주저앉는지 측정되지 않는다.
    _r = read_state()
    if _r is not None:
        log.append((log[-1][0] + CTRL_DT if log else 0.0,
                    float(np.abs(sent - _r[0]).max()), float(np.abs(_r[1]).max()))
                   + ((sim.height(),) if sim is not None else ()))
        _t = torque_now()
        tq_log.append(float(np.abs(_t / ELIM_A).max()) if _t is not None else 0.0)
    if sim is None:
        time.sleep(CTRL_DT)

print_summary()
