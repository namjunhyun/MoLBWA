"""젯슨: 고수준(Unitree 기본 보행) 모드 그대로 오른팔만 arm_sdk 로 움직여 책상 위 버튼을 꾹 누르고 유지한다 (2026-11-16).
Python 3.8. g1_ctl.py button_toggle 이 띄우고(누르기) SIGUSR1 로 떼게 한다.

    python3 g1_button.py [--traj button_press.json] [--hold_max 120] [--dry start_q.json]

순서: 지금 관절각을 잡아 둔다 → arm_sdk 비율(weight) 0→1 (1 s, 자세 그대로) → 오른팔 현재 → 들어 올린 자세(2.5 s)
→ 버튼 위 → 누르기 → 유지(SIGUSR1 또는 hold_max 까지) → 역순 → 처음 자세 → weight 1→0 (1 s) → 끝.
왼팔·허리는 처음 자세에 고정한다(weight 1 인데 명령을 안 주면 kp 0 이라 늘어진다). SIGTERM/SIGINT = 지금 자세에서 weight 를
0.5 s 에 내려 Unitree 컨트롤러에 바로 돌려준다. 관절 명령은 틱당 MAX_STEP 이상 움직이지 않는다(계산 실수 방어).
걷기·균형은 Unitree 가 한다 — 리모컨으로 버튼 앞에 세운 뒤 쓴다. 손이 책상 앞 모서리보다 뒤에 있게 세울 것.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.02                                  # 50 Hz (Unitree arm_sdk 예제와 같음)
WAIST = [12, 13, 14]
LEFT = list(range(15, 22))
RIGHT = list(range(22, 29))
SDK_JOINTS = WAIST + LEFT + RIGHT          # arm_sdk 가 받는 17개
WEIGHT_IDX = 29                            # kNotUsedJoint: q = arm_sdk 섞는 비율
KP, KD = 60.0, 1.5                         # Unitree 예제 값
MAX_STEP = 1.5 * DT                        # 관절당 틱 이동 상한 1.5 rad/s
T_RAMP, T_LIFT, T_OFF = 1.0, 2.5, 0.5


def lerp(a, b, s):
    return [x + s * (y - x) for x, y in zip(a, b)]


def smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def resample(rows, fps):
    """30 fps 행렬 → 50 Hz (선형 보간)."""
    if len(rows) < 2:
        return [list(r) for r in rows]
    T = (len(rows) - 1) / fps
    out = []
    n = int(round(T / DT))
    for k in range(n):
        t = k * DT * fps
        i = min(int(t), len(rows) - 2)
        out.append(lerp(rows[i], rows[i + 1], t - i))
    return out + [list(rows[-1])]                  # 끝은 원본 그대로(누른 자세가 궤적 끝과 정확히 같게)


def press_path(start_r, traj):
    """오른팔(7) 경로: 현재 → 들어 올림 → 버튼 위 → 누름. 떼기는 이것을 거꾸로."""
    n = int(T_LIFT / DT)
    to_lift = [lerp(start_r, traj["q_lift"], smooth((k + 1) / n)) for k in range(n)]
    return to_lift + resample(traj["approach"], traj["fps"]) + resample(traj["press"], traj["fps"])


def ticks(start29, traj, holding):
    """(weight, 17관절 목표) 를 틱마다 낸다. holding() 이 False 가 되면 떼기 시작. 순수 계산 — PC 테스트용."""
    fixed = [start29[i] for i in WAIST + LEFT]
    start_r = [start29[i] for i in RIGHT]
    path = press_path(start_r, traj)
    n = int(T_RAMP / DT)
    for k in range(n):
        yield smooth((k + 1) / n), fixed + start_r
    for r in path:
        yield 1.0, fixed + r
    while holding():
        yield 1.0, fixed + path[-1]
    for r in reversed(path):
        yield 1.0, fixed + r
    for k in range(n):
        yield 1.0 - smooth((k + 1) / n), fixed + start_r


def limit(prev, tgt):
    return [p + max(-MAX_STEP, min(MAX_STEP, t - p)) for p, t in zip(prev, tgt)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=os.path.join(HERE, "button_press.json"))
    ap.add_argument("--hold_max", type=float, default=120.0, help="이만큼 누르고 있으면 스스로 뗀다(초)")
    ap.add_argument("--dry", default="", help="로봇 없이: 시작 관절각 29개 JSON({'q': [...]}) → 명령을 표준출력(csv)")
    a = ap.parse_args()
    traj = json.load(open(a.traj))
    state = {"release": False, "abort": False, "t_hold": None}

    def holding():
        if state["t_hold"] is None:
            state["t_hold"] = time.time()
            print("[누름] 유지 중 — 떼려면 패널 버튼을 다시", flush=True)
        if not a.dry and time.time() - state["t_hold"] > a.hold_max:
            print(f"[누름] {a.hold_max:.0f}s 지나 스스로 뗌", flush=True)
            return False
        return not state["release"] and not state["abort"]

    if a.dry:
        q0 = json.load(open(a.dry))["q"]
        state["release"] = True                     # 유지 없이 바로 떼기까지
        for w, q in ticks(q0, traj, holding):
            print(",".join(f"{v:.5f}" for v in [w] + q))
        return

    if subprocess.run(["pgrep", "-f", "g1_motion_server"], stdout=subprocess.DEVNULL).returncode == 0:
        sys.exit("거부: 정책 서버(저수준)가 돌고 있다 — 버튼 누르기는 고수준 모드에서만")
    sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    out = subprocess.run(["ip", "-4", "-o", "addr"], stdout=subprocess.PIPE, universal_newlines=True).stdout
    ChannelFactoryInitialize(0, next(f.split()[1] for f in out.splitlines() if " 192.168.123." in f))
    ms = MotionSwitcherClient(); ms.SetTimeout(5.0); ms.Init()
    r = ms.CheckMode()
    name = r[1].get("name", "?") if isinstance(r, tuple) and isinstance(r[1], dict) else "?"
    if name in ("", "?"):
        sys.exit(f"거부: 고수준 모드가 아니다(모드 {name!r}) — arm_sdk 는 Unitree 컨트롤러 위에서만 쓴다")

    low = {"q": None}
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda m: low.__setitem__("q", [m.motor_state[i].q for i in range(29)]), 10)
    t0 = time.time()
    while low["q"] is None and time.time() - t0 < 3:
        time.sleep(0.05)
    if low["q"] is None:
        sys.exit("거부: lowstate 를 못 받았다")
    start = list(low["q"])
    print(f"[시작] 모드 {name!r} · 오른팔 시작 {[round(start[i], 2) for i in RIGHT]}", flush=True)

    pub = ChannelPublisher("rt/arm_sdk", LowCmd_); pub.Init()
    cmd = unitree_hg_msg_dds__LowCmd_(); crc = CRC()
    signal.signal(signal.SIGUSR1, lambda *_: (state.__setitem__("release", True), print("[떼기] 요청", flush=True)))
    signal.signal(signal.SIGTERM, lambda *_: state.__setitem__("abort", True))
    signal.signal(signal.SIGINT, lambda *_: state.__setitem__("abort", True))

    def write(w, q17):
        cmd.motor_cmd[WEIGHT_IDX].q = w
        for j, v in zip(SDK_JOINTS, q17):
            mc = cmd.motor_cmd[j]
            mc.q, mc.dq, mc.tau, mc.kp, mc.kd = v, 0.0, 0.0, KP, KD
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

    prev = [start[j] for j in SDK_JOINTS]
    w = 0.0
    nxt = time.time()
    for w, q in ticks(start, traj, holding):
        if state["abort"]:
            break
        prev = limit(prev, q)
        write(w, prev)
        nxt += DT
        time.sleep(max(0.0, nxt - time.time()))
    if state["abort"]:                              # 비상: 지금 자세 그대로 weight 만 빨리 내려 Unitree 에 돌려준다
        print("[중단] weight 를 내려 Unitree 컨트롤러에 반납", flush=True)
        n = int(T_OFF / DT)
        for k in range(n):
            write(w * (1 - (k + 1) / n), prev)
            time.sleep(DT)
    write(0.0, prev)
    print("[끝] arm_sdk 해제", flush=True)


if __name__ == "__main__":
    main()
