"""g1_button.ticks — 버튼 누르기 명령 계산(로봇 없이). 2026-11-16."""
import json
import os
import sys

G1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(G1, "deploy"))
import g1_button as gb  # noqa: E402

TRAJ = json.load(open(os.path.join(G1, "deploy", "button_press.json")))
META = json.load(open(os.path.join(G1, "deploy", "g1_tracking_policy_meta.json")))


def start29():
    # 시작 자세 대용: 트래킹 기본 자세를 Unitree 모터 순서(0..28)로. 실기 시작 자세는 Unitree 고수준 서기.
    order = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
             "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
             "waist_yaw", "waist_roll", "waist_pitch",
             "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
             "left_wrist_pitch", "left_wrist_yaw",
             "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
             "right_wrist_pitch", "right_wrist_yaw"]
    d = dict(zip(META["joint_names"], META["default_joint_pos"]))
    return [d[n + "_joint"] for n in order]


def run(hold_ticks):
    left = {"n": hold_ticks}

    def holding():
        left["n"] -= 1
        return left["n"] >= 0
    return list(gb.ticks(start29(), TRAJ, holding))


def test_starts_and_ends_at_start_pose_with_weight_ramp():
    s = start29()
    seq = run(10)
    fixed0 = [s[i] for i in gb.SDK_JOINTS]
    assert 0 < seq[0][0] < 0.1 and seq[0][1] == fixed0                    # 처음: 자세 그대로, weight 조금
    assert seq[-1][0] == 0.0 and seq[-1][1] == fixed0                      # 끝: 처음 자세, weight 0
    assert max(w for w, _ in seq) == 1.0


def test_left_arm_and_waist_held_still():
    s = start29()
    held = [s[i] for i in gb.WAIST + gb.LEFT]
    assert all(q[:10] == held for _, q in run(5))


def test_hold_keeps_press_pose_and_reaches_traj_press():
    seq = run(25)
    press_end = TRAJ["press"][-1]
    holds = [q[10:] for w, q in seq if q[10:] == press_end]
    assert len(holds) >= 25                                                # 유지 동안 누른 자세 그대로
    assert seq.index(next(x for x in seq if x[1][10:] == press_end)) * gb.DT > gb.T_RAMP + gb.T_LIFT


def test_planned_motion_within_speed_limit():
    seq = run(3)
    step = max(abs(a - b) for (_, p), (_, q) in zip(seq, seq[1:]) for a, b in zip(p, q))
    assert step <= gb.MAX_STEP + 1e-9                                      # 리미터가 경로를 왜곡하지 않는다


def test_traj_checks_passed():
    c = TRAJ["check"]
    assert c["press_err_mm"] < 10 and c["arm_torso_clear_m"] > 0.02 and c["arm_above_desk_min_m"] > 0.02
