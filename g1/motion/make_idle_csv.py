"""촬영 전 임시 idle 클립: G1 기본자세(편하게 선 자세)로 가만히 서 있는 csv (36열, 30 fps).

밀어도 버티는 대기 정책을 촬영 없이 먼저 학습하려고 만든다. 촬영한 idle.csv 가 생기면 그걸 쓴다.
관절 순서는 csv_to_npz.py 의 URDF 순서이고, 기본자세 값(meta, Isaac BFS 순서)은 **이름으로** 옮긴다
(관절 순서를 위치로 옮기다 세 번 틀렸다).

    python3 make_idle_csv.py --out idle_synth.csv --seconds 10
"""
import argparse
import json
import os

import numpy as np

CSV_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def idle_rows(default_by_name: dict, seconds: float, fps: int = 30, root_z: float = 0.76) -> np.ndarray:
    """(T, 36): root pos 3 + quat xyzw 4 + 관절 29. 호흡 정도로 허리 pitch 를 ±0.5° 흔든다."""
    T = int(round(seconds * fps)) + 1
    q = np.array([default_by_name[n] for n in CSV_JOINTS], dtype=float)
    rows = np.zeros((T, 36))
    rows[:, 2] = root_z
    rows[:, 6] = 1.0                                   # 직립, yaw 0
    rows[:, 7:] = q
    t = np.arange(T) / fps
    rows[:, 7 + CSV_JOINTS.index("waist_pitch_joint")] += np.radians(0.5) * np.sin(2 * np.pi * t / 4.0)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=os.path.join(os.path.dirname(__file__), "..", "deploy",
                                                   "g1_tracking_policy_meta.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--root_z", type=float, default=0.76, help="G1 로봇 cfg init_state z")
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    dflt = dict(zip(meta["joint_names"], meta["default_joint_pos"]))
    assert sorted(dflt) == sorted(CSV_JOINTS), "meta 관절 이름과 csv 관절 목록이 다르다"
    rows = idle_rows(dflt, a.seconds, root_z=a.root_z)
    np.savetxt(a.out, rows, delimiter=",", fmt="%.9f")
    print(f"{a.out}: {rows.shape[0]} 프레임, 어깨 pitch {np.degrees(dflt['left_shoulder_pitch_joint']):.1f}°, "
          f"팔꿈치 {np.degrees(dflt['left_elbow_joint']):.1f}°, 무릎 {np.degrees(dflt['left_knee_joint']):.1f}°")
