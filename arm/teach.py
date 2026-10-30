#!/usr/bin/env python
"""캘리브레이션 헬퍼 — 관절 부호 확인, 입 위치 티칭, 작업공간 확인.

  python teach.py signs       # 각 관절을 조금씩 움직여 joint_sign 확인
  python teach.py pose        # 현재 자세의 TCP 좌표 출력 (입 위치 티칭용)
  python teach.py workspace   # top-down 도달 가능 영역 출력
  python teach.py gripper     # 그리퍼 open/closed 값 탐색
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import yaml

from arm import Arm
from kinematics import JOINT_NAMES, ArmModel, IKError, to_motor_deg


def cmd_signs(cfg, arm):
    print("각 관절을 +8deg 움직인다. 화면의 예상 방향과 실제가 반대면 config 의 joint_sign 을 뒤집어라.\n")
    expect = {"shoulder_pan": "TCP 가 왼쪽(+y)으로", "shoulder_lift": "TCP 가 위로",
              "elbow_flex": "팔꿈치가 펴지는 쪽", "wrist_flex": "그리퍼 끝이 위로",
              "wrist_roll": "그리퍼가 반시계로"}
    base = arm.joint_deg()
    for name in JOINT_NAMES:
        input(f"[{name}] 예상: {expect[name]} — Enter")
        tgt = dict(base)
        tgt[name] = base[name] + 8
        arm._send({k: v for k, v in tgt.items() if k in JOINT_NAMES})
        time.sleep(1.2)
        arm._send({k: v for k, v in base.items() if k in JOINT_NAMES})
        time.sleep(0.8)


def cmd_pose(cfg, arm):
    print("팔을 손으로 원하는 위치에 두고 Enter. (토크가 걸려 있으면 먼저 풀어라)")
    print("입 위치 티칭: 컵을 물고 마실 위치에 그리퍼 끝을 두면 된다.\n")
    while True:
        try:
            input("Enter=측정, Ctrl-C=종료")
        except KeyboardInterrupt:
            return
        p, pitch = arm.tcp()
        print(f"  TCP  = [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]   pitch = {math.degrees(pitch):+.1f} deg")
        print(f"  joints = { {k: round(v,1) for k,v in arm.joint_deg().items()} }")
        print(f"  -> config.yaml  task.mouth_pos: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]\n")


def cmd_workspace(cfg, arm):
    m: ArmModel = arm.model
    H = cfg["arm"]["base_height_m"]
    pitch = math.radians(cfg["task"]["grasp_pitch_deg"])
    print(f"받침대 {H*100:.0f}cm, top-down(pitch {cfg['task']['grasp_pitch_deg']:.0f}deg) 기준")
    print("절대높이[m]  도달반경 r [m]")
    for z_abs in np.arange(0.0, 0.30, 0.02):
        z = z_abs - H
        rs = [r for r in np.arange(0.04, 0.32, 0.005) if arm.reachable([r, 0, z], math.degrees(pitch))]
        print(f"  {z_abs:.2f}      " + (f"{min(rs):.3f} ~ {max(rs):.3f}" if rs else "도달 불가"))
    print("\n컵은 위 반경 안에, 사람 입도 위 반경 안에 오도록 배치할 것.")


def cmd_gripper(cfg, arm):
    print("그리퍼 값을 바꿔가며 확인. 숫자 입력, 빈 줄이면 종료.")
    while True:
        s = input("gripper.pos (0~100): ").strip()
        if not s:
            return
        arm._send({k: v for k, v in arm.joint_deg().items() if k in JOINT_NAMES}, float(s))
        time.sleep(0.5)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cfg = yaml.safe_load(open("config.yaml"))
    dry = "--dry-run" in sys.argv
    cmds = {"signs": cmd_signs, "pose": cmd_pose, "workspace": cmd_workspace, "gripper": cmd_gripper}
    fn = cmds.get(sys.argv[1])
    if fn is None:
        print(__doc__)
        return 1
    with Arm(cfg, dry_run=dry) as arm:
        fn(cfg, arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
