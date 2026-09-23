"""SO-ARM101 모션 컨트롤러 — 속도 제한 카테시안 이동 + 그리퍼 + 기울이기.

안전 규칙 (하드코딩, 우회 금지):
  1. 모든 이동은 조인트 공간에서 보간되고 속도 제한이 걸린다.
  2. 얼굴 근처 구간(slow=True)은 별도의 낮은 속도를 쓴다.
  3. IK 실패 / 관절한계 위반은 예외로 즉시 중단한다. 클램프해서 억지로 가지 않는다.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from kinematics import JOINT_NAMES, ArmModel, IKError, from_motor_deg, to_motor_deg

log = logging.getLogger(__name__)


class DryRunBus:
    """하드웨어 없이 로직만 돌릴 때 쓰는 가짜 로봇."""

    def __init__(self, home: dict):
        self._pos = {f"{k}.pos": float(v) for k, v in home.items()}
        self._pos.setdefault("gripper.pos", 30.0)

    def get_observation(self):
        return dict(self._pos)

    def send_action(self, action):
        self._pos.update(action)
        return dict(action)

    def disconnect(self):
        pass


class Arm:
    def __init__(self, cfg: dict, dry_run: bool = False):
        a = cfg["arm"]
        self.cfg = cfg
        self.model = ArmModel(**a["links"])
        self.sign = a["joint_sign"]
        self.offset = a["joint_offset"]
        self.limits = {k: tuple(v) for k, v in a["joint_limits_deg"].items()}
        self.grip = a["gripper"]
        self.motion = a["motion"]
        self.dry_run = dry_run
        self._robot = None

    # ---------- 연결 ----------
    def connect(self):
        if self.dry_run:
            self._robot = DryRunBus(self.motion["home_deg"])
            log.warning("DRY RUN: 실제 팔에 연결하지 않는다")
            return
        from lerobot.robots.so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        a = self.cfg["arm"]
        self._robot = SOFollower(SOFollowerRobotConfig(
            type=a["id"],
            port=a["port"],
            use_degrees=a["use_degrees"],
            max_relative_target=a["max_relative_target"],
        ))
        self._robot.connect()
        log.info("SO-ARM101 연결됨 (%s)", a["port"])

    def disconnect(self):
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    # ---------- 상태 ----------
    def joint_deg(self) -> dict[str, float]:
        obs = self._robot.get_observation()
        return {k[:-4]: float(v) for k, v in obs.items() if k.endswith(".pos")}

    def q_rad(self) -> np.ndarray:
        return from_motor_deg(self.joint_deg(), self.sign, self.offset)

    def tcp(self) -> tuple[np.ndarray, float]:
        return self.model.fk(self.q_rad())

    # ---------- 저수준 이동 ----------
    def _send(self, motor_deg: dict[str, float], gripper: float | None = None):
        action = {f"{k}.pos": float(v) for k, v in motor_deg.items()}
        if gripper is not None:
            action["gripper.pos"] = float(gripper)
        self._robot.send_action(action)

    def goto_joints(self, q_target: np.ndarray, slow: bool = False, gripper: float | None = None):
        """조인트 공간 선형 보간 + 속도 제한 이동."""
        q0 = self.q_rad()
        speed = math.radians(
            self.motion["slow_joint_speed_deg_s"] if slow else self.motion["max_joint_speed_deg_s"]
        )
        dq = q_target - q0
        duration = float(np.max(np.abs(dq))) / speed if speed > 0 else 0.0
        dt = 1.0 / self.motion["rate_hz"]
        n = max(1, int(math.ceil(duration / dt)))
        for i in range(1, n + 1):
            q = q0 + dq * (i / n)
            self._send(to_motor_deg(q, self.sign, self.offset), gripper)
            time.sleep(dt)

    def goto_pose(self, p, pitch_deg: float, slow: bool = False,
                  roll_deg: float = 0.0, gripper: float | None = None):
        """카테시안 목표. IK 불가면 IKError 를 그대로 올린다 (억지로 근사하지 않는다)."""
        q = self.model.ik_checked(
            np.asarray(p, float), math.radians(pitch_deg),
            math.radians(roll_deg), limits_deg=self.limits,
        )
        self.goto_joints(q, slow=slow, gripper=gripper)
        return q

    def reachable(self, p, pitch_deg: float) -> bool:
        try:
            self.model.ik_checked(np.asarray(p, float), math.radians(pitch_deg), limits_deg=self.limits)
            return True
        except IKError:
            return False

    # ---------- 그리퍼 ----------
    def open_gripper(self):
        self._send(to_motor_deg(self.q_rad(), self.sign, self.offset), self.grip["open"])
        time.sleep(self.grip["settle_s"])

    def close_gripper(self):
        self._send(to_motor_deg(self.q_rad(), self.sign, self.offset), self.grip["closed"])
        time.sleep(self.grip["settle_s"])

    def home(self):
        home = self.motion["home_deg"]
        q = from_motor_deg(home, self.sign, self.offset)
        self.goto_joints(q, slow=False, gripper=self.grip["open"])

    # ---------- 기울이기 ----------
    def tilt_in_place(self, from_pitch_deg: float, to_pitch_deg: float, speed_deg_s: float):
        """TCP 위치는 고정한 채 툴 피치만 천천히 바꾼다 = 컵 기울이기.

        wrist_roll 이 아니라 피치를 쓴다. top-down 파지에서 wrist_roll 은
        수직축 회전이라 컵이 전혀 기울지 않는다.
        """
        p, _ = self.tcp()
        steps = max(1, int(abs(to_pitch_deg - from_pitch_deg) / speed_deg_s * self.motion["rate_hz"]))
        dt = 1.0 / self.motion["rate_hz"]
        for i in range(1, steps + 1):
            pitch = from_pitch_deg + (to_pitch_deg - from_pitch_deg) * i / steps
            q = self.model.ik_checked(p, math.radians(pitch), 0.0, limits_deg=self.limits)
            self._send(to_motor_deg(q, self.sign, self.offset))
            time.sleep(dt)
