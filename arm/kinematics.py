"""SO-ARM101 (5-DOF + gripper) 정/역기구학.

구조: shoulder_pan(z축 회전) 뒤에 shoulder_lift / elbow_flex / wrist_flex 세 개가
같은 피치 축으로 평면 3R 을 이루고, 마지막에 wrist_roll(툴 축 회전)이 붙는다.
따라서 위치(3) + 툴 피치(1) = 4개까지만 지정할 수 있고 툴 요(yaw)는 pan 에 종속된다.
=> 컵 파지는 top-down (pitch = -90deg) 으로 고정하는 게 유일하게 안정적이다.

좌표계 B(armbase): x = 팔 정면, y = 왼쪽, z = 위. 원점은 베이스 바닥의 pan 축.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class IKError(ValueError):
    """목표점이 작업공간 밖이거나 관절 한계를 위반할 때."""


@dataclass(frozen=True)
class ArmModel:
    L1: float  # 바닥 -> shoulder_lift 축
    L2: float  # shoulder_lift -> elbow_flex
    L3: float  # elbow_flex -> wrist_flex
    L4: float  # wrist_flex -> TCP(파지점)

    @property
    def reach_max(self) -> float:
        return self.L2 + self.L3 + self.L4

    def fk(self, q_rad: np.ndarray) -> tuple[np.ndarray, float]:
        """관절각(rad, 수학 부호) -> (TCP 위치 [m], 툴 피치 [rad])."""
        t1, t2, t3, t4 = q_rad[0], q_rad[1], q_rad[2], q_rad[3]
        pitch = t2 + t3 + t4
        r = self.L2 * math.cos(t2) + self.L3 * math.cos(t2 + t3) + self.L4 * math.cos(pitch)
        z = self.L1 + self.L2 * math.sin(t2) + self.L3 * math.sin(t2 + t3) + self.L4 * math.sin(pitch)
        return np.array([r * math.cos(t1), r * math.sin(t1), z]), pitch

    def ik(self, p: np.ndarray, pitch_rad: float, roll_rad: float = 0.0,
           elbow_up: bool = True) -> np.ndarray:
        """(위치, 툴 피치) -> 관절각 5개 [rad]. 해가 없으면 IKError."""
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        t1 = math.atan2(y, x)
        r = math.hypot(x, y)

        # 손목 중심(wrist_flex 축)까지 되짚어 내려간다.
        rc = r - self.L4 * math.cos(pitch_rad)
        zc = z - self.L1 - self.L4 * math.sin(pitch_rad)

        d2 = rc * rc + zc * zc
        d = math.sqrt(d2)
        if d > self.L2 + self.L3 - 1e-6:
            raise IKError(f"목표가 너무 멀다: 손목중심거리 {d:.3f}m > {self.L2 + self.L3:.3f}m")
        if d < abs(self.L2 - self.L3) + 1e-6:
            raise IKError(f"목표가 너무 가깝다: 손목중심거리 {d:.3f}m")

        # 2R 역기구학
        cos_t3 = (d2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_t3 = max(-1.0, min(1.0, cos_t3))
        t3 = math.acos(cos_t3)
        if elbow_up:
            t3 = -t3
        t2 = math.atan2(zc, rc) - math.atan2(self.L3 * math.sin(t3), self.L2 + self.L3 * math.cos(t3))
        t4 = pitch_rad - t2 - t3
        return np.array([t1, t2, t3, t4, roll_rad])

    def ik_checked(self, p, pitch_rad, roll_rad=0.0, limits_deg=None, tol=2e-3) -> np.ndarray:
        """IK + FK 왕복 검증 + 관절한계 검사. elbow_up/down 둘 다 시도한다."""
        errors = []
        for elbow_up in (True, False):
            try:
                q = self.ik(np.asarray(p, float), pitch_rad, roll_rad, elbow_up)
            except IKError as e:
                errors.append(str(e))
                continue
            p_fk, pitch_fk = self.fk(q)
            if np.linalg.norm(p_fk - np.asarray(p, float)) > tol:
                errors.append("FK 왕복 불일치")
                continue
            if abs(_wrap(pitch_fk - pitch_rad)) > 1e-3:
                errors.append("피치 불일치")
                continue
            if limits_deg is not None:
                bad = _limit_violations(q, limits_deg)
                if bad:
                    errors.append(f"관절한계 위반 {bad}")
                    continue
            return q
        raise IKError("; ".join(errors) or "해 없음")


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _limit_violations(q_rad: np.ndarray, limits_deg: dict) -> list[str]:
    bad = []
    for name, val in zip(JOINT_NAMES, q_rad):
        lo, hi = limits_deg.get(name, (-180, 180))
        deg = math.degrees(val)
        if not (lo - 1e-6 <= deg <= hi + 1e-6):
            bad.append(f"{name}={deg:.1f}deg not in [{lo},{hi}]")
    return bad


def to_motor_deg(q_rad: np.ndarray, sign: dict, offset: dict) -> dict[str, float]:
    """수학 각도 -> lerobot 모터 지령(deg)."""
    return {
        name: sign.get(name, 1) * math.degrees(val) + offset.get(name, 0.0)
        for name, val in zip(JOINT_NAMES, q_rad)
    }


def from_motor_deg(motor_deg: dict, sign: dict, offset: dict) -> np.ndarray:
    """lerobot 모터 각도 -> 수학 각도 [rad]."""
    return np.array([
        math.radians((motor_deg[name] - offset.get(name, 0.0)) / sign.get(name, 1))
        for name in JOINT_NAMES
    ])
