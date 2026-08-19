"""시선 선택 -> 컵 파지 -> 입 전달 -> 기울이기 시퀀스.

안전 원칙:
  * 매 동작 직전에 anchor 게이트를 다시 확인한다. 한 번 통과했다고 끝까지 가지 않는다.
  * 컵을 든 뒤에는 항상 slow 속도.
  * 기울이기는 툴 피치로만, 각도/속도 상한 고정.
  * abort() 는 어느 단계에서든 컵을 테이블에 내려놓고 home 으로 복귀한다.
"""

from __future__ import annotations

import logging
import time
from enum import Enum

import numpy as np

from anchor import AnchorState
from kinematics import IKError

log = logging.getLogger(__name__)


class Stage(Enum):
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TO_MOUTH = "TO_MOUTH"
    POUR = "POUR"
    RETURN = "RETURN"
    DONE = "DONE"
    ABORTED = "ABORTED"


class AnchorLost(RuntimeError):
    pass


class DrinkTask:
    def __init__(self, cfg: dict, arm, anchor, refresh=None):
        """refresh: 게이트 직전에 SLAM/anchor 입력을 갱신하는 콜러블 (없어도 동작).

        ★ 이게 없으면 안전 게이트가 무의미하다 (2026-08-19).
        run() 은 팔이 다 움직일 때까지 블로킹이라, 그 동안 메인 루프가 SLAM 을 못 읽는다.
        예전 구현은 30초짜리 시퀀스 내내 멈춰 있는 옛 pose 를 검사하면서 "매 동작 직전에
        게이트 재확인"이라고 주장하고 있었다. 이제 매 게이트마다 refresh() 를 부른다.
        """
        self.cfg = cfg
        self.t = cfg["task"]
        self.arm = arm
        self.anchor = anchor
        self.refresh = refresh
        self.stage = Stage.IDLE
        self.cup_pos_ab: np.ndarray | None = None

    # ---------- 게이트 ----------
    def _gate(self, what: str):
        if self.refresh is not None:
            self.refresh()
        ok, why = self.anchor.can_execute_grasp()
        if not ok:
            raise AnchorLost(f"{what} 중단: {why}")

    def _pitch(self) -> float:
        return self.t["grasp_pitch_deg"]

    # ---------- 사전 검증 ----------
    def plan(self, p_cup_ab) -> list[str]:
        """실행 전에 전 구간 IK 도달성을 검사한다. 문제를 미리 말해준다."""
        p = np.asarray(p_cup_ab, float)
        pitch = self._pitch()
        wps = [
            ("pregrasp", p + [0, 0, self.t["pregrasp_height"]], pitch),
            ("grasp", p + [0, 0, self.t["grasp_z_offset"]], pitch),
            ("lift", p + [0, 0, self.t["lift_height"]], pitch),
            ("mouth_approach", np.array(self.t["mouth_pos"]) + self.t["mouth_approach_offset"], pitch),
            ("mouth", np.array(self.t["mouth_pos"]), pitch),
            ("pour", np.array(self.t["mouth_pos"]), pitch + self.t["pour_tilt_deg"]),
        ]
        bad = [f"{n} {np.round(q, 3).tolist()} @pitch{pi:.0f}"
               for n, q, pi in wps if not self.arm.reachable(q, pi)]
        return bad

    # ---------- 실행 ----------
    def run(self, p_cup_ab) -> Stage:
        p = np.asarray(p_cup_ab, float)
        pitch = self._pitch()
        bad = self.plan(p)
        if bad:
            log.error("도달 불가 웨이포인트: %s", bad)
            self.stage = Stage.ABORTED
            return self.stage
        self.cup_pos_ab = p

        try:
            self.stage = Stage.APPROACH
            self._gate("접근")
            self.arm.goto_pose(p + [0, 0, self.t["pregrasp_height"]], pitch,
                               gripper=self.arm.grip["open"])

            self.stage = Stage.GRASP
            self._gate("파지")
            self.arm.goto_pose(p + [0, 0, self.t["grasp_z_offset"]], pitch, slow=True)
            self.arm.close_gripper()

            self.stage = Stage.LIFT
            self._gate("들어올리기")
            self.arm.goto_pose(p + [0, 0, self.t["lift_height"]], pitch, slow=True)

            self.stage = Stage.TO_MOUTH
            self._gate("입으로 이동")
            mouth = np.array(self.t["mouth_pos"], float)
            self.arm.goto_pose(mouth + self.t["mouth_approach_offset"], pitch, slow=True)
            self._gate("입 접근")
            self.arm.goto_pose(mouth, pitch, slow=True)

            self.stage = Stage.POUR
            self._gate("기울이기")
            self.arm.tilt_in_place(pitch, pitch + self.t["pour_tilt_deg"],
                                   self.t["pour_tilt_speed_deg_s"])
            time.sleep(self.t["pour_hold_s"])
            self.arm.tilt_in_place(pitch + self.t["pour_tilt_deg"], pitch,
                                   self.t["pour_tilt_speed_deg_s"])

            self.stage = Stage.RETURN
            self._gate("복귀")
            self.arm.goto_pose(p + [0, 0, self.t["lift_height"]], pitch, slow=True)
            self.arm.goto_pose(p + [0, 0, self.t["grasp_z_offset"]], pitch, slow=True)
            self.arm.open_gripper()
            self.arm.goto_pose(p + [0, 0, self.t["pregrasp_height"]], pitch)
            self.arm.home()
            self.stage = Stage.DONE

        except (AnchorLost, IKError) as e:
            log.error("%s", e)
            self.abort()
        return self.stage

    def abort(self):
        """어느 단계에서든 안전하게 정지. 컵을 들고 있으면 내려놓는다."""
        log.warning("ABORT (stage=%s)", self.stage.value)
        try:
            if self.stage in (Stage.LIFT, Stage.TO_MOUTH, Stage.POUR) and self.cup_pos_ab is not None:
                pitch = self._pitch()
                # 먼저 컵을 세운다 (기울어진 채 이동 금지)
                p_now, pitch_now = self.arm.tcp()
                if abs(pitch_now - np.radians(pitch)) > np.radians(2):
                    self.arm.tilt_in_place(np.degrees(pitch_now), pitch,
                                           self.t["pour_tilt_speed_deg_s"])
                self.arm.goto_pose(self.cup_pos_ab + [0, 0, self.t["lift_height"]], pitch, slow=True)
                self.arm.goto_pose(self.cup_pos_ab + [0, 0, self.t["grasp_z_offset"]], pitch, slow=True)
                self.arm.open_gripper()
                self.arm.goto_pose(self.cup_pos_ab + [0, 0, self.t["pregrasp_height"]], pitch)
            self.arm.home()
        except Exception as e:            # abort 는 절대 예외를 밖으로 던지지 않는다
            log.error("abort 중 오류 (수동 개입 필요): %s", e)
        self.stage = Stage.ABORTED
