"""따르기 경로 (2026-10-14) — 사용자가 짚은 조건들을 계획 단계에서 지키는지."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gaze_hri.pour import PourConfig, axis, plan_pour  # noqa: E402

CFG = PourConfig(bottle_height=0.16, cup_height=0.15, lip_clear=0.10, pre_deg=30, onset_deg=60)


def walk(plan):
    for name in ("pre", "move", "onset", "pour"):
        for g, th, _ in plan["segments"][name]:
            yield name, g, th


class TestPourPlan(unittest.TestCase):
    def setUp(self):
        self.cup = np.array([0.36, -0.06, 0.0])
        self.plan = plan_pour(np.array([0.34, 0.19, 0.06]), self.cup[:2], CFG)
        self.L = CFG.bottle_height - CFG.grasp_height

    def test_bottle_bottom_always_above_cup_rim(self):
        """컵을 치지 않게: 병 바닥이 언제나 컵 높이 + body_clear 위 (실제 위치 오차 ~7cm 대비)."""
        for name, g, th in walk(self.plan):
            bottom = g - CFG.grasp_height * axis(th, self.plan["s"], g)
            self.assertGreaterEqual(bottom[2], CFG.cup_height + CFG.body_clear - 1e-6, (name, th))

    def test_no_tilt_beyond_pre_until_mouth_over_cup(self):
        """컵 밖에서 따르지 않게: 입구가 컵 테두리 안쪽에 오기 전에는 pre 각도를 넘지 않는다."""
        for name, g, th in walk(self.plan):
            mouth = g + self.L * axis(th, self.plan["s"], g)
            if th > math.radians(CFG.pre_deg) + 1e-6:
                d = math.hypot(mouth[0] - self.cup[0], mouth[1] - self.cup[1])
                self.assertLess(d, CFG.cup_rim_radius, (name, math.degrees(th)))

    def test_pour_mouth_height(self):
        """따르는 구간: 입구 높이 = 컵 + lip_clear (사용자 요청 25cm)."""
        for g, th, _ in self.plan["segments"]["pour"]:
            mouth = g + self.L * axis(th, self.plan["s"], g)
            self.assertAlmostEqual(mouth[2], CFG.cup_height + CFG.lip_clear, places=3)


if __name__ == "__main__":
    unittest.main()
