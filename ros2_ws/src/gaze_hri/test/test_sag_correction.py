"""처짐 보정 (FeetechBackend.settle) — 실제 서보 없이 '목표보다 덜 올라가는' 팔을 흉내 낸다."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gaze_hri.arm_server_node import FeetechBackend  # noqa: E402
from gaze_hri.so101_driver import JointMap  # noqa: E402


class SaggingBus:
    """쓴 목표에서 관절 1(어깨)은 sag 틱, 관절 2 는 그 절반만큼 모자란 곳에 선다."""

    def __init__(self, sag=46):
        self.sag = sag
        self.goal = [2048] * 6
        self.writes = []

    def write_ticks(self, goal):
        if self.writes:
            jump = max(abs(a - b) for a, b in zip(goal, self.writes[-1]))
            assert jump * 360 / 4096 <= 8.0, "보정이 한 번에 튀었다"
        self.goal = list(goal)
        self.writes.append(list(goal))

    def ticks(self):
        g = list(self.goal)
        g[1] -= self.sag
        g[2] -= self.sag // 2
        return g


def make(sag=46):
    be = FeetechBackend.__new__(FeetechBackend)
    be.map = JointMap([2048] * 6, [1] * 6)
    be.bus = SaggingBus(sag)
    be.bias = [0.0] * 6
    return be


class TestPanFriction(unittest.TestCase):
    def test_ff_leads_while_moving_and_fades_when_stopped(self):
        be = make(sag=0)
        be.fric_ff = math.radians(0.8)
        for k in range(30):                          # pan 이 + 방향으로 움직이는 중
            be.write([0.001 * k, 1.0, -1.0, -0.5, 0.0, 0.5])
        self.assertAlmostEqual(be.pan_ff, be.fric_ff, places=6)
        for _ in range(30):                          # 멈춤 -> 보상이 서서히 0 으로
            be.write([0.029, 1.0, -1.0, -0.5, 0.0, 0.5])
        self.assertAlmostEqual(be.pan_ff, 0.0, places=6)


class TestSag(unittest.TestCase):
    def test_settle_removes_steady_error(self):
        be = make()
        q = [0.0, 1.0, -1.0, -0.5, 0.0, 0.5]
        be.write(q)
        self.assertGreater(abs(be.read()[1] - q[1]), math.radians(3.5))      # 4° 처짐
        left = be.settle(6.0, 2, wait=0.0)
        act = be.read()
        for j in (1, 2, 3):
            self.assertLess(abs(act[j] - q[j]), math.radians(0.2))
        self.assertLess(left, 4.5)
        be.write(q)                                   # bias 가 다음 쓰기에도 남는다
        self.assertLess(abs(be.read()[1] - q[1]), math.radians(0.2))

    def test_bias_is_clamped(self):
        be = make(sag=200)                            # 17.6° — 비정상, 다 덧대면 안 된다
        q = [0.0, 1.0, -1.0, -0.5, 0.0, 0.5]
        be.write(q)
        be.settle(6.0, 3, wait=0.0)
        self.assertLessEqual(abs(math.degrees(be.bias[1])), 6.0 + 1e-9)

    def test_reset_bias(self):
        be = make()
        be.write([0.0, 1.0, -1.0, -0.5, 0.0, 0.5])
        be.settle(6.0, 2, wait=0.0)
        be.reset_bias()
        self.assertEqual(be.bias, [0.0] * 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestLiveTrack(unittest.TestCase):
    def test_track_step_pays_back_sag_while_moving(self):
        """2026-10-14: 물병 따르기 중 처짐 — 쓸 때마다 조금씩 bias 를 더해 목표에 붙는다."""
        be = make()
        q = [0.0, 1.0, -1.0, -0.5, 0.0, 0.5]
        for _ in range(150):
            be.write(q)
            be.bus.last_pos = be.bus.ticks()
            be.track_step(0.04, math.radians(20))
        self.assertLess(abs(be.read()[1] - q[1]), math.radians(0.5))
        self.assertLess(abs(be.read()[2] - q[2]), math.radians(0.5))
