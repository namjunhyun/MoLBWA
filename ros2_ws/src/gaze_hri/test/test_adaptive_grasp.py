"""적응형 그리퍼 닫기 (2026-10-14) — 지름이 다른 물체를 흉내 내는 가짜 버스로 확인한다."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rclpy  # noqa: E402

from gaze_hri.arm_server_node import ArmServer, FeetechBackend  # noqa: E402
from gaze_hri.so101_driver import JointMap  # noqa: E402

TPR = 4096 / (2 * math.pi)


class ObjectBus:
    """그리퍼(6번)는 obj_rad 에서 막힌다. 막힌 뒤 더 닫으라는 만큼 부하가 오른다 (P 제어)."""

    def __init__(self, obj_rad=None, k=2400):
        self.obj = None if obj_rad is None else 2048 + obj_rad * TPR
        self.k = k                      # 부하 / rad
        self.last_load = [0] * 6
        self.last_pos = [2048] * 6
        self.goal = [2048] * 6

    def write_ticks(self, goal):
        self.goal = list(goal)
        pos = list(goal)
        load = 40                       # 빈손으로 닫힐 때
        if self.obj is not None and goal[5] < self.obj:
            pos[5] = int(self.obj)
            load = int(min(1000, self.k * (self.obj - goal[5]) / TPR))
        self.last_pos, self.last_load = pos, [0] * 5 + [load]
        return goal

    def ticks(self):
        return list(self.last_pos)

    def loads(self):
        return list(self.last_load)


class TestAdaptiveGrasp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=["--ros-args", "-p", "backend:=dry_run", "-p", "stream_hz:=1000.0",
                         "-p", "grasp_check:=load", "-p", "min_grasp_load:=120",
                         "-p", "grasp_settle:=0.0",
                         "-p", "sag_correction:=false"])

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def run_close(self, obj):
        n = ArmServer()
        be = FeetechBackend.__new__(FeetechBackend)
        be.map = JointMap([2048] * 6, [1] * 6)
        be.bus = ObjectBus(obj)
        be.bias = [0.0] * 6
        n.backend = be
        n.q = [0.0, 1.0, -1.0, 0.0, 0.0, 2.24]
        held = n.close_and_verify()
        cmd = n.q[5]
        n.destroy_node()
        return held, cmd, be.bus

    def test_cup_and_thin_bottle_both_held_with_same_squeeze(self):
        for obj in (1.15, 0.55):        # 컵(실측 멈춤 위치), 가는 병
            held, cmd, bus = self.run_close(obj)
            self.assertTrue(held, obj)
            self.assertAlmostEqual(cmd, obj - 0.25, delta=0.03)
            self.assertGreater(bus.last_load[5], 400)

    def test_empty_closes_to_end_and_reports_empty(self):
        held, cmd, _ = self.run_close(None)
        self.assertFalse(held)
        self.assertAlmostEqual(cmd, 0.10, places=3)


if __name__ == "__main__":
    unittest.main()
