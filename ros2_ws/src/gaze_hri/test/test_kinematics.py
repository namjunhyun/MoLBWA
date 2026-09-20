#!/usr/bin/env python3
"""기구학 / 좌표계 정합 검증.

rclpy 없이 도는 순수 기하 테스트다. ROS2 환경이 없어도 실행된다:

    python -m unittest discover ros2_ws/src/gaze_hri/test -v
    # 또는
    pytest ros2_ws/src/gaze_hri/test
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gaze_hri.kinematics import (  # noqa: E402
    TOPDOWN_CALIB_XY, ArmGeometry, IKError, alignment_rmse, fit_plane_ransac,
    forward_kinematics, grasp_detected, height_above_plane,
    inverse_kinematics, project_onto_plane, quat_to_matrix,
    solve_with_fallback, umeyama_rigid)


class TestForwardInverse(unittest.TestCase):
    """IK로 푼 각을 FK에 다시 넣으면 원래 좌표가 나와야 한다."""

    def setUp(self):
        self.geo = ArmGeometry()

    def test_ik_fk_roundtrip(self):
        rng = np.random.default_rng(0)
        checked = 0
        worst = 0.0
        for _ in range(3000):
            p = [rng.uniform(0.16, 0.34), rng.uniform(-0.20, 0.20),
                 rng.uniform(0.0, 0.18)]
            try:
                q, pitch = solve_with_fallback(p, self.geo)
            except IKError:
                continue
            back = forward_kinematics(q, self.geo)
            err = float(np.linalg.norm(back - np.array(p)))
            worst = max(worst, err)
            checked += 1
        self.assertGreater(checked, 100, "검증된 표본이 너무 적다")
        self.assertLess(worst, 1e-9, f"FK/IK 왕복 오차 {worst}")

    def test_approach_pitch_is_respected(self):
        """요청한 접근각이 실제 손목 자세로 반영되는가.

        FK에서 손목 누적각 a3 = j2+j3+j4 가 곧 접근각이다.
        관절 한계 때문에 지점마다 풀리는 각이 다르므로, 풀린 경우에 대해서만
        누적각이 요청값과 일치하는지 본다.
        """
        checked = 0
        for deg in (-60, -45, -30):
            pitch = math.radians(deg)
            for x in np.arange(0.18, 0.38, 0.01):
                for elbow_up in (True, False):
                    try:
                        q = inverse_kinematics([float(x), 0.0, 0.045],
                                               self.geo, pitch, elbow_up)
                    except IKError:
                        continue
                    self.assertAlmostEqual(q[1] + q[2] + q[3], pitch, places=9)
                    checked += 1
        self.assertGreater(checked, 10, "검증된 자세가 너무 적다")

    def test_approach_pitch_shifts_the_reachable_ring(self):
        """접근각은 '안 되면 더 가파르게'가 아니라 거리에 맞춰 고르는 값이다.

        각 접근각마다 도달 가능한 반경 띠가 다르다. 가파를수록 가까운 쪽,
        얕을수록 먼 쪽이 열린다. solve_with_fallback 이 여러 각을 시도하는
        이유이며, 이 성질이 깨지면 파지 성공률 튜닝 전략을 다시 봐야 한다.
        """
        def reach_band(deg):
            pitch = math.radians(deg)
            rs = []
            for x in np.arange(0.16, 0.40, 0.01):
                for y in np.arange(-0.20, 0.21, 0.02):
                    for elbow_up in (True, False):
                        try:
                            inverse_kinematics([x, y, 0.045], self.geo,
                                               pitch, elbow_up)
                            rs.append(math.hypot(x, y))
                            break
                        except IKError:
                            continue
            return (min(rs), max(rs)) if rs else (None, None)

        steep_lo, steep_hi = reach_band(-75)
        flat_lo, flat_hi = reach_band(-20)
        self.assertIsNotNone(steep_lo)
        self.assertIsNotNone(flat_lo)
        self.assertLess(steep_lo, flat_lo, "가파른 각이 더 가까운 쪽을 열어야 한다")
        self.assertLess(steep_hi, flat_hi, "얕은 각이 더 먼 쪽을 열어야 한다")

    def test_out_of_reach_raises(self):
        """링크 합보다 먼 목표는 조용히 통과하면 안 된다."""
        far = self.geo.shoulder_offset + self.geo.l1 + self.geo.l2 + self.geo.l3 + 0.05
        with self.assertRaises(IKError):
            solve_with_fallback([far, 0.0, 0.05], self.geo)

    def test_workspace_radius_is_inside_reach(self):
        """config 의 안전 반경이 실제 도달 범위보다 작아야 검사가 의미를 갖는다."""
        reach = (self.geo.shoulder_offset + self.geo.l1
                 + self.geo.l2 + self.geo.l3)
        self.assertLess(0.38, reach,
                        "workspace_radius 기본값이 도달 범위보다 크다")


class TestCalibrationPoses(unittest.TestCase):
    """호모그래피 캘리브 지점은 전부 '같은 평면'에 있어야 한다.

    높이가 흩어지면 재투영 오차는 작게 나오면서 실제 테이블면 오차가 커진다.
    """

    def test_calib_points_are_coplanar_and_reachable(self):
        geo = ArmGeometry()
        table_z, calib_h = 0.0, 0.005
        heights = []
        for (x, y) in TOPDOWN_CALIB_XY:
            q, _ = solve_with_fallback([x, y, table_z + calib_h], geo)
            heights.append(float(forward_kinematics(q, geo)[2]))
        self.assertEqual(len(heights), len(TOPDOWN_CALIB_XY),
                         "도달 못 하는 캘리브 지점이 있다")
        self.assertLess(max(heights) - min(heights), 1e-6,
                        "캘리브 지점 높이가 흩어져 있다(평면 가정 위반)")

    def test_calib_points_are_spread_out(self):
        """한 직선/한 점에 몰리면 호모그래피가 제대로 안 풀린다."""
        pts = np.array(TOPDOWN_CALIB_XY, dtype=float)
        _, s, _ = np.linalg.svd(pts - pts.mean(axis=0))
        self.assertGreater(s[1] / s[0], 0.3, "캘리브 지점이 한 직선에 가깝다")

    def test_calib_points_cover_near_and_far(self):
        """가까운 쪽에 점이 없으면 화면 안쪽이 전부 외삽이 된다."""
        r = [math.hypot(x, y) for (x, y) in TOPDOWN_CALIB_XY]
        self.assertLess(min(r), 0.23, "로봇 가까운 쪽 캘리브 점이 없다")
        self.assertGreater(max(r) - min(r), 0.08, "캘리브 점의 거리 범위가 좁다")


class TestGraspDetection(unittest.TestCase):
    """물체를 물면 그리퍼가 명령값까지 닫히지 못한다는 원리."""

    def test_held_when_gripper_cannot_close(self):
        # 컵을 물어 명령(0.15)보다 열려 있음 -> 물었음
        self.assertTrue(grasp_detected(0.15, 0.31, 0.08))

    def test_empty_when_gripper_closes_fully(self):
        # 빈손이라 끝까지 닫힘 -> 빈손
        self.assertFalse(grasp_detected(0.15, 0.02, 0.08))

    def test_margin_separates_close_cases(self):
        # 기준보다 확실히 작으면 빈손, 확실히 크면 물었음
        self.assertFalse(grasp_detected(0.15, 0.22, 0.08))   # 차이 0.07
        self.assertTrue(grasp_detected(0.15, 0.24, 0.08))    # 차이 0.09

    def test_commanded_value_alone_is_not_held(self):
        """명령값이 그대로 돌아오면(차이 0) 물었다고 판정하면 안 된다.

        예전 dry-run 백엔드가 성공 시 명령값을 그대로 돌려줘서
        fail_rate 와 무관하게 항상 '빈손'이 나왔다.
        """
        self.assertFalse(grasp_detected(0.15, 0.15, 0.08))


class TestWorldToBase(unittest.TestCase):
    """SLAM world <-> 로봇 베이스 정합."""

    def test_umeyama_recovers_known_transform(self):
        th = math.radians(30)
        R = np.array([[math.cos(th), -math.sin(th), 0],
                      [math.sin(th), math.cos(th), 0],
                      [0, 0, 1.0]])
        t = np.array([-0.50, 0.35, -0.20])
        dst = np.array([[0.30, 0.0, 0.12], [0.25, 0.18, 0.15],
                        [0.25, -0.18, 0.15], [0.31, 0.10, 0.23],
                        [0.31, -0.10, 0.23], [0.33, 0.0, 0.09]])
        src = (np.linalg.inv(R) @ (dst - t).T).T
        T = umeyama_rigid(src, dst)
        self.assertLess(alignment_rmse(T, src, dst), 1e-9)
        np.testing.assert_allclose(T[:3, :3], R, atol=1e-9)
        np.testing.assert_allclose(T[:3, 3], t, atol=1e-9)

    def test_umeyama_rejects_reflection(self):
        """거울상 해가 나오면 안 된다(det = +1)."""
        rng = np.random.default_rng(3)
        dst = rng.normal(size=(8, 3))
        src = dst.copy()
        src[:, 2] *= -1          # 반사된 대응
        T = umeyama_rigid(src, dst)
        self.assertGreater(np.linalg.det(T[:3, :3]), 0.0)

    def test_quat_to_matrix_is_rotation(self):
        for q in [(0, 0, 0, 1), (0.5, 0.5, 0.5, 0.5), (0.1, -0.2, 0.3, 0.9)]:
            R = quat_to_matrix(*q)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)


class TestTablePlane(unittest.TestCase):

    def test_plane_fit_and_projection(self):
        rng = np.random.default_rng(1)
        pts = np.column_stack([rng.uniform(0.2, 0.35, 40),
                               rng.uniform(-0.15, 0.15, 40),
                               np.full(40, 0.02)])
        plane = fit_plane_ransac(pts, threshold=0.005)
        self.assertGreater(plane[2], 0.9, "법선이 위를 향해야 한다")
        for p in pts[:5]:
            self.assertAlmostEqual(height_above_plane(p, plane), 0.0, places=6)
            np.testing.assert_allclose(project_onto_plane(p, plane), p, atol=1e-6)

    def test_height_above_plane_sign(self):
        plane = np.array([0.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(height_above_plane([0.3, 0.0, 0.05], plane), 0.05)
        self.assertAlmostEqual(height_above_plane([0.3, 0.0, -0.01], plane), -0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
