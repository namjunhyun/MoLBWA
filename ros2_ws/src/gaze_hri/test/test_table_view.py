#!/usr/bin/env python3
"""캘리브레이션 절차 / 색 픽커 / 물체 검출 검증.

ROS도 카메라도 없이 돈다. 실기에서 시간이 가장 많이 드는 부분이라
로직이 맞는지 미리 확인해 둔다.
"""

import math
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gaze_hri.kinematics import (TOPDOWN_CALIB_XY, ArmGeometry,  # noqa: E402
                                 forward_kinematics)
from gaze_hri.table_view import (HomographyCalibration,  # noqa: E402
                                 ObjectDetector, TableHomography,
                                 hsv_range_from_pixel)


class _Log:
    def warn(self, *a):
        pass

    def info(self, *a):
        pass


def _camera(tilt_deg=15, pos=(0.27, -0.35, 0.95), f=600.0, c=(320.0, 240.0)):
    """탑다운 카메라를 흉내내는 투영 함수."""
    t = math.radians(tilt_deg)
    R = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)],
                  [0, math.sin(t), math.cos(t)]])
    Rz = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
    Rwc = R @ Rz
    C = np.array(pos, dtype=float)

    def project(p):
        pc = Rwc @ (np.asarray(p, float) - C)
        return f * pc[0] / pc[2] + c[0], f * pc[1] / pc[2] + c[1]
    return project


class TestCalibrationFlow(unittest.TestCase):

    def setUp(self):
        self.geo = ArmGeometry()
        self.calib = HomographyCalibration(self.geo, table_z=0.0,
                                           calib_height=0.005, logger=_Log())

    def test_builds_reachable_poses_at_one_height(self):
        n = self.calib.build(TOPDOWN_CALIB_XY)
        self.assertEqual(n, len(TOPDOWN_CALIB_XY), "도달 못 하는 지점이 있다")
        zs = [forward_kinematics(q, self.geo)[2] for q in self.calib.poses]
        self.assertLess(max(zs) - min(zs), 1e-6, "캘리브 점 높이가 흩어져 있다")

    def test_click_flow_and_undo(self):
        self.calib.build(TOPDOWN_CALIB_XY)
        total = self.calib.total
        self.assertFalse(self.calib.done)
        self.calib.add_click(100, 100)
        self.assertEqual(self.calib.index, 1)
        self.calib.undo()
        self.assertEqual(self.calib.index, 0)
        self.assertEqual(len(self.calib.pixels), 0)
        self.assertFalse(self.calib.undo(), "빈 상태에서 취소는 실패해야 한다")
        for i in range(total):
            self.calib.add_click(100 + i, 100)
        self.assertTrue(self.calib.done)
        self.assertFalse(self.calib.add_click(1, 1), "끝난 뒤 추가되면 안 된다")

    def test_solve_recovers_table_coordinates(self):
        """가상 카메라로 클릭을 흉내내면 테이블 좌표가 복원돼야 한다."""
        self.calib.build(TOPDOWN_CALIB_XY)
        project = _camera()
        for (x, y) in self.calib.targets:
            u, v = project([x, y, 0.005])
            self.calib.add_click(u, v)
        result, why = self.calib.solve()
        self.assertIsNotNone(result, why)
        H, err_mm, n = result
        self.assertLess(err_mm, 1.0, "재투영 오차가 크다")

        # 실제 테이블면 여러 지점에서 오차 확인
        homo = TableHomography()
        homo.set(H)
        worst = 0.0
        for x in np.arange(0.20, 0.33, 0.02):
            for y in np.arange(-0.15, 0.16, 0.05):
                u, v = project([x, y, 0.0])
                got = homo.to_robot(u, v)
                worst = max(worst, math.hypot(got[0] - x, got[1] - y))
        self.assertLess(worst, 0.006,
                        f"테이블면 오차가 {worst * 1000:.1f}mm 로 크다")

    def test_solve_rejects_too_few_points(self):
        self.calib.build(TOPDOWN_CALIB_XY)
        for i in range(3):
            self.calib.add_click(100 + i * 10, 100)
        result, why = self.calib.solve()
        self.assertIsNone(result)
        self.assertIn("4", why)

    def test_ransac_threshold_rejects_misclick(self):
        """한 점을 크게 잘못 클릭하면 이상치로 걸러져야 한다."""
        self.calib.build(TOPDOWN_CALIB_XY)
        project = _camera()
        for i, (x, y) in enumerate(self.calib.targets):
            u, v = project([x, y, 0.005])
            if i == 2:
                u += 60                      # 손이 미끄러진 상황
            self.calib.add_click(u, v)
        result, why = self.calib.solve()
        self.assertIsNotNone(result, why)
        _, _, inliers = result
        self.assertLess(inliers, self.calib.total, "이상치를 못 걸렀다")

    def test_save_and_load_roundtrip(self):
        self.calib.build(TOPDOWN_CALIB_XY)
        project = _camera()
        for (x, y) in self.calib.targets:
            self.calib.add_click(*project([x, y, 0.005]))
        self.calib.solve()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "homography.yaml")
            self.assertTrue(self.calib.save(path))
            homo = TableHomography()
            ok, why = homo.load(path)
            self.assertTrue(ok, why)
            np.testing.assert_allclose(homo.H, self.calib.result[0])
            self.assertIsNotNone(homo.reproj_error_mm)

    def test_to_pixel_is_inverse_of_to_robot(self):
        self.calib.build(TOPDOWN_CALIB_XY)
        project = _camera()
        for (x, y) in self.calib.targets:
            self.calib.add_click(*project([x, y, 0.005]))
        H, _, _ = self.calib.solve()[0]
        homo = TableHomography()
        homo.set(H)
        for (u, v) in [(200.0, 300.0), (400.0, 250.0), (320.0, 400.0)]:
            x, y = homo.to_robot(u, v)
            bu, bv = homo.to_pixel(x, y)
            self.assertAlmostEqual(bu, u, places=6)
            self.assertAlmostEqual(bv, v, places=6)


class TestColorPicker(unittest.TestCase):

    def test_picks_range_containing_clicked_color(self):
        img = np.zeros((100, 100, 3), np.uint8)
        img[:, :] = (40, 40, 200)                    # BGR 붉은색
        lower, upper, _ = hsv_range_from_pixel(img, 50, 50)
        det = ObjectDetector(lower, upper, min_area=10)
        self.assertGreater(int(det.mask(img).sum()), 0, "고른 색이 안 잡힌다")

    def test_wraparound_red_is_handled(self):
        """빨강은 색상환 양 끝에 걸친다. 두 구간을 합쳐야 다 잡힌다."""
        img = np.zeros((60, 120, 3), np.uint8)
        hsv = np.zeros((60, 120, 3), np.uint8)
        hsv[:, :60] = (2, 220, 220)                  # hue 2  (0 쪽)
        hsv[:, 60:] = (178, 220, 220)                # hue 178 (180 쪽)
        img[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        det = ObjectDetector([170, 100, 100], [10, 255, 255], min_area=10)
        mask = det.mask(img)
        self.assertGreater(int(mask[:, :60].sum()), 0, "hue 0 쪽을 놓쳤다")
        self.assertGreater(int(mask[:, 60:].sum()), 0, "hue 180 쪽을 놓쳤다")


class TestObjectDetector(unittest.TestCase):

    def test_detects_and_sorts_by_robot_y(self):
        calib = HomographyCalibration(ArmGeometry(), calib_height=0.005,
                                      logger=_Log())
        calib.build(TOPDOWN_CALIB_XY)
        project = _camera()
        for (x, y) in calib.targets:
            calib.add_click(*project([x, y, 0.005]))
        H, _, _ = calib.solve()[0]
        homo = TableHomography()
        homo.set(H)

        img = np.zeros((480, 640, 3), np.uint8)
        centers = []
        for (x, y) in [(0.27, 0.12), (0.27, -0.12), (0.27, 0.0)]:
            u, v = project([x, y, 0.0])
            cv2.circle(img, (int(u), int(v)), 22, (40, 40, 200), -1)
            centers.append((x, y))
        det = ObjectDetector([0, 120, 80], [12, 255, 255], min_area=200)
        found = det.detect(img, homo)
        self.assertEqual(len(found), 3, "컵 3개를 못 찾았다")
        ys = [xy[1] for (_, xy, _) in found]
        self.assertEqual(ys, sorted(ys), "로봇 y 기준으로 정렬돼야 한다")
        for (_, (gx, gy), _) in found:
            close = min(math.hypot(gx - cx, gy - cy) for (cx, cy) in centers)
            self.assertLess(close, 0.01, "복원된 위치가 부정확하다")

    def test_no_homography_means_no_detection(self):
        det = ObjectDetector([0, 120, 80], [12, 255, 255])
        img = np.zeros((100, 100, 3), np.uint8)
        self.assertEqual(det.detect(img, TableHomography()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
