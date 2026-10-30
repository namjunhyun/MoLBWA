#!/usr/bin/env python3
"""통합 GUI 화면 구성 검증.

ROS도 카메라도 없이 돈다. 화면이 예외 없이 그려지는지, 레이아웃이 맞는지,
한글 폰트가 없는 환경에서도 영문으로 정상 동작하는지 본다.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gaze_hri.hud import Hud                                   # noqa: E402
from gaze_hri.panel_render import (FOOTER_H, HEADER_H, PanelState,  # noqa: E402
                                   _DemoHomography, _demo_frame, render)


def _state(**kw):
    frame, cups, contours = _demo_frame()
    homo = _DemoHomography()
    objects = [((float(u), float(v)), homo.to_robot(u, v), c)
               for (u, v), c in zip(cups, contours)]
    st = PanelState(objects=objects, calibrated=True, reproj_error_mm=4.2,
                    fps=30.0)
    for k, v in kw.items():
        setattr(st, k, v)
    return frame, st, homo


class TestRender(unittest.TestCase):

    def test_canvas_geometry(self):
        """캔버스는 영상 + 헤더/푸터 + 사이드바 크기여야 한다."""
        frame, st, homo = _state()
        canvas = render(frame, st, Hud(), homo)
        vh, vw = frame.shape[:2]
        self.assertEqual(canvas.shape[0], HEADER_H + vh + FOOTER_H)
        self.assertEqual(canvas.shape[1], vw + st.sidebar_w)
        self.assertEqual(canvas.dtype, np.uint8)

    def test_video_area_is_preserved(self):
        """영상 영역이 캔버스에 그대로 들어가야 한다(오버레이 없는 곳 기준)."""
        frame, st, homo = _state(pointer_px=None, objects=[], tcp=None)
        canvas = render(frame, st, Hud(), homo)
        vh, vw = frame.shape[:2]
        # 좌상단 구석은 오버레이가 닿지 않는 자리
        np.testing.assert_array_equal(canvas[HEADER_H:HEADER_H + 20, 0:20],
                                      frame[0:20, 0:20])

    def test_all_states_render(self):
        """세 가지 상태 모두 예외 없이 그려져야 한다."""
        frame, st, homo = _state()
        for task_state in ("IDLE", "PICK_SELECTED", "EXECUTING"):
            st.task_state = task_state
            st.pointer_px = (250, 270)
            st.pick_target = st.objects[1][1]
            st.place_target = homo.to_robot(250, 410)
            st.target_info = {"snapped": True, "confidence": 0.9}
            st.tcp = (0.30, 0.0, 0.06)
            st.arm_phase = "GRASP"
            canvas = render(frame, st, Hud(), homo)
            self.assertIsNotNone(canvas)

    def test_render_without_any_selection(self):
        """아무것도 선택되지 않은 초기 상태에서도 그려져야 한다."""
        frame, st, homo = _state(pointer_px=None, pick_target=None,
                                 place_target=None, target_info=None, tcp=None)
        self.assertIsNotNone(render(frame, st, Hud(), homo))

    def test_english_fallback(self):
        """한글 폰트가 없는 환경에서도 영문으로 그려져야 한다."""
        hud = Hud()
        hud.korean = False                     # 폰트 없는 환경을 흉내
        frame, st, homo = _state(pointer_px=(250, 270), task_state="EXECUTING",
                                 arm_phase="LIFT", tcp=(0.3, 0.0, 0.06))
        canvas = render(frame, st, hud, homo)
        self.assertIsNotNone(canvas)
        self.assertEqual(hud.label("집을 것", "PICK"), "PICK")

    def test_pipeline_lights_are_cumulative(self):
        """뒤 단계가 켜졌으면 앞 단계도 켜져 보여야 한다.

        동작 중에는 커서가 화면 밖일 수 있는데, 그때 '시선' 단계만 꺼져 있으면
        진행 상황을 잘못 읽게 된다.
        """
        frame, st, homo = _state(task_state="EXECUTING", pointer_px=None,
                                 has_fixation=False, pick_target=None,
                                 place_target=None)
        canvas = render(frame, st, Hud(), homo)
        # 사이드바 첫 번째 단계 점의 색을 직접 확인한다
        x = frame.shape[1] + 18 + 5
        y = HEADER_H + 14 + 21 + 7
        dot = canvas[y, x]
        self.assertGreater(int(dot[1]), 150, "첫 단계 점이 꺼져 있다")


class TestHudPrimitives(unittest.TestCase):

    def test_chip_stays_inside_image(self):
        """칩이 화면 밖으로 나가면 안 된다(경계에서 잘리지 않게)."""
        hud = Hud()
        img = np.zeros((200, 300, 3), np.uint8)
        for xy in [(-50, -50), (295, 195), (0, 0), (150, 100)]:
            x, y, w, h = hud.chip(img, xy, "test", size=12)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, img.shape[1])
            self.assertLessEqual(y + h, img.shape[0])

    def test_panel_clips_to_image(self):
        hud = Hud()
        img = np.zeros((100, 100, 3), np.uint8)
        hud.panel(img, -20, -20, 300, 300)      # 전부 밖으로 나가는 경우
        hud.panel(img, 90, 90, 50, 50)
        self.assertEqual(img.shape, (100, 100, 3))

    def test_text_size_is_positive(self):
        hud = Hud()
        w, h = hud.text_size("집을 것", 14)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
