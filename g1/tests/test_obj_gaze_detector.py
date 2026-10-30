import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from obj_gaze_detector import OBJ_TO_KIND, object_bearing, pick_object  # noqa: E402


def test_pick_object_returns_kind_for_gazed_box():
    dets = [["potted plant", 0.9, 100, 100, 200, 200], ["person", 0.8, 0, 0, 640, 480]]
    assert pick_object(dets, (150, 150)) == ("plant", [100, 100, 200, 200])


def test_pick_object_ignores_unmapped_labels():
    assert pick_object([["person", 0.9, 0, 0, 640, 480]], (320, 240)) is None


def test_pick_object_none_when_gaze_outside_boxes():
    assert pick_object([["vase", 0.9, 100, 100, 200, 200]], (10, 10)) is None


def test_pick_object_none_when_no_gaze():
    assert pick_object([["vase", 0.9, 0, 0, 640, 480]], None) is None


def test_pick_object_prefers_higher_conf():
    dets = [["vase", 0.5, 0, 0, 100, 100], ["potted plant", 0.9, 0, 0, 100, 100]]
    assert pick_object(dets, (50, 50)) == ("plant", [0, 0, 100, 100])


def test_obj_to_kind_values():
    assert OBJ_TO_KIND == {"potted plant": "plant", "vase": "plant"}          # 물병·가족사진·휴대폰 제외(2026-10-30)
    assert pick_object([["cup", 0.9, 0, 0, 640, 480], ["cell phone", 0.9, 0, 0, 640, 480]], (320, 240)) is None


def test_object_bearing_center_pixel_identity_T():
    K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
    # T=I 이면 주점 중심 박스의 광선은 +z 뿐 → torso x=0,y=0 → 방위 0
    b = object_bearing([300, 220, 340, 260], K, 1.5, np.eye(4))
    assert b == 0 and isinstance(b, int)
