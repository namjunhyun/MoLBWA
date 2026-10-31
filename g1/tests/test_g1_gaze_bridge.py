"""g1_gaze_bridge 순수 함수: 부위 판정, 방위각, plan, config. 카메라·YOLO·태그 검출기 없이 돈다."""
import math
import os
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.dirname(HERE)
sys.path.insert(0, G1)
sys.path.insert(0, os.path.join(os.path.dirname(G1), "arm"))

from g1_gaze_bridge import bearing_deg, pick_part  # noqa: E402
from g1_protocol import plan  # noqa: E402

TORSO = ["g1_torso", 0.9, 100, 100, 300, 400]
HAND = ["g1_hand", 0.5, 250, 300, 290, 340]       # 몸통 박스 안
FACE = ["g1_face", 0.8, 150, 20, 250, 100]


def test_pick_part_hand_inside_torso_wins():
    assert pick_part([TORSO, HAND, FACE], (270, 320)) == "g1_hand"


def test_pick_part_torso_ignored():
    # 몸통 응시는 판정 안 함 (2026-10-30: 시나리오는 머리·손뿐, 젯슨 라이브러리에 open_arms 없음)
    assert pick_part([TORSO, HAND, FACE], (120, 200)) is None


def test_pick_part_same_class_higher_conf():
    # 같은 클래스 둘 다 포함 -> 라벨은 같다. 그래도 conf 순 정렬이 깨지지 않는지 확인.
    assert pick_part([["g1_face", 0.3, 0, 0, 50, 50], ["g1_face", 0.9, 10, 10, 60, 60]],
                     (20, 20)) == "g1_face"


def test_pick_part_outside_none_and_empty():
    assert pick_part([TORSO, HAND, FACE], (500, 500)) is None
    assert pick_part([TORSO], None) is None
    assert pick_part([], (120, 200)) is None


def test_pick_part_ignores_unknown_class():
    assert pick_part([["person", 0.99, 0, 0, 1000, 1000]], (10, 10)) is None


def _T_hc_torso(p_cam_in_torso):
    """헤드캠이 torso 좌표 p 에서 torso 원점을 바라보는 자세 -> T_hc_torso (p_hc = T @ p_torso)."""
    p = np.asarray(p_cam_in_torso, float)
    z = -p / np.linalg.norm(p)                     # 카메라 광축 = torso 원점 쪽
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T_torso_hc = np.eye(4)
    T_torso_hc[:3, :3] = np.stack([x, y, z], axis=1)
    T_torso_hc[:3, 3] = p
    return np.linalg.inv(T_torso_hc)


@pytest.mark.parametrize("p,expect", [
    ((2, 0, 0.5), 0.0),       # 정면
    ((0, 2, 0.5), 90.0),      # 왼쪽 (+y)
    ((-2, 0, 0.5), 180.0),    # 뒤
    ((0, -2, 0.5), -90.0),    # 오른쪽
    ((1.4, 1.4, 0.3), 45.0),
])
def test_bearing_deg(p, expect):
    assert bearing_deg(_T_hc_torso(p)) == pytest.approx(expect, abs=1e-6)


def test_bearing_behind_wobble_same_plan():
    # Review Focus #1: 뒤에서 179/-179 로 흔들려도 둘 다 turn_l180 + wave.
    for d in (179.0, -179.0):
        ang = math.radians(d)
        deg = bearing_deg(_T_hc_torso((2 * math.cos(ang), 2 * math.sin(ang), 0.5)))
        assert plan("g1_face", deg) == ["turn_l180", "wave"]


def test_config_loads_four_tag_bundle():
    from anchor import build_bundle_obj_pts
    cfg = yaml.safe_load(open(os.path.join(G1, "config.yaml")))
    obj, _, _, _ = build_bundle_obj_pts(cfg["anchor"])
    assert sorted(obj) == [10, 11, 12, 13]
    assert cfg["anchor"]["min_tags_for_latch"] == 1
    # 각 태그 법선(right x up)이 몸통 바깥을 향해야 PnP 가 보는 쪽 면이 맞다.
    for t in cfg["anchor"]["bundle"]:
        n = np.cross(t["right"], t["up"])
        pos = np.asarray(t["pos"], float)
        assert n @ np.r_[pos[:2], 0.0] > 0, t["id"]
        c = obj[t["id"]].mean(axis=0)
        assert np.allclose(c, pos)
    assert set(cfg["yolo"]["classes"]) == {"g1_face", "g1_hand"}   # 몸통 없음, 꽃은 obj_yolo(YOLO-World)


@pytest.mark.parametrize("tag_id,bear", [(10, 10.0), (11, 80.0), (12, -100.0), (13, 170.0), (13, -170.0)])
def test_single_tag_pnp_gives_bearing(tag_id, bear):
    """Review Focus #4: 태그 한 장만 보여도 PnP -> bearing 이 나온다. 그 면 앞 2 m 에서 본 코너를
    만들어 TagBundleDetector 와 같은 solvePnP(ITERATIVE) 를 돌린다 (apriltag 검출만 빼고 같은 경로)."""
    cv2 = pytest.importorskip("cv2")
    from anchor import build_bundle_obj_pts
    cfg = yaml.safe_load(open(os.path.join(G1, "config.yaml")))
    obj = build_bundle_obj_pts(cfg["anchor"])[0][tag_id]
    a = math.radians(bear)
    T = _T_hc_torso((2 * math.cos(a), 2 * math.sin(a), 0.5))
    K = np.array([[460.0, 0, 320], [0, 460.0, 240], [0, 0, 1]])
    p_hc = (T[:3, :3] @ obj.T).T + T[:3, 3]
    img = (K @ p_hc.T).T
    img = img[:, :2] / img[:, 2:]
    # pupil_apriltags 코너 순서 (우상, 좌상, 좌하, 우하) 가 화면에서도 그렇게 보여야 면 방향이 맞다
    assert img[0, 0] > img[1, 0] and img[1, 1] < img[2, 1] and img[3, 0] > img[2, 0]
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok
    R, _ = cv2.Rodrigues(rvec)
    T_est = np.eye(4)
    T_est[:3, :3], T_est[:3, 3] = R, tvec.ravel()
    got = bearing_deg(T_est)
    assert abs((got - bear + 180) % 360 - 180) < 2.0
    assert plan("g1_face", got) == plan("g1_face", bear)


def _cam_at(x, y, yaw_deg):
    """torso 평면 (x, y) 에 서서 yaw 방향을 보는 헤드캠(OpenCV: z 앞, x 오른쪽, y 아래)의 T_hc_torso."""
    import numpy as np
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    z = np.array([c, s, 0.0]); xr = np.array([s, -c, 0.0]); yd = np.array([0, 0, -1.0])
    T_tc = np.eye(4); T_tc[:3, :3] = np.stack([xr, yd, z], 1); T_tc[:3, 3] = [x, y, 1.0]
    return np.linalg.inv(T_tc)


def test_object_bearing_center_and_offsets():
    import numpy as np
    from g1_gaze_bridge import object_bearing_deg
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    # 착용자가 지원이 정면 2 m 에서 지원이 쪽(−x)을 보며 화면 중앙을 응시, 1 m 앞 점 → 지원이 정면 1 m → 0°
    T = _cam_at(2.0, 0.0, 180.0)
    assert abs(object_bearing_deg(T, (320, 240), K, 1.0)) < 1e-6
    # 화면 오른쪽(착용자 기준 오른쪽 = 지원이 기준 왼쪽 +y) → 양수
    assert object_bearing_deg(T, (500, 240), K, 1.0) > 10
    # 착용자가 지원이 왼쪽(+y 2 m)에 서서 정면(+x)을 보고 3 m 앞 점 → torso 기준 (3, 2) → atan2(2,3)
    T2 = _cam_at(0.0, 2.0, 0.0)
    assert abs(object_bearing_deg(T2, (320, 240), K, 3.0) - np.degrees(np.arctan2(2, 3))) < 1e-6


def test_draw_overlay_marks_boxes_and_gaze_without_touching_input():
    cv2 = pytest.importorskip("cv2")
    from g1_gaze_bridge import draw_overlay
    frame = np.zeros((480, 640, 3), np.uint8)
    obj = ["bottle", 0.81, 400, 200, 440, 320]
    out = draw_overlay(frame, [TORSO, HAND], [obj], (270, 320), "g1_hand", "bottle", 12.0)
    assert out.shape == frame.shape and frame.sum() == 0          # 원본은 그대로(복사본에 그림)
    assert out[300, 250].any() and not out[100, 100].any()         # 손 박스는 그리고 몸통은 안 그림
    assert out[200, 400].any()                                     # 물체 박스 테두리
    assert out[320, 270].any() and out[320 - 28, 270].any()        # 시선 십자 + 원(반지름 28)
    assert out[:40].any()                                          # 위쪽 판정 문구


def test_draw_overlay_no_gaze_no_dets():
    pytest.importorskip("cv2")
    from g1_gaze_bridge import draw_overlay
    out = draw_overlay(np.zeros((480, 640, 3), np.uint8), [], [], None, None, None, None)
    assert out.shape == (480, 640, 3)


def test_hold_dets_keeps_missing_class_briefly():
    from g1_gaze_bridge import hold_dets
    held = {}
    assert hold_dets(held, [FACE, HAND], 0.0) == [FACE, HAND]
    assert sorted(hold_dets(held, [HAND], 0.3)) == sorted([FACE, HAND])   # 얼굴 놓쳐도 0.5초는 유지
    assert hold_dets(held, [HAND], 0.9) == [HAND]                         # 넘으면 사라짐
    moved = ["g1_hand", 0.6, 260, 300, 300, 340]
    assert hold_dets(held, [moved], 1.0) == [moved]                       # 새로 보이면 최신 박스로 교체


def test_pick_part_margin_absorbs_gaze_error():
    # 시선 오차(캘리브 16px)만큼 박스 밖에 떨어져도 그 부위로 본다
    assert pick_part([FACE], (150 - 15, 60)) == "g1_face"
    assert pick_part([FACE], (150 - 15, 60), margin=0) is None
    assert pick_part([FACE], (150 - 28, 60)) == "g1_face"           # 30px 까지
    assert pick_part([FACE], (150 - 40, 60)) is None


def test_hold_value_keeps_last_bearing_briefly():
    from g1_gaze_bridge import hold_value
    st = {}
    assert hold_value(st, -16.0, 0.0, 2.0) == -16.0
    assert hold_value(st, None, 1.5, 2.0) == -16.0     # 태그 잠깐 놓쳐도 직전 방위
    assert hold_value(st, None, 2.5, 2.0) is None      # 오래되면 버린다(0° 로 속이지 않게)
    assert hold_value(st, 30.0, 3.0, 2.0) == 30.0


def test_flower_box_drawn_without_margin_and_not_a_part():
    pytest.importorskip("cv2")
    from g1_gaze_bridge import draw_overlay
    fl = ["flower", 0.9, 300, 200, 360, 300]
    assert pick_part([fl], (330, 250)) is None                  # 꽃은 부위 판정 대상이 아니다(물체 YOLO 가 맡음)
    out = draw_overlay(np.zeros((480, 640, 3), np.uint8), [fl], [], None, None, None, None)
    assert out[250, 300].any()                                  # 박스 테두리
    assert not out[250, 300 - 20].any()                         # 판정 여백(굵은 선)은 안 그림


def test_config_obj_yolo_is_world_with_flower_prompts():
    import yaml
    from obj_gaze_detector import OBJ_TO_KIND
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))
    o = cfg["obj_yolo"]
    assert "world" in o["model"]                                   # 2026-10-31: 먼 꽃다발 COCO 57% → World 68%
    assert o["classes"] and all(OBJ_TO_KIND.get(c) == "plant" for c in o["classes"])


def test_obj_margin_same_in_judgement_and_display():
    import g1_gaze_bridge
    import obj_gaze_detector
    assert g1_gaze_bridge.OBJ_MARGIN_PX == obj_gaze_detector.OBJ_MARGIN_PX   # 그린 굵은 선 = 실제 판정 영역


def test_scaled_K_for_upscaled_tag_detection():
    # 2026-10-31: 태그가 13~14px 이면 2배 확대 재검출 21%→61~69%. 확대 영상엔 fx·fy·cx·cy 도 2배인 K 를 쓴다
    from g1_gaze_bridge import scaled_K
    K = np.array([[461.24, 0, 320], [0, 461.24, 240], [0, 0, 1.0]])
    K2 = scaled_K(K, 2)
    assert np.allclose(K2, [[922.48, 0, 640], [0, 922.48, 480], [0, 0, 1]]) and K[0, 0] == 461.24
