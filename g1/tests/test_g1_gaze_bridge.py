"""g1_gaze_bridge 순수 함수: 부위 판정, 방위각, bin, config. 카메라·YOLO·태그 검출기 없이 돈다."""
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

from g1_gaze_bridge import bearing_deg, nearest_bin, pick_part  # noqa: E402

TORSO = ["g1_torso", 0.9, 100, 100, 300, 400]
HAND = ["g1_hand", 0.5, 250, 300, 290, 340]       # 몸통 박스 안
FACE = ["g1_face", 0.8, 150, 20, 250, 100]


def test_pick_part_hand_inside_torso_wins():
    assert pick_part([TORSO, HAND, FACE], (270, 320)) == "g1_hand"


def test_pick_part_torso_only_region():
    assert pick_part([TORSO, HAND, FACE], (120, 200)) == "g1_torso"


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


def test_bearing_behind_wobble_both_bin_180():
    # Review Focus #1: 뒤에서 179/-179 로 흔들려도 둘 다 180.
    a = math.radians(179.0)
    b = math.radians(-179.0)
    for ang in (a, b):
        deg = bearing_deg(_T_hc_torso((2 * math.cos(ang), 2 * math.sin(ang), 0.5)))
        assert nearest_bin(deg) == 180


@pytest.mark.parametrize("deg,b", [
    (22.4, 0), (22.6, 45), (-22.4, 0), (-22.6, -45),
    (179, 180), (-179, 180), (180, 180), (-180, 180),
    (-158, 180), (-157, -135), (158, 180), (157, 135),
])
def test_nearest_bin(deg, b):
    assert nearest_bin(deg) == b


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
    assert set(cfg["yolo"]["classes"]) == {"g1_face", "g1_hand", "g1_torso"}


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
    assert nearest_bin(got) == nearest_bin(bear)
