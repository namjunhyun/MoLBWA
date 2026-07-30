#!/usr/bin/env python3
"""
oCamS-1MGN-U 스테레오 캘리브레이션 로더 + rectify 맵.

ros2_ws/src/ocams_ros2/config/{left,right}_opencv.yaml에 이미 stereo rectification까지
끝난 camera_matrix/distortion/rectification_matrix/projection_matrix가 들어있다
(ROS camera_calibration으로 생성됨, 640x480 기준). 여기선 그 값으로 cv2 remap 맵만 만든다
— stereoRectify를 다시 돌릴 필요 없음.

중요: gaze_on_scene.py는 지금까지 raw(왜곡 보정 전) 이미지 위에서 다점 캘리브레이션(R, fx)을
했다. 이 rectify를 거친 이미지 기준으로 바꿔야, fusion.py의 K(=PROJECTION_MATRIX)와
시선 픽셀(u,v)의 좌표계가 서로 맞아떨어진다. 안 그러면 스테레오 깊이(D)를 구해도 엉뚱한
픽셀의 깊이를 시선 픽셀에 갖다붙이는 꼴이 된다.
"""
import os
import yaml
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "..", "ros2_ws", "src", "ocams_ros2", "config")

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _as_np(d, rows, cols):
    return np.array(d["data"], dtype=np.float64).reshape(rows, cols)


class StereoCalib:
    """left/right 각각의 camera_matrix, distortion, R(rectification), P(projection) 보관."""

    def __init__(self, side):
        path = os.path.join(CONFIG_DIR, f"{side}.yaml")
        d = _load_yaml(path)
        self.width = d["image_width"]
        self.height = d["image_height"]
        self.K = _as_np(d["camera_matrix"], 3, 3)
        self.dist = _as_np(d["distortion_coefficients"], 1, 5)
        self.R = _as_np(d["rectification_matrix"], 3, 3)
        self.P = _as_np(d["projection_matrix"], 3, 4)


LEFT = StereoCalib("left")
RIGHT = StereoCalib("right")

# rectified 좌표계의 실제 intrinsics = P[:3,:3].
# stereoRectify 특성상 좌/우 fx,fy,cy는 같아야 함(D=fx*baseline/disparity 공식이 그걸 전제로 함).
assert np.allclose(LEFT.P[0, 0], RIGHT.P[0, 0]), "left/right fx가 다르면 stereoRectify 결과가 아님"
assert np.allclose(LEFT.P[1, 2], RIGHT.P[1, 2]), "left/right cy가 다르면 스캔라인 정렬이 안 된 것"
RECTIFIED_K = LEFT.P[:3, :3]

# baseline: P_right[0,3] = -fx * baseline (OpenCV/ROS 관례)
BASELINE_M = -RIGHT.P[0, 3] / RIGHT.P[0, 0]


def build_rectify_maps():
    """left/right 각각 (map1, map2) 반환 — cv2.remap(raw_img, map1, map2, cv2.INTER_LINEAR)로 사용."""
    size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    left_maps = cv2.initUndistortRectifyMap(
        LEFT.K, LEFT.dist, LEFT.R, LEFT.P, size, cv2.CV_32FC1)
    right_maps = cv2.initUndistortRectifyMap(
        RIGHT.K, RIGHT.dist, RIGHT.R, RIGHT.P, size, cv2.CV_32FC1)
    return left_maps, right_maps


if __name__ == "__main__":
    print(f"[ocams_calib] rectified K:\n{RECTIFIED_K}")
    print(f"[ocams_calib] baseline = {BASELINE_M*100:.2f} cm")
    left_maps, right_maps = build_rectify_maps()
    for name, (m1, m2) in (("left", left_maps), ("right", right_maps)):
        print(f"[ocams_calib] {name} map1 shape={m1.shape} dtype={m1.dtype}, "
              f"map2 shape={m2.shape} dtype={m2.dtype}")
    print("[ocams_calib] OK — 맵 생성 확인 (실카메라 없이 형태만 검증)")
