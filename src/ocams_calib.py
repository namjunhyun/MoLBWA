#!/usr/bin/env python3
"""
oCamS-1MGN-U 스테레오 캘리브레이션 로더 + rectify 맵.

ros2_ws/src/ocams_ros2/config/{left,right}_opencv.yaml에 이미 stereo rectification까지
끝난 camera_matrix/distortion/rectification_matrix/projection_matrix가 들어있다.
이 값은 2026-08-11에 Kalibr로 재캘리브레이션되었다(docs/11_camera_imu_calibration.md,
SLAM 리셋 루프의 근본 원인이었던 옛 rectification을 대체함). 여기선 그 값으로 cv2 remap
맵만 만든다 — stereoRectify를 다시 돌릴 필요 없음.

주의: 같은 디렉토리의 {left,right}.yaml(2026-07-14 생성, 이후 갱신 안 됨)은 재캘리브레이션
전의 옛 값이다 — 이름이 비슷해서 헷갈리기 쉬운데 절대 이 파일을 읽으면 안 된다. 2026-08-19
세션에서 이 모듈이 실수로 {left,right}.yaml을 읽고 있던 것이 발견되어(옛 rectification ->
StereoSGBM 깊이가 동일 평면에서 0.52~1.38m로 들쭉날쭉하게 나옴) {left,right}_opencv.yaml로
고쳤다. {left,right}_opencv.yaml은 OpenCV FileStorage YAML(`%YAML:1.0` + `!!opencv-matrix`)
포맷이라 PyYAML로는 못 읽는다 — cv2.FileStorage로 읽는다(C++ 드라이버 ocams_stereo_imu_node.cpp
의 setupRectification()과 동일한 방식).

중요: gaze_on_scene.py는 지금까지 raw(왜곡 보정 전) 이미지 위에서 다점 캘리브레이션(R, fx)을
했다. 이 rectify를 거친 이미지 기준으로 바꿔야, fusion.py의 K(=PROJECTION_MATRIX)와
시선 픽셀(u,v)의 좌표계가 서로 맞아떨어진다. 안 그러면 스테레오 깊이(D)를 구해도 엉뚱한
픽셀의 깊이를 시선 픽셀에 갖다붙이는 꼴이 된다.
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "..", "ros2_ws", "src", "ocams_ros2", "config")

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


class StereoCalib:
    """left/right 각각의 camera_matrix, distortion, R(rectification), P(projection) 보관."""

    def __init__(self, side):
        path = os.path.join(CONFIG_DIR, f"{side}_opencv.yaml")
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"캘리브레이션 파일을 못 엶: {path}")
        self.width = int(fs.getNode("image_width").real())
        self.height = int(fs.getNode("image_height").real())
        self.K = fs.getNode("camera_matrix").mat()
        self.dist = fs.getNode("distortion_coefficients").mat()
        self.R = fs.getNode("rectification_matrix").mat()
        self.P = fs.getNode("projection_matrix").mat()
        fs.release()


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
