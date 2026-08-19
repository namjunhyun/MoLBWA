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
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "..", "ros2_ws", "src", "ocams_ros2", "config")

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


# ★ 어느 파일을 읽느냐가 중요하다 (2026-08-19 수정).
#   left.yaml / right.yaml          : 2026-07-14 ROS camera_calibration 산출물.
#                                     이미 rectify된 영상을 다시 캘리브한 "순환 캘리브레이션"이라
#                                     epipolar 정렬이 어긋나 있었다 (fx=433.55, cx=470.35).
#   left_opencv.yaml / right_opencv.yaml : 2026-08-11 raw 영상 기준 Kalibr 재캘리브레이션
#                                     (fx=484.71, cx=326.84). commit 23ff90e. ← 이게 정본.
# 예전엔 여기서 left.yaml을 읽고 있었다. 그래서 RECTIFIED_K가 ORB-SLAM3가 실제로 쓰는
# intrinsics와 달랐고, 역투영 결과가 0.7m 거리에서 횡방향 20cm 이상 틀어졌다.
CALIB_BASENAME = "{side}_opencv.yaml"

# 이 K가 반드시 일치해야 하는 상대 — ORB-SLAM3가 pose를 만들 때 쓰는 설정.
# 둘이 어긋나면 "pose의 좌표계"와 "역투영의 좌표계"가 달라져서 조용히 틀린 3D점이 나온다.
ORBSLAM3_YAML = os.path.join(CONFIG_DIR, "orbslam3_stereo_inertial_oCamS.yaml")


def _load_opencv_yaml(path):
    """OpenCV FileStorage 포맷(%YAML:1.0 + !!opencv-matrix) 로더.

    yaml.safe_load는 !!opencv-matrix 태그를 못 읽는다 — cv2.FileStorage를 쓴다.
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"캘리브레이션 파일을 못 읽음: {path}")
    try:
        out = {}
        for key in ("camera_matrix", "distortion_coefficients",
                    "rectification_matrix", "projection_matrix"):
            node = fs.getNode(key)
            if node.empty():
                raise KeyError(f"{path} 에 {key} 가 없다")
            out[key] = np.asarray(node.mat(), dtype=np.float64)
        for key in ("image_width", "image_height"):
            node = fs.getNode(key)
            out[key] = IMAGE_WIDTH if node.empty() and key.endswith("width") else (
                IMAGE_HEIGHT if node.empty() else int(node.real()))
        return out
    finally:
        fs.release()


class StereoCalib:
    """left/right 각각의 camera_matrix, distortion, R(rectification), P(projection) 보관."""

    def __init__(self, side):
        path = os.path.join(CONFIG_DIR, CALIB_BASENAME.format(side=side))
        d = _load_opencv_yaml(path)
        self.path = path
        self.width = d["image_width"]
        self.height = d["image_height"]
        self.K = d["camera_matrix"].reshape(3, 3)
        self.dist = d["distortion_coefficients"].reshape(1, -1)
        self.R = d["rectification_matrix"].reshape(3, 3)
        self.P = d["projection_matrix"].reshape(3, 4)


LEFT = StereoCalib("left")
RIGHT = StereoCalib("right")

# rectified 좌표계의 실제 intrinsics = P[:3,:3].
# stereoRectify 특성상 좌/우 fx,fy,cy는 같아야 함(D=fx*baseline/disparity 공식이 그걸 전제로 함).
assert np.allclose(LEFT.P[0, 0], RIGHT.P[0, 0]), "left/right fx가 다르면 stereoRectify 결과가 아님"
assert np.allclose(LEFT.P[1, 2], RIGHT.P[1, 2]), "left/right cy가 다르면 스캔라인 정렬이 안 된 것"
RECTIFIED_K = LEFT.P[:3, :3]

# baseline: P_right[0,3] = -fx * baseline (OpenCV/ROS 관례)
BASELINE_M = -RIGHT.P[0, 3] / RIGHT.P[0, 0]


def _orbslam3_intrinsics(path=ORBSLAM3_YAML):
    """ORB-SLAM3 설정 yaml에서 Camera1.fx/fy/cx/cy를 읽는다. 없으면 None."""
    if not os.path.exists(path):
        return None
    want = ("Camera1.fx", "Camera1.fy", "Camera1.cx", "Camera1.cy")
    got = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            for key in want:
                if line.startswith(key + ":"):
                    try:
                        got[key] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
    if len(got) != len(want):
        return None
    return np.array([[got["Camera1.fx"], 0.0, got["Camera1.cx"]],
                     [0.0, got["Camera1.fy"], got["Camera1.cy"]],
                     [0.0, 0.0, 1.0]])


def check_consistency(tol_px=0.5, raise_on_mismatch=True):
    """RECTIFIED_K가 ORB-SLAM3 설정과 같은지 확인한다.

    SLAM pose T_WS와 역투영 K가 다른 캘리브레이션에서 나오면, 어떤 단계도 에러를 내지
    않으면서 3D점만 조용히 틀린다 (2026-08-19 이전에 실제로 그 상태였다).
    그래서 import 시점에 한 번 자동으로 검사한다.
    """
    K_slam = _orbslam3_intrinsics()
    if K_slam is None:
        return None
    diff = np.abs(RECTIFIED_K - K_slam)
    if diff.max() > tol_px:
        msg = ("RECTIFIED_K가 ORB-SLAM3 설정과 다르다 — 역투영 좌표계가 pose와 안 맞는다.\n"
               f"  {LEFT.path}\n    {np.round(RECTIFIED_K, 3).tolist()}\n"
               f"  {ORBSLAM3_YAML}\n    {np.round(K_slam, 3).tolist()}\n"
               f"  최대 차이 {diff.max():.2f}px — 두 파일을 같은 캘리브레이션으로 맞출 것.")
        if raise_on_mismatch:
            raise ValueError(msg)
        return msg
    return None


check_consistency()


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
