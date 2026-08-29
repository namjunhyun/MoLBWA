#!/usr/bin/env python3
"""
docs/09_visualization.md 섹션 4 로깅 스케치 — /orbslam3/pose(SLAM) + 눈 카메라(시선 방향)를
합쳐서 Rerun 3D 뷰에 카메라 pose + 궤적 + 시선 광선을 그린다.

2026-08-29: StereoSGBM 깊이 파이프라인이 실측 검증됐다(docs/12, ocams_calib.py의
tagSize/baseline 재캘리브레이션) — 그래서 이제 시선 픽셀의 실제 깊이 D를 구해
fusion.gaze_point_world(u, v, D, K, T_WS)로 정확한 p_W를 world/gaze/hit(Points3D)로 찍는다.
ocams_stereo_imu_node가 이미 발행 중인 /camera/left, /camera/right(rectified, ORB-SLAM3도
같이 구독하는 바로 그 토픽)를 이 노드도 message_filters로 동기 구독해서 StereoSGBM disparity를
직접 계산한다(oCamS 장치를 이 스크립트가 또 열면 이미 ocams_stereo_imu_node가 물고 있어서
충돌하므로, cv2.VideoCapture로 다시 열지 않고 토픽 구독으로 재사용).

깊이가 아직 안 들어왔거나(캐시 없음) 유효하지 않으면(범위 밖/유효 disparity 부족) 기존
placeholder 광선(GAZE_RAY_LENGTH_M 고정 길이)으로 자동 폴백한다 — 이 광선은
- 원점 = SLAM pose 위치 그대로 (눈-카메라 물리적 오프셋 무시, docs/03의 근사 재사용)
- 방향 = 눈 트래커의 gaze_direction을 SLAM pose 회전으로 world 좌표계로 변환한 것

2026-08-29 (추가): docs/09_visualization.md가 지적한 대로 ORB-SLAM3의 sparse map point는
사람이 보기엔 너무 부실해서(시선이 어느 표면에 닿았는지 알 수 없음), Open3D
ScalableTSDFVolume으로 방 전체의 조밀한 3D 표면을 실시간으로 재구성해서 world/tsdf_map에
Points3D로 띄운다 — docs/10_realsense_tsdf_mapping.md에서 D455로 먼저 검증한 것과 같은
패턴이지만, D455 스크립트(src/d455_tsdf_rerun.py)는 그대로 재사용할 수 없다: 거기는 별도
ORB-SLAM3 실행파일이 pose/depth/color를 파일로 쓰는 방식(파일 폴링 IPC)이고 D455 자체
RGB-D 센서를 쓰는데, 여기는 ROS2 토픽(/orbslam3/pose, /camera/left,right) 구독 + 우리가
StereoSGBM으로 직접 계산한 disparity를 쓴다. oCamS는 컬러가 없어서(흑백만) 왼쪽 영상을
3채널로 복제해 텍스처로 쓴다(진짜 컬러 아님).

주의 — 좌표계: gaze_direction은 지금 눈 카메라 자신의 로컬 좌표계 기준이고, "눈 카메라 광학축 ==
oCamS(SLAM 카메라) 광학축"이라고 가정하고 있다(둘 다 안경에 고정되어 있으니 상대 회전은 있지만
extrinsic 미측정 — IMU-카메라 extrinsic 문제와 같은 종류의 근사). 눈↔oCamS 상대 회전을 실측하면
그 회전을 여기 곱해서 보정해야 함.

사용 (rclpy 때문에 반드시 시스템 python3.12로 실행 — anaconda python3.13에는 rclpy 없음):
  ros2 run ocams_ros2 ocams_stereo_imu_node        (터미널 1)
  ros2 run orbslam3 stereo-inertial <voc> <yaml> false   (터미널 2)
  python3.12 orbslam3_rerun.py                     (터미널 3, 이 스크립트 — 눈 카메라도 이 안에서 염)

rerun-sdk는 python3.12 환경엔 0.20.0으로 고정 설치했다(2026-07-30) — 최신(0.35.0)이 요구하는
numpy>=2가 이 환경의 mediapipe/scipy(다른 ROS2 노드가 씀)를 깨뜨려서, numpy<2를 요구하는
구버전으로 맞춤. anaconda python3.13 쪽(gaze_on_scene.py)은 별개 환경이라 0.35.0 그대로 씀.

카메라를 실제로 들고 움직여야 ORB-SLAM3 VI가 초기화된다("not enough acceleration"은 정상,
가만히 두면 계속 뜸 — notes/2026-07-14_ocams_ros2_slam_live.md 참고). 지금은 IMU-카메라
extrinsic이 identity placeholder라 트래킹 자체가 불안정함(2026-07-30 확인, docs/02 참고) —
시선 광선도 그 위에서 흔들리는 pose를 따라가니 같이 튈 수 있음.
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gaze_on_scene as gos  # noqa: E402  find_cam/open_eye/tracker/make_stereo_matcher/depth_at 재사용
import ocams_calib  # noqa: E402
import fusion  # noqa: E402

GAZE_RAY_LENGTH_M = 2.0  # 깊이 없을 때(폴백)의 광선 길이 placeholder
DEPTH_STALE_SEC = 0.5    # 이보다 오래된 disparity 캐시는 안 씀(스테레오 프레임 끊긴 경우)
# 2026-08-29: 눈 카메라 IR 케이블 파손으로 실측 불가 — 깊이/SLAM 융합 파이프라인만 먼저
# 라이브 검증하려고 정면 고정 시선으로 대체. 진짜 시선 방향이 아님, 눈 카메라 복구되면 제거.
FAKE_GAZE_DIR_NO_EYE = (0.0, 0.0, 1.0)

# docs/10_realsense_tsdf_mapping.md의 D455 실험과 같은 파라미터(방 스케일 실내 공간 기준).
TSDF_VOXEL_SIZE_M = 0.025
TSDF_KEYFRAME_TRANSLATION_M = 0.04   # 이보다 적게 움직였으면 같은 프레임으로 보고 融合 생략
TSDF_KEYFRAME_ROTATION_DEG = 4.0
TSDF_EXTRACT_EVERY = 15              # 이 키프레임마다 한 번씩 point cloud 추출+저장+로깅
TSDF_DEPTH_TRUNC_M = 1.5  # 2026-08-29: 실측 검증은 ~1m까지 정확, 1.5m도 이미 12% 오차 —
                          # 그 이상(D455에서 물려받은 원래 5.0m)은 신뢰 못 할 깊이라 잘라냄
TSDF_OUTPUT_PATH = Path.home() / "molbwa_ocams_tsdf_map.ply"


def rotation_angle_rad(a, b):
    """두 3x3 회전행렬 사이의 각도(라디안) — TSDF 키프레임 선택용."""
    relative = a.T @ b
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


class OrbSlam3RerunBridge(Node):
    def __init__(self):
        super().__init__("orbslam3_rerun_bridge")
        self.trajectory = []
        self.frame_idx = 0
        self.latest_gaze_dir = None  # 눈 카메라 로컬 좌표계, 단위벡터
        self.latest_disparity = None
        self.latest_left = None
        self.latest_disparity_t = 0.0
        self.stereo_matcher = gos.make_stereo_matcher()

        K = ocams_calib.RECTIFIED_K
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=TSDF_VOXEL_SIZE_M,
            sdf_trunc=TSDF_VOXEL_SIZE_M * 4.0,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.tsdf_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        self.tsdf_last_pose = None  # 마지막으로 융합한 T_WS (키프레임 선택용)
        self.tsdf_keyframes = 0

        try:
            self.eye_cap = gos.open_eye(640, 480, 30)
        except (OSError, RuntimeError) as e:
            self.eye_cap = None
            self.get_logger().warn(
                f"눈 카메라 없음 — [FAKE] 정면 고정 시선({FAKE_GAZE_DIR_NO_EYE})으로 대체해서 "
                f"깊이/SLAM 융합 파이프라인만 테스트: {e}")
            self.latest_gaze_dir = np.array(FAKE_GAZE_DIR_NO_EYE, dtype=np.float64)
        self.create_timer(1.0 / 30.0, self.on_eye_timer)

        left_sub = Subscriber(self, Image, "/camera/left")
        right_sub = Subscriber(self, Image, "/camera/right")
        self.stereo_sync = ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=5, slop=0.05)
        self.stereo_sync.registerCallback(self.on_stereo)

        self.create_subscription(PoseStamped, "/orbslam3/pose", self.on_pose, 10)
        self.get_logger().info(
            "orbslam3_rerun_bridge 시작 — /orbslam3/pose + /camera/left,right(깊이용) 구독 "
            "+ 눈 카메라 시선 광선")

    def on_stereo(self, left_msg, right_msg):
        left = np.frombuffer(left_msg.data, dtype=np.uint8).reshape(
            left_msg.height, left_msg.width)
        right = np.frombuffer(right_msg.data, dtype=np.uint8).reshape(
            right_msg.height, right_msg.width)
        self.latest_disparity = self.stereo_matcher.compute(left, right).astype(np.float32) / 16.0
        self.latest_left = left  # TSDF 텍스처용(oCamS는 컬러가 없어 흑백을 그대로 씀)
        self.latest_disparity_t = time.monotonic()

    def on_eye_timer(self):
        if self.eye_cap is None:
            return  # [FAKE] 정면 고정값 유지 (__init__에서 이미 설정, 눈 카메라 없어서 갱신 불가)
        ok, eye = self.eye_cap.read()
        if not ok:
            return
        _ellipse, d = gos.tracker.process_frame(eye)
        if d is not None and np.linalg.norm(d) > 1e-6:
            self.latest_gaze_dir = d / np.linalg.norm(d)
        cv2.waitKey(1)  # 이게 없으면 process_frame 내부의 cv2.imshow 창이 실제로 안 그려짐

    def integrate_tsdf(self, T_WS):
        """docs/10 D455 실험과 같은 패턴 — 현재 disparity+pose를 TSDF에 융합해 방의 조밀한
        3D 표면을 실시간으로 재구성한다. 이동/회전이 거의 없으면(같은 자리) 생략."""
        depth_fresh = (
            self.latest_disparity is not None
            and time.monotonic() - self.latest_disparity_t < DEPTH_STALE_SEC)
        if not depth_fresh or self.latest_left is None:
            return

        if self.tsdf_last_pose is not None:
            d_t = np.linalg.norm(T_WS[:3, 3] - self.tsdf_last_pose[:3, 3])
            d_r = rotation_angle_rad(self.tsdf_last_pose[:3, :3], T_WS[:3, :3])
            if d_t < TSDF_KEYFRAME_TRANSLATION_M and d_r < np.deg2rad(TSDF_KEYFRAME_ROTATION_DEG):
                return

        K = ocams_calib.RECTIFIED_K
        disp = self.latest_disparity
        depth_m = np.zeros_like(disp, dtype=np.float32)
        valid = disp > 0
        depth_m[valid] = K[0, 0] * ocams_calib.BASELINE_M / disp[valid]
        depth_m[(depth_m < 0.1) | (depth_m > TSDF_DEPTH_TRUNC_M)] = 0.0

        color = np.repeat(self.latest_left[:, :, None], 3, axis=2)  # 흑백 -> 의사-RGB 텍스처
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(depth_m)),
            depth_scale=1.0, depth_trunc=TSDF_DEPTH_TRUNC_M, convert_rgb_to_intensity=False)

        T_SW = np.linalg.inv(T_WS)  # Open3D integrate()는 world->camera(Tcw) 규약
        self.tsdf_volume.integrate(rgbd, self.tsdf_intrinsic, T_SW)
        self.tsdf_last_pose = T_WS.copy()
        self.tsdf_keyframes += 1

        if self.tsdf_keyframes % TSDF_EXTRACT_EVERY == 0:
            cloud = self.tsdf_volume.extract_point_cloud()
            cloud = cloud.voxel_down_sample(TSDF_VOXEL_SIZE_M)
            points = np.asarray(cloud.points, dtype=np.float32)
            colors = np.clip(np.asarray(cloud.colors) * 255.0, 0, 255).astype(np.uint8)
            rr.log("world/tsdf_map",
                   rr.Points3D(points, colors=colors, radii=TSDF_VOXEL_SIZE_M * 0.45))
            o3d.io.write_point_cloud(str(TSDF_OUTPUT_PATH), cloud)
            self.get_logger().info(
                f"[tsdf] keyframes={self.tsdf_keyframes} points={len(points)} "
                f"saved={TSDF_OUTPUT_PATH}", throttle_duration_sec=2.0)

    def on_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        translation = [p.x, p.y, p.z]
        quat_xyzw = [q.x, q.y, q.z, q.w]

        T_WS = np.eye(4)
        T_WS[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
        T_WS[:3, 3] = translation

        rr.set_time_sequence("frame", self.frame_idx)
        self.frame_idx += 1

        rr.log(
            "world/cam",
            rr.Transform3D(
                translation=translation,
                quaternion=rr.Quaternion(xyzw=quat_xyzw),
            ),
        )
        # 이 카메라 자리에 핀홀 카메라 아이콘도 그려줌 (world/cam 자식이라 pose를 따라감)
        K = ocams_calib.RECTIFIED_K
        rr.log(
            "world/cam/frustum",
            rr.Pinhole(image_from_camera=K.tolist(), resolution=[640, 480]),
        )

        self.trajectory.append(translation)
        if len(self.trajectory) >= 2:
            rr.log("world/trajectory", rr.LineStrips3D([self.trajectory], colors=[0, 200, 255]))

        self.integrate_tsdf(T_WS)

        if self.latest_gaze_dir is None:
            return

        p_W = None
        gz = self.latest_gaze_dir[2]
        depth_fresh = (
            self.latest_disparity is not None
            and time.monotonic() - self.latest_disparity_t < DEPTH_STALE_SEC)
        if depth_fresh and gz > 1e-6:
            # 눈 로컬 좌표계 시선 방향을 씬 카메라 이미지 픽셀로 투영(눈-oCamS 광학축 일치
            # 근사, gaze_on_scene.py의 회전전용 투영과 같은 공식) -> 그 픽셀의 스테레오 깊이 조회.
            u = K[0, 0] * self.latest_gaze_dir[0] / gz + K[0, 2]
            v = -K[1, 1] * self.latest_gaze_dir[1] / gz + K[1, 2]
            D, n_valid = gos.depth_at(
                self.latest_disparity, u, v, K[0, 0], ocams_calib.BASELINE_M, radius=5)
            if D is not None and 0.1 < D < 5.0:
                p_W, origin, _ray_dir = fusion.gaze_point_world(u, v, D, K, T_WS)

        if p_W is not None:
            rr.log("world/gaze/hit", rr.Points3D([p_W], radii=0.03, colors=[[255, 0, 255]]))
            rr.log(
                "world/gaze/ray",
                rr.Arrows3D(origins=[origin], vectors=[(p_W - origin).tolist()],
                            colors=[[255, 0, 255]]),
            )
        else:
            # 깊이 없음/유효하지 않음 -> 기존 placeholder 광선으로 폴백(방향만 표시)
            world_dir = Rotation.from_quat(quat_xyzw).apply(self.latest_gaze_dir)
            rr.log(
                "world/gaze/ray",
                rr.Arrows3D(
                    origins=[translation],
                    vectors=[(world_dir * GAZE_RAY_LENGTH_M).tolist()],
                    colors=[[255, 0, 255]],
                ),
            )


def main():
    rr.init("molbwa_orbslam3", spawn=True)
    rclpy.init()
    node = OrbSlam3RerunBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.tsdf_keyframes > 0:
            cloud = node.tsdf_volume.extract_point_cloud()
            o3d.io.write_point_cloud(str(TSDF_OUTPUT_PATH), cloud)
            print(f"[tsdf] 최종 맵 저장: {TSDF_OUTPUT_PATH} "
                  f"({len(cloud.points)}점, keyframes={node.tsdf_keyframes})")
        if node.eye_cap is not None:
            node.eye_cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
