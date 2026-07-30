#!/usr/bin/env python3
"""
docs/09_visualization.md 섹션 4 로깅 스케치 — /orbslam3/pose(SLAM) + 눈 카메라(시선 방향)를
합쳐서 Rerun 3D 뷰에 카메라 pose + 궤적 + 시선 광선을 그린다.

시선 광선은 아직 "근사"다 — 진짜 지점(p_W, docs/03_fusion.md)을 찍으려면 시선 픽셀의 스테레오
깊이(D)가 필요한데, StereoSGBM 깊이 파이프라인이 아직 없다. 그래서 지금은:
  - 광선의 원점 = SLAM pose 위치 그대로 (눈-카메라 물리적 오프셋은 무시, docs/03의
    "눈과 스테레오 카메라가 머리 위 몇 cm 차이라 실용적으로 같다고 봐도 된다" 가정 재사용)
  - 광선의 방향 = 눈 트래커가 주는 gaze_direction을 SLAM pose의 회전으로 world 좌표계로 변환
  - 광선의 길이 = GAZE_RAY_LENGTH_M 고정값 (진짜 깊이가 아니라 "이 방향을 보고 있다"는 것만
    보여주는 placeholder)

나중에 깊이가 생기면: fusion.gaze_point_world(u, v, D, K, T_WS)로 정확한 p_W를 계산해서
world/gaze/hit(Points3D)로 찍고, 이 광선은 그 지점까지만 그리도록 바꾸면 됨.

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

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gaze_on_scene as gos  # noqa: E402  find_cam/open_eye/tracker 재사용 (같은 디렉토리)

GAZE_RAY_LENGTH_M = 2.0  # 깊이 없을 때의 광선 길이 placeholder


class OrbSlam3RerunBridge(Node):
    def __init__(self):
        super().__init__("orbslam3_rerun_bridge")
        self.trajectory = []
        self.frame_idx = 0
        self.latest_gaze_dir = None  # 눈 카메라 로컬 좌표계, 단위벡터

        self.eye_cap = gos.open_eye(640, 480, 30)
        self.create_timer(1.0 / 30.0, self.on_eye_timer)

        self.create_subscription(PoseStamped, "/orbslam3/pose", self.on_pose, 10)
        self.get_logger().info("orbslam3_rerun_bridge 시작 — /orbslam3/pose 구독 + 눈 카메라 시선 광선(근사)")

    def on_eye_timer(self):
        ok, eye = self.eye_cap.read()
        if not ok:
            return
        _ellipse, d = gos.tracker.process_frame(eye)
        if d is not None and np.linalg.norm(d) > 1e-6:
            self.latest_gaze_dir = d / np.linalg.norm(d)
        cv2.waitKey(1)  # 이게 없으면 process_frame 내부의 cv2.imshow 창이 실제로 안 그려짐

    def on_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        translation = [p.x, p.y, p.z]
        quat_xyzw = [q.x, q.y, q.z, q.w]

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
        rr.log(
            "world/cam/frustum",
            rr.Pinhole(
                image_from_camera=[[433.553025, 0.0, 470.352457],
                                    [0.0, 433.553025, 236.488789],
                                    [0.0, 0.0, 1.0]],
                resolution=[640, 480],
            ),
        )

        self.trajectory.append(translation)
        if len(self.trajectory) >= 2:
            rr.log("world/trajectory", rr.LineStrips3D([self.trajectory], colors=[0, 200, 255]))

        if self.latest_gaze_dir is not None:
            # 눈 로컬 좌표계 -> world (SLAM pose 회전만 적용, 눈-oCamS 상대회전은 미보정 근사)
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
        node.eye_cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
