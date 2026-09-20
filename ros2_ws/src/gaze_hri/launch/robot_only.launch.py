"""안경 없이 로봇 쪽만 테스트.

마우스 클릭이 시선을 대신합니다. SLAM도 안경도 필요 없습니다.

    # 최초 1회 캘리브레이션
    ros2 launch gaze_hri robot_only.launch.py mode:=calib

    # 테스트 (처음엔 반드시 dry_run)
    ros2 launch gaze_hri robot_only.launch.py
    ros2 launch gaze_hri robot_only.launch.py backend:=feetech
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("gaze_hri")
    params = os.path.join(share, "config", "gaze_hri.yaml")

    mode = LaunchConfiguration("mode")
    backend = LaunchConfiguration("backend")
    fail_rate = ParameterValue(LaunchConfiguration("fail_rate"), value_type=float)
    cam_idx = ParameterValue(LaunchConfiguration("camera_index"), value_type=int)
    detect = ParameterValue(LaunchConfiguration("detect_objects"), value_type=bool)

    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="run",
                              description="run | calib"),
        DeclareLaunchArgument("backend", default_value="dry_run",
                              description="dry_run | feetech | joint"),
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument(
            "fail_rate", default_value="0.0",
            description="dry_run에서 파지 실패를 흉내 낼 확률 (0.5 = 50%)"),
        DeclareLaunchArgument(
            "detect_objects", default_value="false",
            description="색 기반 컵 검출을 켠다. 여러 컵 구분 시나리오를 "
                        "테스트하려면 true (컵 색이 일정해야 함)"),

        # map -> base_link 를 항등변환으로. 안경이 없으니 두 좌표계가 같습니다.
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="identity_tf",
             arguments=["--frame-id", "map", "--child-frame-id", "base_link"]),

        Node(package="gaze_hri", executable="topdown_click",
             name="topdown_click", output="screen",
             parameters=[params, {"mode": mode, "camera_index": cam_idx,
                                  "detect_objects": detect}]),

        Node(package="gaze_hri", executable="dwell_detector",
             name="dwell_detector", output="screen", parameters=[params]),

        Node(package="gaze_hri", executable="target_resolver",
             name="target_resolver", output="screen", parameters=[params]),

        Node(package="gaze_hri", executable="task_manager",
             name="task_manager", output="screen", parameters=[params]),

        Node(package="gaze_hri", executable="arm_server",
             name="arm_server", output="screen",
             parameters=[params, {"backend": backend,
                                  "simulate_fail_rate": fail_rate}]),
    ])
