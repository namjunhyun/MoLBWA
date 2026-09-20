"""T_BW(SLAM world <-> 로봇 베이스) 캘리브레이션 전용 기동.

    # 터미널 1
    ros2 launch gaze_hri calibrate_world_to_base.launch.py source:=udp backend:=feetech
    # 터미널 2
    ros2 run gaze_hri calibrate_world_to_base

★ 왜 전체 런치(gaze_hri.launch.py)를 쓰면 안 되는가

캘리브레이션은 "그리퍼 끝을 응시"하는 작업이라 응시점이 계속 발행된다.
전체 런치에는 target_resolver 와 task_manager 가 떠 있어서, 그 응시가
'집을 물체 선택' -> '놓을 자리 선택' 으로 해석되고 팔이 캘리브레이션 도중에
제멋대로 집기 동작을 시작할 수 있다.

그래서 이 런치는 캘리브레이션에 필요한 것만 띄운다:
  gaze_bridge     응시점 공급
  dwell_detector  응시 확정
  arm_server      /arm/goto_joints 로 팔을 자세로 보냄

target_resolver / task_manager 는 일부러 띄우지 않는다.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("gaze_hri")
    params = os.path.join(share, "config", "gaze_hri.yaml")

    source = LaunchConfiguration("source")
    backend = LaunchConfiguration("backend")

    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="udp",
                              description="fake | udp | inproc"),
        DeclareLaunchArgument("backend", default_value="feetech",
                              description="dry_run | feetech | joint. "
                                          "팔이 실제로 자세를 잡아야 하므로 "
                                          "보통 feetech 다."),

        Node(package="gaze_hri", executable="gaze_bridge",
             name="gaze_bridge", output="screen",
             parameters=[params, {"source": source}]),

        Node(package="gaze_hri", executable="dwell_detector",
             name="dwell_detector", output="screen", parameters=[params]),

        Node(package="gaze_hri", executable="arm_server",
             name="arm_server", output="screen",
             parameters=[params, {"backend": backend}]),
    ])
