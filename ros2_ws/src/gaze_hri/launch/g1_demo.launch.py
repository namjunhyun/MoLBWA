"""데모 2(G1 교감): g1_interaction 만 띄운다. 브리지(g1/g1_gaze_bridge.py)는 별도 프로세스.

    ros2 launch gaze_hri g1_demo.launch.py jetson_host:=192.168.50.119
    ros2 launch gaze_hri g1_demo.launch.py jetson_host:=127.0.0.1   # sim2sim (mujoco 서버를 같은 PC 에서)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory("gaze_hri"), "config", "gaze_hri.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("jetson_host", default_value="192.168.50.119"),
        DeclareLaunchArgument("g1_dir", default_value="",
                              description="저장소 g1/ 경로. 비면 노드가 위로 올라가며 찾는다"),
        Node(package="gaze_hri", executable="g1_interaction", name="g1_interaction", output="screen",
             parameters=[params, {"jetson_host": LaunchConfiguration("jetson_host"),
                                  "g1_dir": LaunchConfiguration("g1_dir")}]),
    ])
