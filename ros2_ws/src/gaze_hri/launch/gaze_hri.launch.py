"""전체 파이프라인 기동 (안경 + 로봇).

    ros2 launch gaze_hri gaze_hri.launch.py                  # 시뮬레이션 (안전)
    ros2 launch gaze_hri gaze_hri.launch.py source:=udp backend:=feetech

좌표계 요약:
    gaze_bridge   -> /gaze/point_raw   frame=map        (SLAM world)
    topdown_click -> /objects/poses    frame=base_link  (호모그래피 결과)
    target_resolver 가 둘 다 base_link 로 변환해 비교하고, 목표도 base_link 로 낸다.
    map <-> base_link 는 calib_tf_publisher 가 T_BW 캘리브 결과로 발행한다.

gaze_frame (2026-09-23):
    map       : SLAM 경로. 시선이 SLAM world 로 오고 T_BW 로 변환한다(기본값, 예전 그대로).
    base_link : 태그 직결 경로(arm/gaze_tag_bridge.py). 시선이 이미 로봇 기준이라
                변환이 필요 없다. calib_tf_publisher 는 띄우지 않는다 — 띄우면
                "캘리브 파일 없음, 항등변환" 경고가 계속 떠서 오해를 부른다.
        ros2 launch gaze_hri gaze_hri.launch.py source:=udp gaze_frame:=base_link

use_topdown:
    탑다운 카메라가 컵의 정확한 위치를 담당한다. 시선은 '어느 컵인지'만 고른다.
    이게 켜져 있어야 시선 오차(수 cm)가 파지 정확도로 넘어가지 않는다.
    꺼면 응시점을 그대로 파지점으로 쓰므로 정확도가 시선 오차에 직접 묶인다.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("gaze_hri")
    params = os.path.join(share, "config", "gaze_hri.yaml")

    source = LaunchConfiguration("source")
    backend = LaunchConfiguration("backend")
    calib = LaunchConfiguration("calib_file")
    use_topdown = LaunchConfiguration("use_topdown")
    # 탑다운이 켜져 있으면 스냅 실패를 거부한다(= 모르면 안 집는다).
    require_snap = ParameterValue(use_topdown, value_type=bool)
    cam_idx = ParameterValue(LaunchConfiguration("camera_index"), value_type=int)
    gaze_frame = LaunchConfiguration("gaze_frame")
    gaze_is_base = PythonExpression(["'", gaze_frame, "' == 'base_link'"])

    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="fake",
                              description="fake | udp | inproc"),
        DeclareLaunchArgument("backend", default_value="dry_run",
                              description="dry_run | feetech | joint"),
        DeclareLaunchArgument(
            "calib_file",
            default_value=os.path.expanduser("~/.ros/gaze_hri_calib.yaml")),
        DeclareLaunchArgument(
            "use_topdown", default_value="true",
            description="탑다운 카메라로 컵 위치를 잡는다(권장). "
                        "끄면 응시점을 그대로 파지점으로 쓴다."),
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument(
            "gaze_frame", default_value="map",
            description="map (SLAM 경로) | base_link (태그 직결, arm/gaze_tag_bridge.py)"),

        Node(package="gaze_hri", executable="calib_tf_publisher",
             name="calib_tf_publisher", output="screen",
             condition=UnlessCondition(gaze_is_base),
             parameters=[params, {"calib_file": calib}]),

        Node(package="gaze_hri", executable="gaze_bridge",
             name="gaze_bridge", output="screen",
             parameters=[params, {"source": source, "world_frame": gaze_frame}]),

        # 탑다운: 마우스 입력은 쓰지 않고 물체 검출만 담당한다.
        # (커서를 안 움직이면 /gaze/point_raw 는 나가지 않는다)
        Node(package="gaze_hri", executable="topdown_click",
             name="topdown_click", output="screen",
             condition=IfCondition(use_topdown),
             parameters=[params, {"mode": "run", "camera_index": cam_idx,
                                  "detect_objects": True}]),

        Node(package="gaze_hri", executable="dwell_detector",
             name="dwell_detector", output="screen", parameters=[params]),

        Node(package="gaze_hri", executable="target_resolver",
             name="target_resolver", output="screen",
             parameters=[params, {"require_snap_for_pick": require_snap}]),

        Node(package="gaze_hri", executable="task_manager",
             name="task_manager", output="screen", parameters=[params]),

        Node(package="gaze_hri", executable="arm_server",
             name="arm_server", output="screen",
             parameters=[params, {"backend": backend}]),
    ])
