"""통합 GUI 기동 — 한 창에서 전체 파이프라인을 보고 조작한다.

    # 마우스로 시선을 대신 (안경 없이, 로봇만 검증할 때)
    ros2 launch gaze_hri gui.launch.py
    ros2 launch gaze_hri gui.launch.py backend:=feetech

    # 안경 연동 (시선 입력)
    ros2 launch gaze_hri gui.launch.py input:=gaze source:=udp backend:=feetech

무엇이 뜨는가

    control_panel    통합 GUI (카메라 + 물체 검출 + 화면 + 조작)
    dwell_detector   응시 판정
    target_resolver  무엇을/어디에
    task_manager     상태 기계
    arm_server       역기구학 + 서보

    input:=mouse 면 map=base_link 항등 TF 를, gaze 면 gaze_bridge 와
    T_BW 캘리브레이션 TF(calib_tf_publisher)를 같이 띄운다.

주의: 이 런치는 topdown_click 을 띄우지 않는다. control_panel 이 카메라를
독점하기 때문이다. 호모그래피 캘리브레이션은 따로 한다:

    ros2 launch gaze_hri robot_only.launch.py mode:=calib backend:=feetech
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

    input_src = LaunchConfiguration("input")
    backend = LaunchConfiguration("backend")
    source = LaunchConfiguration("source")
    calib = LaunchConfiguration("calib_file")
    cam_idx = ParameterValue(LaunchConfiguration("camera_index"), value_type=int)

    is_gaze = PythonExpression(["'", input_src, "' == 'gaze'"])
    # 시선 모드에서는 탑다운이 컵의 정확한 위치를 담당하므로,
    # 물체를 특정하지 못하면 목표를 내지 않는다(모르면 안 집는다).
    require_snap = ParameterValue(is_gaze, value_type=bool)

    return LaunchDescription([
        DeclareLaunchArgument("input", default_value="mouse",
                              description="mouse | gaze"),
        DeclareLaunchArgument("backend", default_value="dry_run",
                              description="dry_run | feetech | joint"),
        DeclareLaunchArgument("source", default_value="udp",
                              description="input:=gaze 일 때 gaze_bridge 소스 "
                                          "(fake | udp | inproc)"),
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument(
            "calib_file",
            default_value=os.path.expanduser("~/.ros/gaze_hri_calib.yaml")),

        # --- 좌표계 ---
        # 마우스 모드: 안경이 없으니 map 과 base_link 가 같다
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="identity_tf", condition=UnlessCondition(is_gaze),
             arguments=["--frame-id", "map", "--child-frame-id", "base_link"]),
        # 시선 모드: T_BW 캘리브레이션 결과를 TF 로 발행
        Node(package="gaze_hri", executable="calib_tf_publisher",
             name="calib_tf_publisher", output="screen",
             condition=IfCondition(is_gaze),
             parameters=[params, {"calib_file": calib}]),
        Node(package="gaze_hri", executable="gaze_bridge",
             name="gaze_bridge", output="screen",
             condition=IfCondition(is_gaze),
             parameters=[params, {"source": source}]),

        # --- 통합 GUI ---
        Node(package="gaze_hri", executable="control_panel",
             name="control_panel", output="screen",
             parameters=[params, {"input_source": input_src,
                                  "camera_index": cam_idx,
                                  "detect_objects": True}]),

        # --- 파이프라인 ---
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
