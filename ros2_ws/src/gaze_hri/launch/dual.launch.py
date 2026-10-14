"""두 팔(a, b) 동시 기동 — 마우스 GUI 하나로, 집을 컵에 가까운 팔이 움직인다 (2026-10-14).

    ros2 launch gaze_hri dual.launch.py backend:=feetech

좌표 기준은 팔 a 의 base_link 다 (control_panel 이 팔 a 의 탑다운 호모그래피를 쓴다).
task_manager 가 같은 화면 픽셀을 팔 b 의 호모그래피로 다시 풀어 팔 b 기준 좌표를 만든다.
그래서 두 팔 사이 TF 를 따로 재지 않는다 — 두 팔 다 같은 카메라로 탑다운 캘리브만 돼 있으면 된다.

팔마다 arm_server 가 하나씩 뜬다 (액션 /arm_a/pick_place, /arm_b/pick_place).
/task/cancel(스페이스)은 두 팔 모두 멈춘다.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARMS = ("a", "b")


def generate_launch_description():
    share = get_package_share_directory("gaze_hri")
    params = os.path.join(share, "config", "gaze_hri.yaml")
    cfg_dir = os.path.dirname(os.path.realpath(params))
    ros = os.path.expanduser("~/.ros")
    homography = {a: f"{ros}/topdown_homography_{a}.yaml" for a in ARMS}

    # arm_server 는 팔마다 이름이 달라(arm_server_a/_b) yaml 의 arm_server: 키가 안 붙는다.
    # 그 블록을 직접 꺼내 넘긴다.
    with open(params) as f:
        arm_params = yaml.safe_load(f)["arm_server"]["ros__parameters"]
    band = yaml.safe_load(open(params))["target_resolver"]["ros__parameters"]

    backend = LaunchConfiguration("backend")
    cam_idx = ParameterValue(LaunchConfiguration("camera_index"), value_type=int)

    def leader_port(a):
        f = os.path.join(ros, "so101_leaders.yaml")
        if not os.path.exists(f):
            return None
        with open(f) as fh:
            return (yaml.safe_load(fh) or {}).get(a)

    def roll_sign(a):
        # 손목 roll 방향 (따르기용). 팔마다 눈으로 확인한 값을 관절 맵 파일에 roll_sign 으로 적는다
        with open(f"{cfg_dir}/so101_follower_{a}.yaml") as f:
            return int(yaml.safe_load(f).get("roll_sign", 0))

    arm_nodes = [
        Node(package="gaze_hri", executable="arm_server",
             name=f"arm_server_{a}", output="screen",
             parameters=[arm_params, {
                 "backend": backend,
                 "joint_map_file": f"{cfg_dir}/so101_follower_{a}.yaml",
                 "rest_ticks_file": f"{ros}/so101_rest_ticks_{a}.json",
                 "action_name": f"/arm_{a}/pick_place",
                 "joint_states_topic": f"/arm_{a}/joint_states",
                 "goto_topic": f"/arm_{a}/goto_joints",
                 "roll_sign": roll_sign(a),
                 "arm_name": a,
                 "homography_file": homography[a],      # 따르기 보정: 화면 방향 -> 책상 방향
                 # 리더 (tools/so101_leader.py assign/solve). 아직 없으면 "" — 따르기 보정만 거부된다
                 "leader_port": leader_port(a) or "",
                 "leader_map_file": f"{cfg_dir}/so101_leader_{a}.yaml",
             }])
        for a in ARMS
    ]

    return LaunchDescription([
        DeclareLaunchArgument("backend", default_value="dry_run",
                              description="dry_run | feetech"),
        DeclareLaunchArgument("camera_index", default_value="0"),

        # 마우스 모드: map = base_link (= 팔 a 의 base_link)
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="identity_tf",
             arguments=["--frame-id", "map", "--child-frame-id", "base_link"]),

        Node(package="gaze_hri", executable="control_panel",
             name="control_panel", output="screen",
             parameters=[params, {
                 "input_source": "mouse", "homography_file": homography["a"],
                 "camera_index": cam_idx, "detect_objects": True,
                 "joint_states_topic": "/arm_a/joint_states", "arm_label": "A",
                 "allow_calib": False,
                 "other_arm_labels": ["B"],
                 "other_arm_homographies": [homography["b"]],
                 "other_arm_joint_topics": ["/arm_b/joint_states"],
             }]),
        Node(package="gaze_hri", executable="dwell_detector",
             name="dwell_detector", output="screen",
             parameters=[params, {"rearm_on_new_click": True}]),     # 마우스: 클릭마다 새 선택
        # 놓을 자리 거리 보정은 고른 팔 기준이어야 해서 task_manager 로 옮긴다 (여기선 끈다)
        Node(package="gaze_hri", executable="target_resolver",
             name="target_resolver", output="screen",
             parameters=[params, {"require_snap_for_pick": False,
                                  "place_min_r": 0.0, "place_max_r": 10.0}]),
        Node(package="gaze_hri", executable="task_manager",
             name="task_manager", output="screen",
             parameters=[params, {
                 "arms": list(ARMS),
                 "arm_homographies": [homography[a] for a in ARMS],
                 "arm_actions": [f"/arm_{a}/pick_place" for a in ARMS],
                 "pick_min_r": float(band["place_min_r"]),     # 숙여 집기 포함 (arm_server side_tilt_min_radius 와 같게)
                 "pick_max_r": float(band["place_max_r"]),
                 "place_min_r": float(band["place_min_r"]),
                 "place_max_r": float(band["place_max_r"]),
             }]),
        *arm_nodes,
    ])
