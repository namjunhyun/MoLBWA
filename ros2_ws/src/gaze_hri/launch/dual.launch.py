"""두 팔(a, b) 동시 기동 — 마우스 GUI 하나로, 집을 컵에 가까운 팔이 움직인다 (2026-10-14).

    ros2 launch gaze_hri dual.launch.py backend:=feetech

좌표 기준은 팔 a 의 base_link 다 (control_panel 이 팔 a 의 탑다운 호모그래피를 쓴다).
task_manager 가 같은 화면 픽셀을 팔 b 의 호모그래피로 다시 풀어 팔 b 기준 좌표를 만든다.
그래서 두 팔 사이 TF 를 따로 재지 않는다 — 두 팔 다 같은 카메라로 탑다운 캘리브만 돼 있으면 된다.

팔마다 arm_server 가 하나씩 뜬다 (액션 /arm_a/pick_place, /arm_b/pick_place).

2026-10-22 input:=gaze — 마우스 대신 안경 시선. arm/gaze_tag_bridge.py 가 태그(팔 A 기준,
~/.ros/dual_anchor.yaml)로 시선을 팔 A base_link 점으로 만들어 UDP 55055 로 보내면 gaze_bridge 가
/gaze/point_raw 로 낸다. 그 뒤(dwell -> target_resolver -> task_manager -> 팔 고르기)는 마우스와 같다.
    ros2 launch gaze_hri dual.launch.py backend:=feetech input:=gaze
/task/cancel(스페이스)은 두 팔 모두 멈춘다.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARMS = ("a", "b")
# 2026-10-22: 팔별 높이 보정 [m] (팔 a 가 ~2cm 낮게 다님 — 임시, 어깨 0점 재측정 전까지)
Z_OFFSET = {"a": 0.02, "b": 0.0}
# 2026-10-22: 컵 뒤 9cm 에서 곧장 내려가다 계속 컵을 쳤다 (사용자) -> 내려갈 땐 닿는 데까지 최대한 뒤로
# (arm_server side_descend_extra = 최대치, 실제는 위치마다 IK 로). standoff 자체를 늘리는 건 IK 불가라 안 됨
DESCEND_EXTRA = {"a": 0.15, "b": 0.15}
# 2026-10-22: 놓은 컵이 늘 6cm 바깥 (손 안에서 컵이 기준점보다 앞) -> 놓을 때 6cm 안쪽 (arm_server place_hold_offset)
PLACE_HOLD = {"a": 0.06, "b": 0.06}


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
    inp = LaunchConfiguration("input")
    is_gaze = PythonExpression(["'", inp, "' == 'gaze'"])
    is_mouse = ParameterValue(PythonExpression(["'", inp, "' == 'mouse'"]), value_type=bool)
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
                 "z_offset": Z_OFFSET.get(a, 0.0),
                 "side_descend_extra": DESCEND_EXTRA.get(a, 0.0),
                 "place_hold_offset": PLACE_HOLD.get(a, 0.0),
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
        DeclareLaunchArgument("input", default_value="mouse", description="mouse | gaze"),
        DeclareLaunchArgument("record_dir", default_value="", description="GUI 화면 녹화 폴더 (빈 값이면 안 함)"),
        # 시선: 태그 브리지가 이미 팔 A base_link 로 보낸다 -> 변환 없이 그대로
        Node(package="gaze_hri", executable="gaze_bridge", name="gaze_bridge", output="screen",
             condition=IfCondition(is_gaze),
             # 광선이 지나는 물체에 응시점을 붙인다 (= 자동 클릭). 반경은 광선 실측 오차(컵에서 11~15cm) 기준
             parameters=[params, {"source": "udp", "world_frame": "base_link", "ray_snap": True,
                                  "ray_snap_radius": 0.15, "ray_snap_height": 0.06,
                                  "ray_snap_margin": 0.05}]),

        # 마우스 모드: map = base_link (= 팔 a 의 base_link)
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="identity_tf",
             arguments=["--frame-id", "map", "--child-frame-id", "base_link"]),

        Node(package="gaze_hri", executable="control_panel",
             name="control_panel", output="screen",
             parameters=[params, {
                 "input_source": inp, "homography_file": homography["a"],
                 "record_dir": LaunchConfiguration("record_dir"),
                 "camera_index": cam_idx, "detect_objects": True,
                 "joint_states_topic": "/arm_a/joint_states", "arm_label": "A",
                 "allow_calib": False,
                 "other_arm_labels": ["B"],
                 "other_arm_homographies": [homography["b"]],
                 "other_arm_joint_topics": ["/arm_b/joint_states"],
             }]),
        Node(package="gaze_hri", executable="dwell_detector",
             name="dwell_detector", output="screen",
             parameters=[params, {"rearm_on_new_click": is_mouse}]),     # 마우스: 클릭마다 새 선택
        # 놓을 자리 거리 보정은 고른 팔 기준이어야 해서 task_manager 로 옮긴다 (여기선 끈다)
        Node(package="gaze_hri", executable="target_resolver",
             name="target_resolver", output="screen",
             # 시선이면 물체에 스냅 못 하면 안 집는다 (09-29 시연 gaze_hri.launch 와 같게). 마우스는 클릭한 곳 그대로
             parameters=[params, {"require_snap_for_pick": ParameterValue(is_gaze, value_type=bool),
                                  "place_snap_objects": ParameterValue(is_gaze, value_type=bool),
                                  # 기준이 팔 A 라 팔 B 앞 물체는 0.6m 를 넘는다 (물병 0.62m 가 빠져 선택 거부)
                                  "object_max_range": 1.0,
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
