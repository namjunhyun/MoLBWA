#!/usr/bin/env python3
"""
[노드 5] arm_server  —  로봇 동작 (PickPlace 액션 서버)

map(SLAM world) 기준 좌표를 받아서 base_link로 변환하고, 역기구학을 풀어
SO-ARM101을 실제로 움직입니다.

동작 시퀀스:
    HOME -> 집을 위치 위쪽(접근) -> 하강 -> 그리퍼 닫기 -> 들어올리기
         -> 놓을 위치 위쪽 -> 하강 -> 그리퍼 열기 -> 상승 -> HOME

서보 백엔드는 두 가지입니다:
  feetech : scservo_sdk 로 STS3215 서보를 직접 제어 (의존성 최소)
  joint   : /joint_trajectory 토픽으로 trajectory_msgs 발행
            (ros2_so_arm + ros2_control + MoveIt을 이미 쓰고 있다면 이쪽)

MoveIt2를 쓰면 충돌 회피와 경로 계획이 공짜로 따라오지만 셋업이 무겁습니다.
책상 위 컵 옮기기 정도면 여기 있는 직선 보간(linear interpolation)으로 충분하고,
대회 시연 때 훨씬 덜 깨집니다.
"""

import math
import os
import time

import numpy as np
import rclpy
import tf2_ros
from gaze_hri_msgs.action import PickPlace
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gaze_hri.so101_driver import SafetyStop
from gaze_hri.kinematics import (JOINT_ORDER, ArmGeometry, IKError,
                                 forward_kinematics, grasp_detected,
                                 inverse_kinematics, quat_to_matrix,
                                 solve_with_fallback)


# ======================================================================
# 서보 백엔드
# ======================================================================

class DryRunBackend:
    """실제 하드웨어 없이 로그만 출력. 처음에는 반드시 이걸로 먼저 검증하세요.

    fail_rate를 주면 파지 실패를 확률적으로 흉내 냅니다.
    재시도 로직을 하드웨어 없이 검증할 때 씁니다.

    파지 시뮬레이션의 원리(close_and_verify와 짝을 이룸):
      성공 = 물체가 손가락 사이에 있어 그리퍼가 명령값까지 못 닫힘
             -> read()가 명령값보다 held_gap 만큼 '더 열린' 값을 돌려준다.
      실패 = 빈손이라 끝까지 닫힘
             -> read()가 거의 완전히 닫힌 값을 돌려준다.

    held_gap 은 grasp_margin 보다 확실히 커야 '물었음'으로 판정됩니다.
    (arm_server가 grasp_margin의 2배를 넣어줍니다.)
    """

    def __init__(self, logger, fail_rate=0.0, held_gap=0.16, empty_pos=0.02):
        self.logger = logger
        self.q = [0.0] * 6
        self.fail_rate = float(fail_rate)
        self.held_gap = float(held_gap)
        self.empty_pos = float(empty_pos)
        self._failed_this_grasp = False
        self._grasp_simulated = False

    def write(self, q):
        self.q = list(q)
        deg = ", ".join(f"{math.degrees(v):6.1f}" for v in q)
        self.logger.info(f"  [dry-run] 관절각(deg): [{deg}]")

    def read(self):
        q = list(self.q)
        if not self._grasp_simulated:
            return q
        if self._failed_this_grasp:
            # 빈손: 끝까지 닫힌다
            q[5] = min(q[5], self.empty_pos)
        else:
            # 물었다: 물체 두께만큼 덜 닫힌다
            q[5] = q[5] + self.held_gap
        return q

    def simulate_grasp(self):
        import random
        self._failed_this_grasp = random.random() < self.fail_rate
        self._grasp_simulated = True
        return not self._failed_this_grasp

    def release(self):
        """그리퍼를 열면 파지 시뮬 상태를 해제한다."""
        self._grasp_simulated = False
        self._failed_this_grasp = False

    def close(self):
        pass


class FeetechBackend:
    """SO-101 팔로워 직접 제어 — so101_driver 위의 얇은 어댑터 (2026-09-23 재작성).

    예전 구현은 Feetech 공식 SDK 의 sms_sts 를 가정했는데 pip 의 scservo_sdk 에는 없고,
    "2048틱 = 기구학 0도, 방향 전부 +" 를 가정했다. 이제 틱<->각도 대응은
    tools/so101_map_calib.py 로 실측한 파일(joint_map_file)에서만 가져온다.
    그 파일이 없으면 **시작을 거부한다** — 짐작한 대응으로 팔을 움직이지 않는다.
    """

    def __init__(self, map_file, logger):
        import yaml
        from gaze_hri.so101_driver import JointMap, So101Bus
        map_file = os.path.expanduser(map_file)
        if not map_file or not os.path.exists(map_file):
            raise RuntimeError(
                f"joint_map_file 이 없다: {map_file!r}. tools/so101_map_calib.py 로 먼저 "
                "틱<->각도 대응을 실측할 것. 짐작한 값으로는 움직이지 않는다.")
        cfg = yaml.safe_load(open(map_file))
        if cfg.get("warnings"):
            logger.warn(f"관절 대응 캘리브에 경고가 남아 있다: {cfg['warnings']}")
        self.logger = logger
        self.map = JointMap(cfg["zero_ticks"], cfg["signs"])
        self.bus = So101Bus(cfg["port"], cfg["servo_ids"],
                            log=lambda m: logger.info(m))
        self.bus.enable()          # 목표=현재로 맞춘 뒤 토크 ON -> 튀지 않는다

    def write(self, q):
        # SafetyStop 은 그대로 올려 보낸다 -> execute 가 복귀 없이 멈춘다
        self.bus.write_ticks(self.map.to_ticks(q))

    def read(self):
        return self.map.to_rad(self.bus.ticks())

    def read_gripper_load(self):
        """그리퍼 서보의 부하. 물체를 물고 있으면 값이 올라갑니다."""
        try:
            return abs(int(self.bus.loads()[5]))
        except Exception:
            return None

    def close(self):
        self.bus.hold()            # 토크는 유지한 채 그 자리에 둔다 (끄면 팔이 떨어진다)
        self.bus.close()


class JointTrajectoryBackend:
    """ros2_control / MoveIt을 이미 쓰고 있을 때."""

    def __init__(self, node, topic="/arm_controller/joint_trajectory"):
        self.node = node
        self.pub = node.create_publisher(JointTrajectory, topic, 10)
        self.q = [0.0] * 6

    def write(self, q):
        self.q = list(q)
        msg = JointTrajectory()
        msg.joint_names = JOINT_ORDER[:len(q)]
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 200_000_000
        msg.points.append(pt)
        self.pub.publish(msg)

    def read(self):
        return list(self.q)

    def close(self):
        pass


# ======================================================================
# 액션 서버
# ======================================================================

class ArmServer(Node):

    def __init__(self):
        super().__init__("arm_server")

        self.declare_parameter("backend", "dry_run")     # dry_run | feetech | joint
        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("servo_ids", [1, 2, 3, 4, 5, 6])
        self.declare_parameter("servo_signs", [1, 1, 1, 1, 1, 1])
        self.declare_parameter("servo_offsets", [0, 0, 0, 0, 0, 0])
        # feetech 백엔드 전용: tools/so101_map_calib.py 가 만든 틱<->각도 대응 파일
        self.declare_parameter("joint_map_file", "")
        # 홈 자세를 관절각 대신 TCP 위치로 준다 (비어 있으면 home_q). 관절각 home_q 기본값은
        # 이 기구학에서 팔꿈치가 베이스 바닥 아래(z=-0.009)로 가는 자세였다.
        self.declare_parameter("home_xyz", [0.0])
        self.declare_parameter("home_pitch", -1.5708)
        # 관절 공간 이동(홈/goto_joints) 속도 상한 [deg/s]
        self.declare_parameter("joint_speed_deg_s", 25.0)
        # 손가락 끝 충돌 검사 (2026-09-24). TCP(l3)는 컵 중심이 오는 파지점이고, 고정 손가락
        # 끝은 그보다 finger_tip_extra 만큼 더 나가 있다 (실측 손목축->끝 0.18m, l3 0.135m).
        # 45° 접근이면 끝이 TCP 보다 3.2cm 낮고, 팔꿈치가 중력으로 ~1.2cm 처진다.
        # 경로 전체에서 끝이 min_tip_z 아래로 가는 접근각은 쓰지 않는다.
        self.declare_parameter("finger_tip_extra", 0.045)
        self.declare_parameter("min_tip_z", 0.02)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("world_frame", "map")

        self.declare_parameter("approach_height", 0.10)   # 목표 위 몇 m에서 접근
        self.declare_parameter("lift_height", 0.12)       # 잡은 뒤 들어올리는 높이
        self.declare_parameter("step_time", 0.04)         # 보간 한 스텝 시간 [s]
        self.declare_parameter("cartesian_step", 0.008)   # 보간 간격 [m]
        self.declare_parameter("gripper_open", 1.2)       # rad
        self.declare_parameter("gripper_closed", 0.15)    # rad
        self.declare_parameter("home_q", [0.0, -0.6, 1.2, -0.6, 0.0, 1.2])
        self.declare_parameter("workspace_radius", 0.42)  # 안전 반경 [m]
        self.declare_parameter("min_z", -0.02)            # 이보다 낮으면 거부 [m]

        # ---- 파지 실패 감지 / 재시도 ----
        # 물체를 물었다면 그리퍼가 끝까지 닫히지 못합니다. 그 차이로 판별합니다.
        # grasp_margin: 완전히 닫힌 값보다 이만큼 이상 열려 있어야 "물었다"로 봅니다.
        #               컵 두께의 절반 정도에 해당하는 각도로 잡으세요.
        self.declare_parameter("grasp_margin", 0.08)      # rad
        self.declare_parameter("grasp_settle", 0.9)       # 닫고 나서 대기 [s]
        self.declare_parameter("min_grasp_load", 0)       # 0이면 부하 검사 안 함
        self.declare_parameter("max_grasp_retries", 3)
        self.declare_parameter("retry_offset", 0.006)     # 재시도 시 흔드는 폭 [m]
        self.declare_parameter("simulate_fail_rate", 0.0)  # dry_run 전용

        # URDF에서 반드시 실제 값으로 바꾸세요
        self.declare_parameter("base_height", 0.0563)
        self.declare_parameter("shoulder_offset", 0.0304)
        self.declare_parameter("l1", 0.1160)
        self.declare_parameter("l2", 0.1350)
        self.declare_parameter("l3", 0.1350)

        self.geo = ArmGeometry(
            base_height=float(self.get_parameter("base_height").value),
            shoulder_offset=float(self.get_parameter("shoulder_offset").value),
            l1=float(self.get_parameter("l1").value),
            l2=float(self.get_parameter("l2").value),
            l3=float(self.get_parameter("l3").value),
        )

        self.base_frame = self.get_parameter("base_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.backend = self._make_backend()
        self.home = self._resolve_home()
        if isinstance(self.backend, FeetechBackend):
            # 실제 팔은 home 이 아니라 지금 있는 자세에서 출발한다. home_q 로 두면
            # 첫 보간이 엉뚱한 곳에서 시작해 팔이 한 번에 튄다.
            self.q = list(self.backend.read())
            self.get_logger().info(f"현재 관절각(rad): {[round(v, 3) for v in self.q]}")
        else:
            self.q = list(self.home)

        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        # 동작 단계를 토픽으로도 알린다. 액션 피드백은 목표를 보낸 클라이언트만
        # 받기 때문에, GUI 같은 제3자가 진행 상황을 보려면 이게 필요하다.
        self.pub_phase = self.create_publisher(String, "/arm/phase", 10)
        self.create_timer(0.05, self._publish_joint_state)

        # 캘리브레이션 도구가 팔을 원하는 자세로 보낼 때 사용
        self.create_subscription(
            Float64MultiArray, "/arm/goto_joints", self.on_goto_joints, 10)

        self._cancel = False
        cb = ReentrantCallbackGroup()
        self._server = ActionServer(
            self, PickPlace, "/pick_place",
            execute_callback=self.execute,
            goal_callback=lambda g: GoalResponse.ACCEPT,
            cancel_callback=self.on_cancel_request,
            callback_group=cb,
        )
        # 액션 취소와 별개로, 토픽 한 방으로도 즉시 멈출 수 있게 한다.
        # (액션 목표가 아직 accept 되기 전이거나 클라이언트가 죽은 경우 대비)
        self.create_subscription(
            Empty, "/task/cancel", self.on_cancel_topic, 10, callback_group=cb)

        self._check_workspace_radius()
        self.get_logger().info("arm_server 준비 완료 (/pick_place)")

    # ------------------------------------------------------------------
    def on_cancel_request(self, goal_handle):
        """액션 취소 요청. 동작 루프가 실제로 멈추도록 플래그를 세운다.

        이 플래그가 없으면 move_to/go_home 의 중단 검사가 전부 죽은 코드가 되어,
        취소를 눌러도 팔이 시퀀스를 끝까지 수행한다.
        """
        self._cancel = True
        self.get_logger().warn("취소 요청 수신 — 현재 동작을 중단합니다.")
        return CancelResponse.ACCEPT

    def on_cancel_topic(self, _msg):
        self._cancel = True
        self.get_logger().warn("/task/cancel 수신 — 현재 동작을 중단합니다.")

    def _check_workspace_radius(self):
        """안전 반경이 실제 도달 범위보다 크면 검사가 무의미해지므로 경고한다."""
        reach = (self.geo.shoulder_offset + self.geo.l1
                 + self.geo.l2 + self.geo.l3)
        radius = float(self.get_parameter("workspace_radius").value)
        if radius > reach:
            self.get_logger().warn(
                f"workspace_radius({radius:.3f} m)가 링크 합 최대 도달거리"
                f"({reach:.3f} m)보다 큽니다. 안전 검사를 통과한 목표가 "
                "역기구학에서 실패하게 됩니다. 링크 길이를 실측했는지 확인하고 "
                f"{reach * 0.97:.2f} 이하로 낮추세요."
            )

    # ------------------------------------------------------------------
    def _make_backend(self):
        kind = self.get_parameter("backend").value
        if kind == "feetech":
            return FeetechBackend(self.get_parameter("joint_map_file").value,
                                  self.get_logger())
        if kind == "joint":
            return JointTrajectoryBackend(self)
        # held_gap: '물었을 때' 명령값보다 얼마나 더 열려 있는지. 판정 기준인
        # grasp_margin 보다 확실히 커야 재시도 로직을 의도대로 검증할 수 있다.
        margin = float(self.get_parameter("grasp_margin").value)
        return DryRunBackend(
            self.get_logger(),
            float(self.get_parameter("simulate_fail_rate").value),
            held_gap=margin * 2.0)

    def _publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_ORDER
        msg.position = [float(v) for v in self.q]
        self.pub_js.publish(msg)

    # ------------------------------------------------------------------
    def on_goto_joints(self, msg: Float64MultiArray):
        """관절각 직접 지정 이동. 좌표계 캘리브레이션 때 팔을 기준 자세로 보냅니다."""
        target = list(msg.data)
        if len(target) < 4:
            self.get_logger().warn("goto_joints: 관절값이 4개 미만입니다.")
            return
        while len(target) < 6:
            target.append(self.q[len(target)])
        try:
            self._joint_move(target)
        except SafetyStop as exc:
            self.get_logger().error(f"안전 정지 (goto_joints): {exc}")
            self.q = list(self.backend.read())

    # ------------------------------------------------------------------
    def transform_to_base(self, point, source_frame):
        """주어진 프레임의 점을 base_frame 좌표로 변환한다.

        source_frame 이 비어 있으면 world_frame 으로 간주한다(하위호환).
        이미 base_frame 이면 변환하지 않는다 — target_resolver 가 변환을 마치고
        보내는 정상 경로가 여기에 해당한다.

        TF를 못 찾으면 '그대로 사용'하지 않고 목표를 거부한다. 좌표계가 다른 값을
        그대로 쓰면 팔이 엉뚱한 곳으로 가는데, 그게 조용히 일어나면 안 된다.
        """
        p = np.array([point.x, point.y, point.z])
        src = source_frame or self.world_frame
        if src == self.base_frame:
            return p
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, src,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as exc:
            raise IKError(
                f"TF {src}->{self.base_frame} 를 찾을 수 없습니다 ({exc}). "
                "calib_tf_publisher(통합) 또는 robot_only.launch 의 identity_tf가 "
                "떠 있는지 확인하세요."
            )

        t = tf.transform.translation
        r = tf.transform.rotation
        R = quat_to_matrix(r.x, r.y, r.z, r.w)
        p = np.array([point.x, point.y, point.z])
        return R @ p + np.array([t.x, t.y, t.z])

    def check_safe(self, p, label):
        radius = float(np.linalg.norm(p[:2]))
        if radius > float(self.get_parameter("workspace_radius").value):
            raise IKError(f"{label} 목표가 안전 반경 밖입니다 (수평거리 {radius:.3f} m)")
        if p[2] < float(self.get_parameter("min_z").value):
            raise IKError(f"{label} 목표가 너무 낮습니다 (z={p[2]:.3f} m). 테이블 평면 캘리브레이션을 확인하세요.")

    # ------------------------------------------------------------------
    def _ik_continuous(self, p, pitch, q_ref):
        """중간점 IK. 팔꿈치 해가 도중에 뒤집히지 않도록 직전 자세에 가까운 해를 고른다."""
        best, best_d = None, None
        for elbow_up in (True, False):
            try:
                q = inverse_kinematics(p, self.geo, pitch, elbow_up)
            except IKError:
                continue
            d = max(abs(a - b) for a, b in zip(q, q_ref))
            if best_d is None or d < best_d:
                best, best_d = q, d
        if best is None:
            raise IKError("중간점 풀이 실패")
        return best

    def move_to(self, target_xyz, pitch, gripper, label=""):
        """현재 TCP에서 목표까지 직교공간 직선 보간으로 이동.

        관절공간만으로 보간하면 두 자세를 잇는 TCP 궤적이 직선이 아니라 휘어서,
        중간에 다른 컵이나 테이블 위를 스칠 수 있다. 그래서 cartesian_step 간격으로
        TCP 직선을 따라가며 각 점에서 IK를 푼다. 중간점이 하나라도 안 풀리면
        기존처럼 관절 보간으로 떨어진다(도달 자체는 보장).
        """
        cands = [pitch, -math.pi / 6, -math.pi / 4, -math.pi / 3, -math.pi / 2]
        q_goal = used_pitch = path = None
        reasons = []
        for cand in dict.fromkeys(cands):            # 순서 유지 중복 제거
            try:
                qg, up = solve_with_fallback(target_xyz, self.geo, pitch_candidates=[cand])
            except IKError as exc:
                reasons.append(f"{math.degrees(cand):.0f}deg IK 불가")
                continue
            cand_path = self._plan_path(target_xyz, qg, up)
            low = self._lowest_tip(cand_path + [list(qg)])
            if low < float(self.get_parameter("min_tip_z").value):
                reasons.append(f"{math.degrees(cand):.0f}deg 손가락 끝 {low * 100:.1f}cm")
                continue
            q_goal, used_pitch, path = qg, up, cand_path
            break
        if q_goal is None:
            raise IKError(f"{label} 목표 {np.round(np.asarray(target_xyz), 3).tolist()} "
                          f"안전한 접근각 없음 (손가락 끝 책상 위 "
                          f"{float(self.get_parameter('min_tip_z').value) * 100:.1f}cm 기준): "
                          + ", ".join(reasons))
        if abs(used_pitch - pitch) > 1e-6:
            self.get_logger().info(
                f"  접근각 {math.degrees(pitch):.0f}deg 불가 -> "
                f"{math.degrees(used_pitch):.0f}deg 로 대체"
            )

        step_time = float(self.get_parameter("step_time").value)
        for q in path:
            if self._cancel:
                return
            self.q = list(q) + [0.0, gripper]
            self.backend.write(self.q)
            time.sleep(step_time)

        self.q = list(q_goal) + [0.0, gripper]
        self.backend.write(self.q)

    def _plan_path(self, target_xyz, q_goal, used_pitch):
        """TCP 직선 보간 경로(관절각 목록). 중간점이 안 풀리면 관절 보간으로."""
        q_start = list(self.q[:4])
        step_len = float(self.get_parameter("cartesian_step").value)
        start = forward_kinematics(q_start, self.geo)
        target = np.asarray(target_xyz, dtype=float)
        dist = float(np.linalg.norm(target - start))

        path = []
        if step_len > 1e-6 and dist > step_len:
            n = max(2, int(math.ceil(dist / step_len)))
            q_ref = q_start
            try:
                for i in range(1, n + 1):
                    # 코사인 이징: 시작/끝이 부드러워 저가 서보에서 진동이 덜하다
                    s = 0.5 - 0.5 * math.cos(math.pi * i / n)
                    q_ref = self._ik_continuous(start + (target - start) * s,
                                                used_pitch, q_ref)
                    path.append(q_ref)
            except IKError:
                path = []
                self.get_logger().info(
                    "  직교 경로 중간점이 안 풀려 관절 보간으로 이동합니다.")

        if not path:
            n = max(6, int(np.max(np.abs(np.array(q_goal) - np.array(q_start))) / 0.03))
            for i in range(1, n + 1):
                s = 0.5 - 0.5 * math.cos(math.pi * i / n)
                path.append([qs + (qg - qs) * s for qs, qg in zip(q_start, q_goal)])
        return path

    def _lowest_tip(self, path):
        """경로에서 고정 손가락 끝의 최저 높이 [m] (base_link z)."""
        extra = float(self.get_parameter("finger_tip_extra").value)
        tip_geo = ArmGeometry(base_height=self.geo.base_height,
                              shoulder_offset=self.geo.shoulder_offset,
                              l1=self.geo.l1, l2=self.geo.l2, l3=self.geo.l3 + extra)
        return min(float(forward_kinematics(q, tip_geo)[2]) for q in path)

    def set_gripper(self, value, settle=0.6):
        if self._cancel:
            return
        # 여는 동작이면 dry-run 파지 시뮬 상태도 함께 해제한다.
        if value > self.q[5] and hasattr(self.backend, "release"):
            self.backend.release()
        self.q[5] = value
        self.backend.write(self.q)
        time.sleep(settle)

    # ------------------------------------------------------------------
    def close_and_verify(self):
        """그리퍼를 닫고, 실제로 물체를 물었는지 확인합니다.

        원리: 물체가 사이에 있으면 그리퍼가 명령한 만큼 닫히지 못하고
        물체 두께에서 멈춥니다. 명령값과 실제값의 차이가 곧 증거입니다.

        빈손이면 끝까지 닫히므로 차이가 거의 0이 됩니다.
        """
        if self._cancel:
            return False

        g_close = float(self.get_parameter("gripper_closed").value)
        margin = float(self.get_parameter("grasp_margin").value)
        settle = float(self.get_parameter("grasp_settle").value)
        min_load = int(self.get_parameter("min_grasp_load").value)

        if isinstance(self.backend, DryRunBackend):
            self.backend.simulate_grasp()

        self.q[5] = g_close
        self.backend.write(self.q)
        time.sleep(settle)

        try:
            actual = self.backend.read()[5]
        except Exception as exc:
            self.get_logger().warn(f"그리퍼 위치를 못 읽었습니다 ({exc}). 성공으로 간주합니다.")
            return True

        gap = actual - g_close
        held = grasp_detected(g_close, actual, margin)
        self.get_logger().info(
            f"  파지 확인: 명령 {g_close:.3f} / 실제 {actual:.3f} "
            f"(차이 {gap:.3f} rad, 기준 {margin:.3f}) -> {'물었음' if held else '빈손'}"
        )

        # 부하 검사 (설정된 경우에만)
        if held and min_load > 0 and hasattr(self.backend, "read_gripper_load"):
            load = self.backend.read_gripper_load()
            if load is not None and load < min_load:
                self.get_logger().warn(f"  부하가 낮습니다 ({load} < {min_load}). 놓친 것 같습니다.")
                held = False

        return held

    def pick_with_retry(self, pick, pitch, fb):
        """집기를 시도하고, 실패하면 위치를 조금씩 바꿔가며 다시 시도합니다.

        한 번 성공률이 80%여도 3회 시도하면 99%가 됩니다.
        투입 대비 효과가 가장 큰 개선입니다.
        """
        approach = float(self.get_parameter("approach_height").value)
        g_open = float(self.get_parameter("gripper_open").value)
        max_retries = int(self.get_parameter("max_grasp_retries").value)
        offset = float(self.get_parameter("retry_offset").value)
        up = np.array([0.0, 0.0, 1.0])

        # 재시도할 때마다 목표를 조금씩 옮깁니다.
        # 매번 똑같은 자리를 찍으면 똑같이 실패하기 때문입니다.
        nudges = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -offset]),          # 조금 더 깊이
            np.array([offset, 0.0, -offset]),       # 앞쪽으로
            np.array([-offset, 0.0, -offset]),      # 뒤쪽으로
        ]

        for attempt in range(max_retries):
            if self._cancel:
                return False
            target = pick + nudges[min(attempt, len(nudges) - 1)]
            # 재시도로 목표를 흔들었으니 안전 검사를 다시 건다.
            self.check_safe(target, "집을(재시도)")

            if attempt > 0:
                self.get_logger().warn(f"파지 재시도 {attempt + 1}/{max_retries}")
                fb(f"RETRY_{attempt + 1}", 0.3)

            fb("APPROACH_PICK", 0.1)
            self.move_to(target + up * approach, pitch, g_open)

            fb("DESCEND", 0.25)
            self.move_to(target, pitch, g_open)

            fb("GRASP", 0.4)
            if self.close_and_verify():
                return True

            # 실패: 열고 살짝 올라간 뒤 다시
            self.set_gripper(g_open, settle=0.4)
            self.move_to(target + up * approach, pitch, g_open)

        return False

    def _resolve_home(self):
        home_q = list(self.get_parameter("home_q").value)
        xyz = list(self.get_parameter("home_xyz").value)
        if len(xyz) != 3:
            return home_q
        pitch = float(self.get_parameter("home_pitch").value)
        # 팔꿈치를 높이 드는 해를 먼저 시도한다. 이 기구학에서 그건 elbow_up=False 쪽이다
        # (TCP (0.22,0,0.16) -45°: False -> 팔꿈치 17cm, True -> 해 없음).
        q4 = None
        for up in (False, True):
            try:
                q4 = inverse_kinematics(xyz, self.geo, approach_pitch=pitch, elbow_up=up)
                break
            except IKError:
                continue
        if q4 is None:
            raise IKError(f"home_xyz {xyz} pitch {pitch:.2f} 를 풀 수 없다")
        q = list(q4) + [0.0, home_q[5] if len(home_q) > 5 else 0.0]
        self.get_logger().info(f"홈 = TCP {xyz} pitch {pitch:.2f} -> q {[round(v, 3) for v in q]}")
        return q

    def _joint_move(self, target):
        """관절 공간 코사인 보간. 가장 많이 움직이는 관절이 joint_speed_deg_s 를 넘지 않게
        시간을 잡는다. 예전엔 거리와 무관하게 0.8초(20스텝)였다 — 먼 자세면 휘두른다."""
        q_start = list(self.q)
        span = max(abs(b - a) for a, b in zip(q_start, target))
        vmax = math.radians(float(self.get_parameter("joint_speed_deg_s").value))
        n = max(10, int(span / vmax / 0.04))
        for i in range(1, n + 1):
            if self._cancel:
                return False
            s = 0.5 - 0.5 * math.cos(math.pi * i / n)
            self.q = [a + (b - a) * s for a, b in zip(q_start, target)]
            self.backend.write(self.q)
            time.sleep(0.04)
        self.q = list(target)
        return True

    def go_home(self):
        self._joint_move(list(self.home))

    # ------------------------------------------------------------------
    def execute(self, goal_handle):
        self._cancel = False
        g = goal_handle.request
        result = PickPlace.Result()

        def fb(phase, progress):
            f = PickPlace.Feedback()
            f.phase = phase
            f.progress = float(progress)
            goal_handle.publish_feedback(f)
            self.pub_phase.publish(String(data=phase))
            self.get_logger().info(f"[{phase}]")

        # dry_run 목표는 실제 백엔드를 잠시 로그 백엔드로 갈아끼워 수행한다.
        # 경고만 찍고 그대로 구동하면 "모터 끄고 좌표만 확인" 단계가 성립하지 않는다.
        real_backend, q_backup = None, list(self.q)
        if g.dry_run and not isinstance(self.backend, DryRunBackend):
            self.get_logger().warn(
                "dry_run 목표 — 서보로 명령을 보내지 않고 좌표/IK만 검증합니다.")
            real_backend = self.backend
            margin = float(self.get_parameter("grasp_margin").value)
            self.backend = DryRunBackend(self.get_logger(), 0.0,
                                         held_gap=margin * 2.0)

        try:
            pick = self.transform_to_base(g.pick, g.frame_id)
            place = self.transform_to_base(g.place, g.frame_id)
            self.check_safe(pick, "집을")
            self.check_safe(place, "놓을")

            self.get_logger().info(
                f"base_link 기준 — 집기: {np.round(pick, 3)}, 놓기: {np.round(place, 3)}"
            )

            approach = float(self.get_parameter("approach_height").value)
            lift = float(self.get_parameter("lift_height").value)
            g_open = float(self.get_parameter("gripper_open").value)
            g_close = float(self.get_parameter("gripper_closed").value)
            pitch = float(g.approach_pitch) if g.approach_pitch != 0.0 else -math.pi / 4

            up = np.array([0.0, 0.0, 1.0])

            if not self.pick_with_retry(pick, pitch, fb):
                if self._cancel or goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = "사용자 취소 — 그 자리에서 정지했습니다."
                    return result
                self.go_home()
                goal_handle.abort()
                result.success = False
                result.message = (
                    f"파지 실패 — {self.get_parameter('max_grasp_retries').value}회 시도했으나 "
                    "물체를 잡지 못했습니다. 물체 위치나 grasp_height를 확인하세요."
                )
                self.get_logger().error(result.message)
                return result

            fb("LIFT", 0.5)
            self.move_to(pick + up * lift, pitch, g_close)

            # 들어올린 뒤 한 번 더 확인 — 옮기는 중에 빠지는 경우가 있습니다
            if not self.close_and_verify():
                self.go_home()
                goal_handle.abort()
                result.success = False
                result.message = "들어올리는 중에 물체를 놓쳤습니다."
                self.get_logger().error(result.message)
                return result

            fb("TRANSFER", 0.65)
            self.move_to(place + up * lift, pitch, g_close)

            fb("PLACE_DESCEND", 0.8)
            self.move_to(place, pitch, g_close)

            fb("RELEASE", 0.9)
            self.set_gripper(g_open)

            fb("RETREAT", 0.95)
            self.move_to(place + up * approach, pitch, g_open)
            self.go_home()

            if self._cancel or goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = "사용자 취소"
                return result

            goal_handle.succeed()
            result.success = True
            result.message = "집어서 옮기기 완료"
            return result

        except SafetyStop as exc:
            # 충돌/막힘/과부하로 멈춘 것이다. 여기서 home 으로 가면 같은 곳에 다시 부딪힐
            # 수 있다 -> 그 자리에 세운 채로 사람을 부른다.
            self.get_logger().error(f"★ 안전 정지 — 복귀하지 않고 그 자리에 멈춤: {exc}")
            self.q = list(self.backend.read())
            goal_handle.abort()
            result.success = False
            result.message = f"안전 정지: {exc}"
            return result
        except IKError as exc:
            self.get_logger().error(f"역기구학 실패: {exc}")
            # 실패 지점에서 그대로 멈추면 팔이 어정쩡한 자세로 남는다.
            self._safe_go_home()
            goal_handle.abort()
            result.success = False
            result.message = str(exc)
            return result
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"동작 중 예외: {exc}")
            self._safe_go_home()
            goal_handle.abort()
            result.success = False
            result.message = f"내부 오류: {exc}"
            return result
        finally:
            # 동작이 어떻게 끝나든 '지금 멈춰 있다'는 것을 GUI에 알린다
            self.pub_phase.publish(String(data=""))
            if real_backend is not None:
                self.backend = real_backend
                # dry_run 동안 self.q 만 움직였을 뿐 실제 팔은 그대로다.
                # 상태를 되돌려 놓지 않으면 다음 실제 동작이 엉뚱한 곳에서 출발한다.
                self.q = q_backup

    def _safe_go_home(self):
        """복귀 시도. 복귀 자체가 실패해도 원래 오류를 덮지 않는다."""
        try:
            self.go_home()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"홈 복귀 실패: {exc}")


def main():
    rclpy.init()
    node = ArmServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.backend.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
