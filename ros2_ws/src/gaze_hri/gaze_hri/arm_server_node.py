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
import json
import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
import tf2_ros
from gaze_hri_msgs.action import PickPlace
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gaze_hri.so101_driver import SafetyStop, min_jerk
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
        # 2026-10-12: 이제 50Hz 로 스트리밍하므로 매번 찍으면 초당 50줄 -> 25번에 한 번만
        self._n = getattr(self, "_n", 0) + 1
        if self._n % 25 == 1:
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


# 2026-10-14: 따를 물병은 쥔 각도만큼 기운 채 들린다 (계획은 그리퍼 수평 = 병 수직 가정) -> 제한
POUR_MAX_GRASP_TILT = 0.0      # 사용자: 물병은 수평으로만 잡는다


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
        # 2026-10-12: 처짐 보정 (관절 1~3 = 어깨/팔꿈치/손목 꺾기). 서보는 P 제어라 무게가
        # 걸리면 목표보다 덜 올라간 채 멈춘다 (팔 a home 에서 어깨 4.1° -> 파지점 1.9cm 낮음,
        # 자로 확인). 도착할 때마다 실제 각도를 읽어 차이만큼 명령에 덧대고, 그 값은 다음
        # 이동에도 이어서 쓴다 (이동 중 뒤처짐도 준다).
        self.bias = [0.0] * 6
        self.fric_ff = 0.0          # 1번 마찰 보상 [rad] — arm_server 가 파라미터로 넣는다

    def write(self, q):
        # SafetyStop 은 그대로 올려 보낸다 -> execute 가 복귀 없이 멈춘다
        prev = getattr(self, "last_q", None)
        self.last_q = list(q)
        cmd = [a + b for a, b in zip(q, self.bias)]
        # 2026-10-12: 1번(pan) 마찰 보상. 궤적 기록상 목표가 움직이는 동안 24% 는 실제가 멈춰
        # 있다가 최대 5틱씩 따라잡았다 (오차 ~1° 쌓여야 정지 마찰을 이김 = 뚝뚝 끊김).
        # 움직이는 방향으로 목표를 fric_ff 만큼 앞서 준다. 멈추면(속도 0) 보상도 0 이 되게
        # 직전 보상에서 서서히 줄인다 (한 번에 빼면 그 자체가 툭 튄다).
        ff = getattr(self, "pan_ff", 0.0)
        fric = getattr(self, "fric_ff", 0.0)
        if fric and prev is not None:
            dq = q[0] - prev[0]
            want = math.copysign(fric, dq) if abs(dq) > 1e-5 else 0.0
            ff += max(-fric / 10, min(fric / 10, want - ff))
        self.pan_ff = ff
        cmd[0] += ff
        goal = self.bus.write_ticks(self.map.to_ticks(cmd))
        self._log_row(goal)

    def start_log(self, path):
        """2026-10-12: 궤적 기록 — 쓸 때마다 (시각, 목표 틱 6, 실제 틱 6). 끊김/떨림을 추측 말고 잰다."""
        self.stop_log()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._log = open(path, "w")
        self._log.write("t," + ",".join(f"goal{i}" for i in range(1, 7)) + ","
                        + ",".join(f"pos{i}" for i in range(1, 7)) + ","
                        + ",".join(f"load{i}" for i in range(1, 7)) + "\n")
        self._log_t0 = time.monotonic()

    def stop_log(self):
        f = getattr(self, "_log", None)
        if f:
            f.close()
        self._log = None

    def _log_row(self, goal):
        f = getattr(self, "_log", None)
        pos = getattr(self.bus, "last_pos", None)
        load = getattr(self.bus, "last_load", None) or [0] * 6
        if f and pos:
            f.write(f"{time.monotonic() - self._log_t0:.4f},"
                    + ",".join(str(int(v)) for v in list(goal) + list(pos) + list(load)) + "\n")

    def settle(self, max_deg, iters, wait=0.25, joints=(1, 2, 3)):
        """목표(마지막 write 의 q)에 실제 관절이 오도록 bias 를 고친다. 바뀌는 bias 는
        10단계로 나눠 써서 툭 튀지 않게 한다. 남은 최대 오차(deg)를 돌려준다."""
        q = getattr(self, "last_q", None)
        if q is None:
            return 0.0
        lim = math.radians(max_deg)
        err = 0.0
        for _ in range(iters):
            time.sleep(wait)
            act = self.read()
            new = list(self.bias)
            for j in joints:
                new[j] = max(-lim, min(lim, self.bias[j] + (q[j] - act[j])))
            err = max(abs(q[j] - act[j]) for j in joints)
            if err < math.radians(0.3):
                break
            old = list(self.bias)
            for k in range(1, 11):
                self.bias = [o + (n - o) * k / 10 for o, n in zip(old, new)]
                self.bus.write_ticks(self.map.to_ticks([a + b for a, b in zip(q, self.bias)]))
                time.sleep(0.02)
        return math.degrees(err)

    def relax(self):
        """2026-10-14: 휴식자세(접힘)에 내려놓은 뒤 토크 OFF — 기다리는 동안 모터 부하 0."""
        self.bus.relax()
        self.relaxed = True

    def wake(self):
        """토크 다시 ON. enable 은 목표=현재로 맞추고 켜므로 켜는 순간 튀지 않는다."""
        self.bus.enable()
        self.bias = [0.0] * 6
        self.pan_ff = 0.0
        self.last_q = None
        self.relaxed = False

    def track_step(self, gain, max_rad, joints=(1, 2, 3)):
        """2026-10-14: 움직이는 중에도 처짐을 조금씩 갚는다 (적분). settle 은 도착 뒤에만 고쳐서,
        가득 찬 물병을 들고 기울이는 동안 어깨·팔꿈치가 각 ~10° 처져 입구가 계획보다 8~11cm
        낮았다(궤적 기록). 직전 write 의 목표(last_q)와 실제 위치 차이에 gain 만큼 bias 를 더한다."""
        q = getattr(self, "last_q", None)
        pos = getattr(self.bus, "last_pos", None)
        if q is None or pos is None:
            return
        act = self.map.to_rad(pos)
        for j in joints:
            self.bias[j] = max(-max_rad, min(max_rad, self.bias[j] + gain * (q[j] - act[j])))

    def reset_bias(self):
        """안전 정지 뒤: 서보 목표가 '지금 위치'로 바뀌었으니 bias 를 덧대면 튄다."""
        self.bias = [0.0] * 6

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
        # 2026-09-29: 작업마다 -55~-30° 중 닿는 각도를 고를지. 켜면 범위는 넓어지지만 멀면 눕혀서
        # 너무 낮게 움직인다는 피드백 -> 기본 끔(-55° 고정, 성공했던 설정).
        self.declare_parameter("adaptive_pitch", False)
        self.declare_parameter("pick_approach_height", 0.0)
        self.declare_parameter("lift_height", 0.12)       # 잡은 뒤 들어올리는 높이
        # 2026-10-12: 궤적 스트리밍 (텔레옵처럼 촘촘한 목표를 일정 주기로). 예전 step_time 방식은
        # 8mm 점마다 쓰기->6축 읽기->sleep 이라 실제 주기가 ~14Hz 로 늘어지고 들쭉날쭉했다.
        self.declare_parameter("stream_hz", 50.0)         # 목표 송신 주기 [Hz]
        self.declare_parameter("tcp_speed", 0.10)         # 직선 이동 TCP 최고 속도 [m/s]
        self.declare_parameter("gripper_speed_deg_s", 90.0)  # 그리퍼 개폐 최고 속도
        self.declare_parameter("cartesian_step", 0.008)   # 직선 경로 IK 샘플 간격 [m]
        # 2026-10-12: 집기 방식. top = 위에서 비스듬히 (기존), side = 그리퍼를 책상과 수평으로
        # 두고 컵 옆에서 밀고 들어가 잡는다 (컵 입구가 그리퍼보다 넓어 위로는 못 끼운다).
        # side 순서: 컵 뒤 standoff 지점 위(high_z) -> 그대로 내려감 -> 수평으로 밀고 들어감 ->
        # 잡기 -> 수직으로 들기(high_z) -> 놓을 곳 위 -> 내려놓기 -> 열기 -> 뒤로 빠짐 -> 위로.
        self.declare_parameter("grasp_mode", "top")
        self.declare_parameter("pour_wait_timeout", 150.0)   # 병 들고 컵 팔 기다리는 최대 시간 [s]
        self.declare_parameter("side_standoff", 0.09)
        # 2026-10-22: 내려갈 때만 이만큼 더 뒤에서 side_mid_z 까지 내려간 뒤 비스듬히 standoff 점으로.
        # 9cm 뒤에서 곧장 내려가면 손가락 끝이 컵 테두리 0.5cm 옆을 지나 계속 쳤다 (팔 B, 사용자).
        # standoff 자체를 11cm 로 늘리면 낮은 점이 받침대에 가까워 IK 불가(집기 37 -> 39cm~, 따르기 자리 못 감)
        # 2026-10-22(2): 2cm 로도 쳤다 -> '닿는 데까지 최대한 뒤로' (사용자, 두 팔). 이 값은 최대치,
        # 위치마다 pre_high·중간점 IK 가 풀리는 가장 먼 점을 1cm 단위로 고른다 (B: 37cm 컵 14cm, 44cm 21cm)
        self.declare_parameter("side_descend_extra", 0.0)
        # 2026-10-22: 컵은 밀면서 잡혀 손 안에서 기준점보다 ~6cm 앞에 있다 -> 놓으면 늘 6cm 바깥 (팔 A 6.0,
        # 6.1, 5.8cm). 놓을 때 그만큼 안쪽을 목표로. 안쪽이라 놓은 뒤 9cm 뒤로는 못 빠져 2~5cm 중 닿는 만큼
        self.declare_parameter("place_hold_offset", 0.0)
        # 2026-10-26: 수평 자세에서 실제 컵 중심이 모델 TCP 보다 팔 방향으로 더 나간다 (빈 컵 테이프 실측:
        # 팔 A 2.2cm, 팔 B 3.5cm). 그만큼 더 들어가 컵을 3~4cm 밀고 잡았다 -> 목표를 팔 쪽으로 당긴다.
        # 집기는 늘, 놓기는 컵 놓기(task "" / grasp_calib)만 (따르기 place 는 컵 위치라 그대로)
        self.declare_parameter("reach_offset", 0.0)
        self.declare_parameter("place_hold_labels", ["cup"])
        self.declare_parameter("place_retreat_max", 0.05)
        self.declare_parameter("place_retreat_min", 0.02)
        self.declare_parameter("side_mid_z", 0.10)     # 컵 중심 -> 밀고 들어가기 시작점 [m]
        self.declare_parameter("side_high_z", 0.20)       # 이동 높이 (TCP, 컵 높이 위) [m]
        self.declare_parameter("side_min_radius", 0.37)   # 이보다 가까우면 수평 자세 불가
        # 처짐 보정 (feetech 전용, FeetechBackend.settle)
        self.declare_parameter("sag_correction", True)
        self.declare_parameter("sag_max_deg", 6.0)        # 관절당 덧대는 최대량
        self.declare_parameter("sag_iters", 2)
        self.declare_parameter("traj_log_dir", "")        # 비우면 궤적 기록 안 함
        self.declare_parameter("pan_friction_ff_deg", 0.0)  # 1번 마찰 보상 (FeetechBackend.write)
        self.declare_parameter("gripper_open", 1.2)       # rad
        self.declare_parameter("gripper_closed", 0.15)    # rad
        # 2026-10-14: 태스크가 끝나면 home 대신 휴식자세로 접고 토크를 끈다 (사용자 요청 —
        # home 에 떠 있으면 어깨가 계속 무게를 버틴다). 휴식자세는 so101_goto.py save-rest 로
        # 저장한 틱 (팔마다 다름, 런치가 arm 으로 파일을 고른다). 파일이 없으면 home 에서 끝낸다.
        # 2026-10-14: 두 팔 동시 구동 — 팔마다 액션/토픽 이름을 나눈다 (dual.launch.py)
        self.declare_parameter("action_name", "/pick_place")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("goto_topic", "/arm/goto_joints")
        # 2026-10-14: 적응형 닫기 — 정해진 위치(gripper_closed)까지 닫는 대신, 닫다가 부하로
        # 닿는 순간을 잡고 거기서 grasp_squeeze 만큼만 더 쥔다. 0.9 고정이면 컵(7cm)은 됐지만
        # 가는 플라스틱 병은 0.9 에서 겨우 닿아(부하 80~160) 빈손 판정/헐겁게 들려 빠졌다.
        self.declare_parameter("grasp_close_mode", "adaptive")   # adaptive | fixed
        self.declare_parameter("gripper_full_close", 0.10)      # 더 닫을 데까지 (서보 한계 ~0)
        # 닿음 = 명령은 계속 닫히는데 실제 위치가 stall_window 동안 stall_deg 도 안 움직임.
        # 부하로는 못 본다 — 빈손으로 닫기만 해도 속도 때문에 부하가 300 넘게 나온다(궤적 기록).
        self.declare_parameter("stall_window", 0.12)
        self.declare_parameter("stall_deg", 1.0)
        # 무른 병은 계속 찌그러져서 '멈춤'이 안 온다 (13:30 병: 끝 0.1 까지 명령, 0.57 에서 부하 900).
        # 명령보다 contact_lag 이상 뒤처지면 그것도 닿음 (빈손 닫힘 뒤처짐은 최대 ~0.17 rad, 기록)
        self.declare_parameter("contact_lag", 0.30)
        self.declare_parameter("grasp_squeeze", 0.25)           # 닿은 위치보다 더 닫으라는 명령 [rad]
        # 2026-10-14: 물 따르기 (gaze_hri/pour.py). roll_sign: 이 팔의 kinematic roll + 가
        # '받침대에서 그리퍼를 볼 때 반시계' 가 되도록 하는 부호. 0 = 아직 확인 안 함 -> 따르기 거부.
        self.declare_parameter("roll_sign", 0)
        self.declare_parameter("pour_bottle_height", 0.20)
        self.declare_parameter("pour_grasp_height", 0.06)       # 병 바닥~쥔 곳 (target_resolver grasp_height 와 같게)
        self.declare_parameter("pour_cup_height", 0.145)
        self.declare_parameter("pour_cup_rim_radius", 0.045)
        self.declare_parameter("pour_lip_clear", 0.05)          # 입구 높이 = 컵 높이 + 이것
        self.declare_parameter("pour_pre_deg", 30.0)
        self.declare_parameter("pour_onset_deg", 50.0)
        self.declare_parameter("pour_max_deg", 105.0)
        self.declare_parameter("pour_fast_deg_s", 20.0)
        self.declare_parameter("pour_slow_deg_s", 6.0)
        self.declare_parameter("pour_hold_s", 1.5)
        # 2026-10-14: 따르기 보정 (리더 텔레옵으로 실제 컵 중심에 맞춘 값을 기록 -> 계획에 반영)
        self.declare_parameter("leader_port", "")
        self.declare_parameter("leader_map_file", "")
        self.declare_parameter("teleop_deg_s", 30.0)            # 보정 중 팔로워 최대 속도 (미세 조정)
        self.declare_parameter("pour_calib_dir", "~/MoLBWA-gaze-hri/notes/pour_calib")
        self.declare_parameter("pour_correction", True)          # 기록이 있으면 계획에 반영
        # latest = 가장 최근 보정 세션만 (같은 자리에서 다듬은 값이 옛 값과 평균되지 않게 — 사용자)
        # all    = 전부 (3개 이상이면 컵 위치에 대한 1차식)
        self.declare_parameter("pour_calib_use", "latest")
        # 2026-10-22: 집기 보정 — 닫기 직전에 멈춰 8/4/2/6 으로 손가락 가운데에 컵을 맞춘 차이를
        # 기록해서 이 팔의 집기·놓기에 더한다 (팔 A 가 컵을 옆으로 미세하게 빗나가 3번 빈손)
        self.declare_parameter("grasp_correction", True)
        # 2026-10-22: 팔별 높이 보정 [m] — 집기·놓기 높이에 더한다. 팔 a 가 명령보다 ~2cm 낮게 다녀
        # (탑다운 캘리브 때 손가락이 책상에 닿고, 4cm 집기에서 바닥에 닿을락 말락 — 사용자) +0.02.
        # 임시값: 근본적으로는 높이를 재서 어깨 0점을 고칠 것.
        self.declare_parameter("z_offset", 0.0)
        self.declare_parameter("homography_file", "")            # 이 팔의 탑다운 호모그래피 (입구 조정 방향)
        self.declare_parameter("pour_calib_deg", 80.0)           # 보정 멈춤 각도 (실제 따르는 자세)
        self.declare_parameter("nudge_step", 0.01)               # 8/4/2/6(화살표) 한 번 = 1cm
        self.declare_parameter("tilt_step_deg", 5.0)             # +/- 한 번 = 5° (입구 고정, 그 점을 축으로)
        # 2026-10-14: 옆 집기 — 수평이 안 닿으면 숙인다 (IK 실측 팔 a: 0° 36~44, 10~20° 32~44, 30° 30~42, 45° 26~39cm)
        self.declare_parameter("side_pitch_options_deg", [0.0])   # 2026-10-14: 사용자 — 잡을 땐 항상 책상과 수평 (숙여 잡기 뺌)
        self.declare_parameter("side_tilt_min_radius", 0.36)
        # 2026-10-14: 물병은 가까우면 위에서 수직으로 내려와 잡는다 (그리퍼는 수평 그대로).
        # 손으로 그려 보니 그리퍼 끝은 23~30cm 까지 수평으로 들어갔다 — 36cm 한계는 뒤 9cm 에서
        # 밀고 들어가는 접근 경로 탓. 컵은 입구(9cm)가 벌림(7.5cm)보다 넓어 위에서 못 내려온다.
        # 2026-10-14(2): 끔 — 내려오면서 물병/옆 컵을 쳤다 (사용자). 다시 켜려면 ["bottle"]
        self.declare_parameter("vertical_labels", [""])
        self.declare_parameter("vertical_min_radius", 0.27)
        self.declare_parameter("arm_name", "")
        self.declare_parameter("pour_track_gain", 0.04)          # 50Hz 에서 시정수 ~0.5s
        self.declare_parameter("pour_sag_max_deg", 20.0)         # 물병 가득: 어깨·팔꿈치 ~10° + 기존 bias
        self.declare_parameter("end_pose", "rest")            # rest | home
        self.declare_parameter("rest_ticks_file", "")
        self.declare_parameter("rest_torque_off", True)
        self.declare_parameter("home_q", [0.0, -0.6, 1.2, -0.6, 0.0, 1.2])
        self.declare_parameter("workspace_radius", 0.42)  # 안전 반경 [m]
        self.declare_parameter("min_z", -0.02)            # 이보다 낮으면 거부 [m]
        # 팔 앞쪽만 (2026-09-24). pan 은 한계상 뒤쪽(-112°)까지 돌 수 있어서, 시선이 팔 뒤의
        # 태그판을 찍으면 팔이 뒤로 돌아 판을 친다. x 가 이보다 작거나 정면에서 max_yaw 넘게
        # 벗어난 목표는 거부한다.
        self.declare_parameter("min_x", 0.12)
        self.declare_parameter("max_yaw_deg", 70.0)

        # ---- 파지 실패 감지 / 재시도 ----
        # 물체를 물었다면 그리퍼가 끝까지 닫히지 못합니다. 그 차이로 판별합니다.
        # grasp_margin: 완전히 닫힌 값보다 이만큼 이상 열려 있어야 "물었다"로 봅니다.
        #               컵 두께의 절반 정도에 해당하는 각도로 잡으세요.
        self.declare_parameter("grasp_margin", 0.08)      # rad
        self.declare_parameter("grasp_settle", 0.9)       # 닫고 나서 대기 [s]
        self.declare_parameter("min_grasp_load", 0)       # 0이면 부하 검사 안 함
        self.declare_parameter("adaptive_held_gap", 0.02)    # 적응형 닫기 후 '물었음' 기준 (빈손 <= 0.009)
        self.declare_parameter("adaptive_held_load", 100)    # 〃 그리퍼 부하 (빈손 <= 36, 가는 물병 116~) — 사용자 100
        self.declare_parameter("grasp_check", "gap")      # gap | load | none (2026-10-12)
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
        if isinstance(self.backend, FeetechBackend):
            self.backend.fric_ff = math.radians(float(self.get_parameter("pan_friction_ff_deg").value))
            # 2026-10-12: 관절 한계를 이 팔의 서보 한계에서 환산 (팔로워 2대, 팔마다 다르다)
            from gaze_hri.so101_driver import joint_limits_rad
            lim = joint_limits_rad(self.backend.bus, self.backend.map)
            self.geo.limits.update(lim)
            self.get_logger().info("관절 한계(서보 기준, deg): " + ", ".join(
                f"{n} {math.degrees(a):.0f}~{math.degrees(b):.0f}" for n, (a, b) in lim.items()))
        self.home = self._resolve_home()
        self.rest_q = self._load_rest()
        if isinstance(self.backend, FeetechBackend):
            # 실제 팔은 home 이 아니라 지금 있는 자세에서 출발한다. home_q 로 두면
            # 첫 보간이 엉뚱한 곳에서 시작해 팔이 한 번에 튄다.
            self._resync_after_stop()
            self.get_logger().info(f"현재 관절각(rad): {[round(v, 3) for v in self.q]}")
            # 휴식자세에서 켜졌으면 토크를 바로 끈다 (기다리는 동안 부하 0, 첫 태스크 때 다시 켬)
            if (self.rest_q is not None and self.get_parameter("rest_torque_off").value
                    and max(abs(a - b) for a, b in zip(self.q[:5], self.rest_q[:5]))
                    < math.radians(8.0)):
                self.backend.relax()
                self.get_logger().info("휴식자세에서 시작 — 토크 OFF")
        else:
            self.q = list(self.home)

        self.pub_js = self.create_publisher(
            JointState, self.get_parameter("joint_states_topic").value, 10)
        # 동작 단계를 토픽으로도 알린다. 액션 피드백은 목표를 보낸 클라이언트만
        # 받기 때문에, GUI 같은 제3자가 진행 상황을 보려면 이게 필요하다.
        self.pub_phase = self.create_publisher(String, "/arm/phase", 10)
        self.create_timer(0.05, self._publish_joint_state)

        # 캘리브레이션 도구가 팔을 원하는 자세로 보낼 때 사용
        self.create_subscription(
            Float64MultiArray, self.get_parameter("goto_topic").value, self.on_goto_joints, 10)

        self._goal_active = False   # execute 진행 중 여부 (goto_joints 동시 실행 방지)
        self._cancel = False
        cb = ReentrantCallbackGroup()
        self._server = ActionServer(
            self, PickPlace, self.get_parameter("action_name").value,
            execute_callback=self.execute,
            goal_callback=lambda g: GoalResponse.ACCEPT,
            cancel_callback=self.on_cancel_request,
            callback_group=cb,
        )
        # 액션 취소와 별개로, 토픽 한 방으로도 즉시 멈출 수 있게 한다.
        # (액션 목표가 아직 accept 되기 전이거나 클라이언트가 죽은 경우 대비)
        self.create_subscription(
            Empty, "/task/cancel", self.on_cancel_topic, 10, callback_group=cb)

        self._teleop_cmd = None
        self.create_subscription(String, "/teleop/cmd", self._on_teleop_cmd, 10,
                                 callback_group=cb)
        # 2026-10-22: 두 팔 동시 따르기 — 병 든 팔은 병을 들고 기다리다가, 컵 팔이 컵을 놓고 빠지기
        # 시작하면 task_manager 가 "go[:x:y]"(컵 바닥 중심, 이 팔 base_link) 를 보낸다. "abort" 면 병 제자리.
        self._gate = None
        gate_topic = str(self.get_parameter("action_name").value).rsplit("/", 1)[0] + "/pour_gate"
        self.create_subscription(String, gate_topic, lambda m: setattr(self, "_gate", m.data), 10,
                                 callback_group=cb)

        self._check_workspace_radius()
        self.get_logger().info(
            f"arm_server 준비 완료 ({self.get_parameter('action_name').value})")

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
        # 코드리뷰(2026-09-28): 작업(execute) 중이면 거부한다. 예전엔 여기서 _cancel 을 무조건
        # 지워서, 작업 중 취소 직후 온 goto_joints 가 그 취소를 삼키고 두 동작 루프가 동시에
        # 모터에 쓸 수 있었다(execute 는 Reentrant 그룹 + MultiThreadedExecutor).
        if self._goal_active:
            self.get_logger().warn("goto_joints: 집기/놓기 작업 중이라 무시합니다. 먼저 /task/cancel.")
            return
        # 2026-09-27: 예전 /task/cancel 로 켜진 _cancel 은 execute() 에서만 지워져서, 그 뒤의
        # goto_joints(홈 등)가 로그 한 줄 없이 무시됐다. 작업이 없을 때의 옛 취소는 새 명령이 대체한다.
        if self._cancel:
            self.get_logger().info("goto_joints: 이전 취소 상태를 해제하고 이동합니다.")
        self._cancel = False
        self.get_logger().info(f"goto_joints: 관절 이동 시작 -> {[round(v, 3) for v in target]}")
        try:
            self._wake_if_relaxed()
            done = self._joint_move(target)
            self.get_logger().info("goto_joints: 도착" if done else
                                   "goto_joints: /task/cancel 로 중단됨")
        except SafetyStop as exc:
            self.get_logger().error(f"안전 정지 (goto_joints): {exc}")
            self._resync_after_stop()

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
        if p[0] < float(self.get_parameter("min_x").value):
            raise IKError(f"{label} 목표가 팔 앞쪽이 아닙니다 (x={p[0]:.3f} m) — 팔 뒤(태그판)로 "
                          "돌지 않도록 거부")
        yaw = math.degrees(math.atan2(p[1], p[0]))
        if abs(yaw) > float(self.get_parameter("max_yaw_deg").value):
            raise IKError(f"{label} 목표가 정면에서 {yaw:.0f}° 벗어났습니다 — 거부")
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

    def move_to(self, target_xyz, pitch, gripper, label="", strict=False):
        """현재 TCP에서 목표까지 직교공간 직선 보간으로 이동.

        관절공간만으로 보간하면 두 자세를 잇는 TCP 궤적이 직선이 아니라 휘어서,
        중간에 다른 컵이나 테이블 위를 스칠 수 있다. 그래서 cartesian_step 간격으로
        TCP 직선을 따라가며 각 점에서 IK를 푼다. 중간점이 하나라도 안 풀리면
        기존처럼 관절 보간으로 떨어진다(도달 자체는 보장).
        """
        # strict: 이 각도가 아니면 의미가 없는 동작(수평 집기) — 다른 각도로 대체하지 않는다
        cands = [pitch] if strict else [pitch, -math.pi / 6, -math.pi / 4, -math.pi / 3,
                                         -math.pi / 2]
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

        # 2026-10-12: 경로(path)는 s=0..1 에 고르게 놓인 관절각 목록이다. 시간축은 최소 저크로
        # 따로 입혀서 stream_hz 로 끊김 없이 보낸다. 관절 속도 제한(2026-09-28, 26° 추종 오차
        # 안전정지 원인)은 _path_duration 이 최고 속도 기준으로 지킨다.
        start = forward_kinematics(self.q[:4], self.geo)
        dist = float(np.linalg.norm(np.asarray(target_xyz, dtype=float) - start))
        # path 의 마지막 점이 곧 목표 IK 해다 (직선: 연속성 유지한 해, 관절: q_goal). q_goal 을
        # 따로 덧붙이지 않는다 — 팔꿈치 해가 다르면 마지막 한 칸에서 튄다.
        grid = [list(self.q[:4])] + [list(q) for q in path]
        self._stream_grid(grid, self.q[5], gripper, self._path_duration(grid, dist))

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
            # 2026-10-12: 접근각도 출발 자세의 각도에서 목표 각도로 경로를 따라 고르게 바꾼다.
            # 예전엔 첫 칸부터 목표 각도로 풀어서 첫 칸에서 손목이 ~50° 꺾였고, 속도 제한을
            # 그 칸에 맞추느라 경로 전체가 수십 초로 늘어졌다 (팔 B, 캘리브 끝 자세 -> 수평 접근 54s+).
            p0 = float(q_start[1] + q_start[2] + q_start[3])
            try:
                for i in range(1, n + 1):
                    # 경로는 기하만 (s 균등). 가감속은 _stream_grid 의 최소 저크가 맡는다.
                    s = i / n
                    q_ref = self._ik_continuous(start + (target - start) * s,
                                                p0 + (used_pitch - p0) * s, q_ref)
                    path.append(q_ref)
            except IKError:
                path = []
                self.get_logger().info(
                    "  직교 경로 중간점이 안 풀려 관절 보간으로 이동합니다.")

        if not path:
            # 관절 공간 직선 (속도/가감속은 _stream_grid 가 정한다)
            path = [list(q_start), list(q_goal)]
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
        self._move_gripper(value)
        time.sleep(settle)

    # ------------------------------------------------------------------
    def close_and_verify(self, reclose=True):
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

        if (self.get_parameter("grasp_close_mode").value == "adaptive"
                and hasattr(self.backend, "bus")):
            # 들어 올린 뒤 재확인(reclose=False)은 다시 닫지 않는다 — 닫으면 그때마다 더 쥔다
            g_close = self._close_adaptive() if reclose else self.q[5]
        else:
            self._move_gripper(g_close)
        time.sleep(settle)

        try:
            actual = self.backend.read()[5]
        except Exception as exc:
            self.get_logger().warn(f"그리퍼 위치를 못 읽었습니다 ({exc}). 성공으로 간주합니다.")
            return True

        gap = actual - g_close
        load = (self.backend.read_gripper_load()
                if hasattr(self.backend, "read_gripper_load") else None)
        # 2026-10-12: 판정 방식. gap = 명령보다 덜 닫혔나 (단단한 물체), load = 그리퍼 부하
        # (플라스틱 컵은 눌려서 명령 위치까지 닫히므로 gap 으로는 늘 '빈손'이었다),
        # none = 판정 안 함 (부하 기준을 실측하는 동안). 어느 방식이든 값은 항상 기록한다.
        mode = self.get_parameter("grasp_check").value
        adaptive = (self.get_parameter("grasp_close_mode").value == "adaptive"
                    and hasattr(self.backend, "bus"))
        if adaptive and mode != "none":
            # 적응형: 무른 병은 눌려서 부하가 낮게 남을 수 있다 -> 명령보다 덜 닫혔거나(무언가
            # 끼었다) 부하가 있으면 물었음. 빈손이면 끝(gripper_full_close)까지 닫혀 둘 다 ~0.
            # 2026-10-22: 가는 물병은 끝까지 닫혀 차이 0.040 / 부하 116 -> 기준(0.05, 120) 바로 밑이라 '빈손'
            # 으로 보고 놓았다 다시 잡기를 반복 (팔 B). 오늘 실측: 빈손 15회 차이 <= 0.009, 부하 <= 36 /
            # 물병 차이 >= 0.040, 부하 >= 116 -> 그 사이 기준
            mode = "adaptive"
            min_load = int(self.get_parameter("adaptive_held_load").value)
            held = (gap > float(self.get_parameter("adaptive_held_gap").value)
                    or (load is not None and load >= min_load))
        elif mode == "none":
            held = True
        elif mode == "load":
            held = load is not None and load >= min_load
        else:
            held = grasp_detected(g_close, actual, margin)
            if held and min_load > 0 and load is not None and load < min_load:
                held = False
        self.get_logger().info(
            f"  파지 확인[{mode}]: 명령 {g_close:.3f} / 실제 {actual:.3f} "
            f"(차이 {gap:.3f} rad, 기준 {margin:.3f}), 그리퍼 부하 {load} (기준 {min_load}) "
            f"-> {'물었음' if held else '빈손'}"
        )
        return held

    def _close_adaptive(self):
        """일정 속도로 닫으며 실제 위치를 본다. 멈추면(닿음) (닿은 위치 - grasp_squeeze) 를 명령해
        물체 크기와 상관없이 같은 만큼만 쥔다. 끝까지 안 멈추면 빈손. 돌려주는 값 = 마지막 명령."""
        bus = self.backend.bus
        full = float(self.get_parameter("gripper_full_close").value)
        win = max(2, int(float(self.get_parameter("stall_window").value)
                         * float(self.get_parameter("stream_hz").value)))
        stall = math.radians(float(self.get_parameter("stall_deg").value))
        squeeze = float(self.get_parameter("grasp_squeeze").value)
        lag = float(self.get_parameter("contact_lag").value)
        hz = float(self.get_parameter("stream_hz").value)
        v = math.radians(float(self.get_parameter("gripper_speed_deg_s").value))
        g0 = self.q[5]
        n = max(1, int(math.ceil(abs(g0 - full) / (v / hz))))
        skip = int(0.15 * hz)        # 출발 직후 (아직 따라 움직이기 전) 은 보지 않는다
        hit = None
        hist = []
        self._moved = True
        t0 = time.monotonic()
        for i in range(1, n + 1):
            if self._cancel:
                return self.q[5]
            self.q[5] = g0 + (full - g0) * i / n
            self.backend.write(self.q)
            load = abs(int(bus.last_load[5])) if bus.last_load else 0
            pos = self.backend.map.to_rad(bus.last_pos)[5]
            hist.append(pos)
            if i > skip and (pos - self.q[5] > lag or (
                    i > skip + win and abs(hist[-1 - win] - pos) < stall
                    and pos - self.q[5] > stall)):
                hit = pos
                break
            time.sleep(max(0.0, t0 + i / hz - time.monotonic()))
        if hit is None:
            self.get_logger().info(f"  적응형 닫기: 끝({full:.2f})까지 닿지 않음")
            return self.q[5]
        target = max(full, hit - squeeze)
        self.get_logger().info(f"  적응형 닫기: {hit:.3f} 에서 닿음 (부하 {load}) -> {target:.3f} 로 쥠")
        self._move_gripper(target)
        return target

    def pick_with_retry(self, pick, pitch, fb):
        """집기를 시도하고, 실패하면 위치를 조금씩 바꿔가며 다시 시도합니다.

        한 번 성공률이 80%여도 3회 시도하면 99%가 됩니다.
        투입 대비 효과가 가장 큰 개선입니다.
        """
        # 2026-09-29: 집을 때만 더 높이서 접근(pick_approach_height). 8cm 는 어깨 처짐까지 더해
        # 손가락 끝이 실제 8~9cm(인형 몸통 중간)로 들어가 옆에서 밀었다. 0 이면 approach_height.
        approach = float(self.get_parameter("pick_approach_height").value) or \
            float(self.get_parameter("approach_height").value)
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
        vmax = math.radians(float(self.get_parameter("joint_speed_deg_s").value))
        span = max(abs(b - a) for a, b in zip(q_start, target))
        duration = max(0.5, 1.875 * span / vmax)       # 최소 저크 최고 속도 = 1.875 x 평균
        done = self._stream(lambda s: [a + (b - a) * s for a, b in zip(q_start, target)],
                            duration)
        if done:
            self._settle()
        return done

    # ------------------------------------------------------------------
    # 궤적 스트리밍 (2026-10-12)
    # ------------------------------------------------------------------
    def _stream(self, q_of, duration, after_write=None):
        """q_of(s) 를 s = min_jerk(t/T) 로 따라가며 stream_hz 주기로 쓴다.

        텔레옵이 부드러운 이유와 같은 구조: 서보 내부 가감속은 끄고(so101_driver), PC 가
        촘촘하고 연속인 목표를 일정 주기로 준다. sleep 은 마감 시각 기준이라 버스 지연이
        흔들려도 주기가 밀리지 않는다."""
        hz = float(self.get_parameter("stream_hz").value)
        n = max(1, int(math.ceil(duration * hz)))
        self._moved = True
        t0 = time.monotonic()
        for i in range(1, n + 1):
            if self._cancel:
                return False
            self.q = list(q_of(min_jerk(i / n)))
            self.backend.write(self.q)
            if after_write is not None:
                after_write()
            time.sleep(max(0.0, t0 + i / hz - time.monotonic()))
        return True

    def _path_duration(self, grid, tcp_dist):
        """경로를 최소 저크로 지날 때 관절 최고 속도와 TCP 최고 속도가 한계를 넘지 않는 시간.
        최소 저크의 최고 ds/dt 는 1.875/T 이고, 관절 속도 = (dq/ds)·(ds/dt)."""
        vmax = math.radians(float(self.get_parameter("joint_speed_deg_s").value))
        vtcp = float(self.get_parameter("tcp_speed").value)
        g = np.asarray(grid, dtype=float)
        n_seg = len(g) - 1
        dq_ds = np.max(np.abs(np.diff(g, axis=0))) * n_seg if n_seg > 0 else 0.0
        return max(0.4, 1.875 * dq_ds / vmax, 1.875 * tcp_dist / vtcp)

    def _stream_grid(self, grid, g_start, g_end, duration):
        """s 균등 격자(관절 4개) 위를 선형 보간하며 스트리밍. 그리퍼도 같이 보간한다."""
        g = np.asarray(grid, dtype=float)
        n_seg = len(g) - 1

        def q_of(s):
            x = s * n_seg
            k = min(int(x), n_seg - 1)
            q4 = g[k] + (g[k + 1] - g[k]) * (x - k)
            return list(q4) + [0.0, g_start + (g_end - g_start) * s]
        done = self._stream(q_of, duration)
        if done:
            self.q = list(g[-1]) + [0.0, g_end]
            self.backend.write(self.q)
            self._settle()
        return done

    def _resync_after_stop(self):
        """안전 정지 뒤 상태 맞추기: 서보 목표 = 현재 위치이므로 bias 도 지운다."""
        if hasattr(self.backend, "reset_bias"):
            self.backend.reset_bias()
        self.q = list(self.backend.read())

    def _settle(self):
        if not (hasattr(self.backend, "settle") and self.get_parameter("sag_correction").value):
            return
        left = self.backend.settle(float(self.get_parameter("sag_max_deg").value),
                                   int(self.get_parameter("sag_iters").value))
        b = getattr(self.backend, "bias", [0.0] * 6)
        self.get_logger().info(f"  처짐 보정 bias(deg) {[round(math.degrees(v), 1) for v in b[1:4]]}, "
                               f"남은 오차 {left:.1f}°")

    def _move_gripper(self, value):
        g0 = self.q[5]
        v = math.radians(float(self.get_parameter("gripper_speed_deg_s").value))
        q4 = list(self.q[:5])
        self._stream(lambda s: q4 + [g0 + (value - g0) * s],
                     max(0.2, 1.875 * abs(value - g0) / v))

    def go_home(self):
        self._joint_move(list(self.home))

    def _load_rest(self):
        path = os.path.expanduser(str(self.get_parameter("rest_ticks_file").value))
        if not path or not hasattr(self.backend, "map"):
            return None
        if not os.path.exists(path):
            self.get_logger().warn(f"휴식자세 파일 없음({path}) — 태스크 끝은 home 에서 멈춤")
            return None
        q = list(self.backend.map.to_rad(json.load(open(path))))
        self.get_logger().info(f"휴식자세(rad): {[round(v, 3) for v in q]}")
        return q

    def go_end(self):
        """태스크 끝 자세. home 을 거쳐 휴식자세로 접고(관절 이동) 토크를 끈다.

        처짐 보정 bias 는 내려가는 동안 0 으로 줄인다 — 접힌 자세에서 bias 를 덧대면
        팔이 자기 몸 쪽으로 밀어붙인다. 도착 뒤 settle 도 하지 않는다 (같은 이유)."""
        if self.get_parameter("end_pose").value != "rest" or self.rest_q is None:
            self.go_home()
            return
        if not self._joint_move(list(self.home)):
            return
        q_start, target = list(self.q), list(self.rest_q)
        b0 = list(getattr(self.backend, "bias", [0.0] * 6))
        vmax = math.radians(float(self.get_parameter("joint_speed_deg_s").value))
        span = max(abs(b - a) for a, b in zip(q_start, target))
        if not self._stream(lambda s: [a + (b - a) * s - bb * s
                                       for a, b, bb in zip(q_start, target, b0)],
                            max(0.5, 1.875 * span / vmax)):
            return
        if hasattr(self.backend, "reset_bias"):
            self.backend.reset_bias()
        self.q = target
        if self.get_parameter("rest_torque_off").value and hasattr(self.backend, "relax"):
            time.sleep(0.5)
            self.backend.relax()
            self.get_logger().info("휴식자세 도착 — 토크 OFF (다음 태스크 때 다시 켭니다)")

    def _wake_if_relaxed(self):
        if getattr(self.backend, "relaxed", False):
            self.backend.wake()
            self.q = list(self.backend.read())
            self.get_logger().info("토크 다시 ON (휴식자세에서 출발)")

    # ------------------------------------------------------------------
    def execute(self, goal_handle):
        self._cancel = False
        self._gate = None
        g = goal_handle.request
        # 2026-10-22: "pour_wait"/"pour_calib_wait" = 병을 들고 pour_gate 신호를 기다렸다가 따른다
        task = getattr(g, "task", "") or ""
        wait = task.endswith("_wait")
        task = task[:-len("_wait")] if wait else task
        self._pick_label = getattr(g, "pick_label", "") or ""   # 물병이면 가까울 때 위에서 수직 접근
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

        self._goal_active = True     # finally 에서 반드시 내린다
        self._moved = False          # 이번 목표에서 팔을 움직였나 (거부만 했으면 home 안 감)
        log_dir = os.path.expanduser(str(self.get_parameter("traj_log_dir").value))
        if log_dir and hasattr(self.backend, "start_log"):
            arm = self.get_parameter("arm_name").value or "arm"   # 두 팔이 같은 초에 시작하면 한 파일에 섞였다
            path = os.path.join(log_dir, time.strftime(f"traj_%m%d_%H%M%S_{arm}.csv"))
            self.backend.start_log(path)
            self.get_logger().info(f"궤적 기록: {path}")
        try:
            pick = self.transform_to_base(g.pick, g.frame_id)
            place = self.transform_to_base(g.place, g.frame_id)
            dz = float(self.get_parameter("z_offset").value)
            if dz:
                pick = pick + np.array([0.0, 0.0, dz])
                if task not in ("pour", "pour_calib"):   # 따르기 place 는 컵 축(높이 무관)
                    place = place + np.array([0.0, 0.0, dz])
                self.get_logger().info(f"  팔별 높이 보정 {dz * 100:+.1f}cm")
            # 2026-10-22: 집기 보정(xy)도 여기서 — 예전엔 _execute_side 안에서 더해서, 움직이기 전 검사는
            # 보정 전 위치로 통과하고 보정 후 위치가 IK 불가라 home 까지 간 뒤 실패했다 (팔 B, 36.5cm)
            go = self._grasp_offset()
            self._grasp_cam_pick = np.array(pick, float)
            if go is not None and np.linalg.norm(go) > 1e-6:
                pick = pick + np.array([go[0], go[1], 0.0])
                if task in ("", "grasp_calib"):   # 같은 물체를 놓을 때만 (2026-10-26: 물체별 보정)
                    place = place + np.array([go[0], go[1], 0.0])
                self.get_logger().info(f"  집기 보정 적용({self._pick_label}): {np.round(go * 100, 1).tolist()} cm")
            if task not in ("", "grasp_calib"):
                gc = self._grasp_offset("cup")      # 따르기의 컵 위치는 컵 보정(없으면 그대로)
                if gc is not None:
                    place = place + np.array([gc[0], gc[1], 0.0])
            ro = float(self.get_parameter("reach_offset").value)
            if ro and self.get_parameter("grasp_mode").value == "side":
                # 2026-10-26: 따르기의 컵 위치에도 — 빼면 실제로 닿는 컵(43cm)을 IK 불가로 거부했다
                pick, place = self._reach_pull(pick), self._reach_pull(place)
                self.get_logger().info(f"  팔별 뻗음 보정 {ro * 100:.1f}cm (팔 쪽으로)")
            self._place_standoff = None
            hold = float(self.get_parameter("place_hold_offset").value)
            if (hold > 0.0 and task in ("", "grasp_calib")
                    and self.get_parameter("grasp_mode").value == "side"
                    and self._pick_label in list(self.get_parameter("place_hold_labels").value)):
                u = np.array([place[0], place[1], 0.0]) / max(float(np.linalg.norm(place[:2])), 1e-9)
                place = place - u * hold
                self._place_standoff = self._place_retreat(place)
                self.get_logger().info(
                    f"  놓을 곳 {hold * 100:.0f}cm 안쪽으로 (손 안의 컵이 앞에 있음), 놓은 뒤 "
                    f"{self._place_standoff * 100:.1f}cm 뒤로 빠짐")
            elif (ro and task in ("", "grasp_calib")
                    and self.get_parameter("grasp_mode").value == "side"):
                # 2026-10-26: 밀림 보정을 0 으로 두면 기본 9cm 뒤 접근점이 받침대 28cm 라 IK 가 안 풀렸다
                # (따르기 자리, 팔 B) -> 뻗음 보정을 쓸 때는 놓은 뒤 닿는 만큼(2~5cm)만 빠진다
                self._place_standoff = self._place_retreat(place)
                self.get_logger().info(f"  놓은 뒤 {self._place_standoff * 100:.1f}cm 뒤로 빠짐")
            self.check_safe(pick, "집을")
            self.check_safe(place, "놓을")
            if task in ("pour", "pour_calib"):
                self._pour_plan_pulled(pick, place)   # 안 되면 IKError (아직 안 움직임). 끝이면 조금 당겨서
                self._side_pitch(pick, "병", max_tilt_deg=POUR_MAX_GRASP_TILT)
            elif self.get_parameter("grasp_mode").value == "side":
                # 2026-10-14: 움직이기 전에 집기/놓기 둘 다 닿는지 실제 IK 로 확인 (안 되면 다른 팔에게)
                self._side_pitch(pick, "집을")
                self._side_pitch(place, "놓을", standoff=self._place_standoff)

            self.get_logger().info(
                f"base_link 기준 — 집기: {np.round(pick, 3)}, 놓기: {np.round(place, 3)}"
            )

            self._wake_if_relaxed()
            # 2026-10-14: 휴식자세(접힘)는 모델상 손가락 끝이 책상 아래(-4cm)로 계산돼, 거기서
            # 출발하는 경로는 전부 손가락 끝 검사에 걸려 거부됐다(팔 B, "선택하면 초기화됨").
            # 출발 자세가 안전 높이보다 낮으면 먼저 home 으로 관절 이동해 펴고 시작한다.
            if self._lowest_tip([list(self.q[:4])]) < float(
                    self.get_parameter("min_tip_z").value):
                self.get_logger().info("  출발 자세가 낮음(휴식자세) — 먼저 home 으로 폅니다.")
                fb("HOME", 0.05)
                self.go_home()
                if self._cancel:
                    self._end_cancel(goal_handle)
                    result.success = False
                    result.message = "사용자 취소 — 그 자리에서 정지했습니다."
                    return result

            if task in ("pour", "pour_calib"):
                return self._execute_pour(pick, place, fb, goal_handle, result,
                                          calib=task == "pour_calib", wait=wait)
            if self.get_parameter("grasp_mode").value == "side":
                return self._execute_side(pick, place, fb, goal_handle, result,
                                          grasp_calib=task == "grasp_calib")

            approach = float(self.get_parameter("approach_height").value)
            lift = float(self.get_parameter("lift_height").value)
            g_open = float(self.get_parameter("gripper_open").value)
            g_close = float(self.get_parameter("gripper_closed").value)
            pitch = float(g.approach_pitch) if g.approach_pitch != 0.0 else -math.pi / 4
            # 2026-09-29: 작업마다 접근각 하나를 고른다. 요청 각도부터 5°씩 눕혀 가며, 집기·놓기
            # 모든 지점(접근/파지/들기/놓기 위/놓기)에서 IK 가 풀리는 가장 가파른 각도를 쓴다.
            # 도중에 각도를 바꾸지 않으므로(손목이 휘며 물체를 밀던 문제) 범위만 넓어진다
            # (-55° 고정 29cm -> 최대 -30° 까지 ~36cm).
            # 2026-09-29 수정: 집기/놓기 각도를 따로 고른다. 하나로 묶으면 먼 쪽(놓을 곳)에 맞춰
            # 집을 때도 -40° 로 눕혀 들어가 인형을 밀었다. 각도는 공중 이동(TRANSFER) 때만 바뀐다.
            base_pitch = pitch
            place_pitch = pitch
            if bool(self.get_parameter("adaptive_pitch").value):
                up_ = np.array([0.0, 0.0, 1.0])
                pitch = self._choose_task_pitch(
                    base_pitch, [pick + up_ * approach, pick, pick + up_ * lift], "집기")
                place_pitch = self._choose_task_pitch(
                    base_pitch, [place + up_ * lift, place, place + up_ * approach], "놓기")

            up = np.array([0.0, 0.0, 1.0])

            if not self.pick_with_retry(pick, pitch, fb):
                if self._cancel or goal_handle.is_cancel_requested:
                    self._end_cancel(goal_handle)
                    result.success = False
                    result.message = "사용자 취소 — 그 자리에서 정지했습니다."
                    return result
                self.go_end()
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
                self.go_end()
                goal_handle.abort()
                result.success = False
                result.message = "들어올리는 중에 물체를 놓쳤습니다."
                self.get_logger().error(result.message)
                return result

            fb("TRANSFER", 0.65)
            self.move_to(place + up * lift, place_pitch, g_close)

            fb("PLACE_DESCEND", 0.8)
            self.move_to(place, place_pitch, g_close)

            fb("RELEASE", 0.9)
            self.set_gripper(g_open)

            fb("RETREAT", 0.95)
            self.move_to(place + up * approach, place_pitch, g_open)
            self.go_end()

            if self._cancel or goal_handle.is_cancel_requested:
                self._end_cancel(goal_handle)
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
            self._resync_after_stop()
            goal_handle.abort()
            result.success = False
            result.message = f"안전 정지: {exc}"
            return result
        except IKError as exc:
            self.get_logger().error(f"역기구학 실패: {exc}")
            if not getattr(self, "_moved", True):
                # 팔을 움직이기 전에 거부 -> task_manager 가 다른 팔에게 넘길 수 있게 표시
                goal_handle.abort()
                result.success = False
                result.message = f"[거부] {exc}"
                return result
            # 실패 지점에서 그대로 멈추면 팔이 어정쩡한 자세로 남는다.
            # 2026-10-12: 단, 아직 한 번도 안 움직였으면(범위 밖 목표 거부) 그대로 둔다.
            if getattr(self, "_moved", True):
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
            if hasattr(self.backend, "stop_log"):
                self.backend.stop_log()
            self._goal_active = False
            # 동작이 어떻게 끝나든 '지금 멈춰 있다'는 것을 GUI에 알린다
            self.pub_phase.publish(String(data=""))
            if real_backend is not None:
                self.backend = real_backend
                # dry_run 동안 self.q 만 움직였을 뿐 실제 팔은 그대로다.
                # 상태를 되돌려 놓지 않으면 다음 실제 동작이 엉뚱한 곳에서 출발한다.
                self.q = q_backup

    # ------------------------------------------------------------------
    # 수평 옆 집기 (2026-10-12)
    # ------------------------------------------------------------------
    def _side_points(self, p, label, standoff=None):
        """p(파지점)의 반경 방향 u, 밀고 들어가기 시작점(낮은/높은), p 바로 위(높은).
        standoff: 놓을 곳을 안쪽으로 당겼을 때 빠지는 거리 (그때는 집기 최소 거리 검사 안 함)."""
        r = float(np.linalg.norm(p[:2]))
        r_min = float(self.get_parameter(
            "vertical_min_radius" if self._vertical_ok() else "side_tilt_min_radius").value)
        if standoff is None and r < r_min - 0.002:     # 판정은 task_manager 와 같은 값 (반올림 37.0 이 거부됐었다)
            raise IKError(f"{label} 위치가 받침대에서 {r * 100:.1f}cm — 옆 집기는 "
                          f"{r_min * 100:.0f}cm 이상이어야 한다 (그리퍼 20cm + 손목 한계)")
        u = np.array([p[0] / r, p[1] / r, 0.0])
        d = float(self.get_parameter("side_standoff").value) if standoff is None else standoff
        z_hi = float(self.get_parameter("side_high_z").value)
        pre_low = p - u * d
        far = pre_low - u * self._descend_extra(pre_low)
        pre_high = np.array([far[0], far[1], z_hi])
        above = np.array([p[0], p[1], z_hi])
        return u, pre_low, pre_high, above

    def _descend_extra(self, pre_low):
        """pre_low 에서 받침대 쪽으로 더 물러날 거리 — 높은 점(side_high_z)과 중간점(side_mid_z)이
        둘 다 수평 그리퍼로 풀리는 가장 먼 거리 (최대 side_descend_extra, 1cm 단위)."""
        e_max = float(self.get_parameter("side_descend_extra").value)
        if e_max <= 0.0:
            return 0.0
        key = (self._key(pre_low), round(e_max, 3))
        cache = self.__dict__.setdefault("_extra_cache", {})
        if key in cache:
            return cache[key]
        r = float(np.linalg.norm(pre_low[:2]))
        u = np.array([pre_low[0] / r, pre_low[1] / r, 0.0])
        zs = (float(self.get_parameter("side_high_z").value),
              max(float(self.get_parameter("side_mid_z").value), float(pre_low[2])))
        best = 0.0
        for k in range(int(round(e_max * 100)), 0, -1):
            q = pre_low - u * (k / 100.0)
            try:
                for z in zs:
                    solve_with_fallback([q[0], q[1], z], self.geo, pitch_candidates=[0.0])
            except IKError:
                continue
            best = k / 100.0
            break
        cache[key] = best
        return best

    def _pour_plan_pulled(self, bottle, cup, max_pull=0.05, step=0.005):
        """2026-10-26: 놓인 컵이 따르기 도달 끝(병 쥔 손이 컵보다 5~7cm 바깥, 높이 23cm)을 1~3cm 넘으면
        병을 그냥 내려놨다. 컵 목표를 이 팔 쪽으로 0.5cm 씩 최대 max_pull 당겨 풀리는 곳에서 따른다
        (입구가 그만큼 팔 쪽으로 치우침 -> 따르기 보정 p 로 맞춘다)."""
        try:
            return self._pour_plan(bottle, cup), cup
        except IKError as first:
            r = float(np.linalg.norm(cup[:2]))
            d = step
            while d <= max_pull + 1e-9 and r > d:
                c = np.array([cup[0] * (r - d) / r, cup[1] * (r - d) / r, cup[2]])
                try:
                    plan = self._pour_plan(bottle, c)
                    self.get_logger().warn(f"  따르기 도달 끝 — 컵 목표를 팔 쪽으로 {d * 100:.1f}cm 당겨 계획"
                                           " (입구가 그만큼 팔 쪽, 따르기 보정으로 맞출 것)")
                    return plan, c
                except IKError:
                    d += step
            raise first

    def _reach_pull(self, q):
        """수평 자세에서 실제가 모델보다 더 뻗는 만큼(reach_offset) 목표를 받침대 쪽으로 당긴다."""
        ro = float(self.get_parameter("reach_offset").value)
        r = float(np.linalg.norm(q[:2]))
        if not ro or r < 1e-6:
            return q
        return np.array([q[0] * (r - ro) / r, q[1] * (r - ro) / r, q[2]])

    def _place_retreat(self, place):
        """안쪽으로 당긴 놓을 곳에서 뒤로 빠질 거리: max~min 을 0.5cm 씩 줄이며 IK 가 풀리는 가장 큰 값."""
        hi = float(self.get_parameter("place_retreat_max").value)
        lo = float(self.get_parameter("place_retreat_min").value)
        s = hi
        while s >= lo - 1e-9:
            try:
                self._side_pitch(place, "놓을", standoff=s)
                return s
            except IKError:
                s -= 0.005
        raise IKError(f"놓을 위치 {np.round(place, 3).tolist()} 에서 뒤로 {lo * 100:.0f}cm 도 못 빠집니다")

    def _side_mid(self, pre_low):
        """테두리 높이를 지날 때 거치는 점 (더 뒤, side_mid_z). 더 물러날 수 없으면 None."""
        e = self._descend_extra(pre_low)
        if e <= 0.0:
            return None
        r = float(np.linalg.norm(pre_low[:2]))
        m = pre_low - np.array([pre_low[0] / r, pre_low[1] / r, 0.0]) * e
        m[2] = max(float(self.get_parameter("side_mid_z").value), pre_low[2])
        return m

    def _via_mid(self, pre_low, pitch, grip):
        m = self._side_mid(pre_low)
        if m is not None:
            self.get_logger().info(
                f"  물체 뒤 {(float(self.get_parameter('side_standoff').value) + self._descend_extra(pre_low)) * 100:.0f}cm, "
                f"높이 {m[2] * 100:.0f}cm 을 거침")
            self.move_to(m, pitch, grip, strict=True)

    def _side_pitch(self, p, label="집을", max_tilt_deg=None, standoff=None):
        if not hasattr(self, "_vertical_at"):
            self._vertical_at = set()
        self._vertical_at.discard(self._key(p))
        """옆 집기 접근각. 수평(0)이 기본이고, 36cm 보다 가까워 수평으로 안 닿으면 10°, 20° 숙인다
        (2026-10-14: 수평 범위가 36~44cm 로 좁아 물병이 34.8cm 에서 계속 거부됐다).
        p, 뒤 접근점(낮은/높은) 셋 다 그 각도로 풀려야 한다."""
        _, pre_low, pre_high, _ = self._side_points(p, label, standoff)
        for deg in self.get_parameter("side_pitch_options_deg").value:
            if max_tilt_deg is not None and abs(float(deg)) > max_tilt_deg + 1e-6:
                continue
            pitch = math.radians(float(deg))
            try:
                for q in (pre_high, self._side_mid(pre_low), pre_low, p):
                    if q is not None:
                        solve_with_fallback(list(q), self.geo, pitch_candidates=[pitch])
            except IKError:
                continue
            if deg:
                self.get_logger().info(f"  {label} 위치가 가까워 그리퍼를 {abs(deg):.0f}° 숙여 집습니다")
            return pitch
        if self._vertical_ok():
            above = np.array([p[0], p[1], pre_high[2]])
            try:
                for q in (above, p):
                    solve_with_fallback(list(q), self.geo, pitch_candidates=[0.0])
                self._vertical_at.add(self._key(p))
                self.get_logger().info(f"  {label} 위치가 가까워 위에서 수직으로 내려와 잡습니다 (그리퍼 수평)")
                return 0.0
            except IKError:
                pass
        raise IKError(f"{label} 위치 {np.round(p, 3).tolist()} 는 옆 집기 각도 "
                      f"{list(self.get_parameter('side_pitch_options_deg').value)}° 어느 것으로도 안 닿습니다")

    def _vertical_ok(self):
        return getattr(self, "_pick_label", "") in list(self.get_parameter("vertical_labels").value)

    @staticmethod
    def _key(p):
        return (round(float(p[0]), 3), round(float(p[1]), 3))

    def _is_vertical(self, p):
        return self._key(p) in getattr(self, "_vertical_at", set())

    def _side_pick(self, pick, fb, max_tilt_deg=None):
        g_open = float(self.get_parameter("gripper_open").value)
        off = float(self.get_parameter("retry_offset").value)
        u, pre_low, pre_high, above = self._side_points(pick, "집을")
        pp = self._side_pitch(pick, max_tilt_deg=max_tilt_deg)
        z = np.array([0.0, 0.0, 1.0])
        if self._is_vertical(pick):
            # 위에서 수직으로: 바로 위 -> 내려가 잡기 -> 실패면 다시 위로 (조금 위/아래로 재시도)
            for attempt, dz in enumerate([0.0, off, -off][:int(self.get_parameter("max_grasp_retries").value)]):
                if self._cancel:
                    return False
                fb("APPROACH_PICK" if attempt == 0 else f"RETRY_{attempt + 1}", 0.1 + 0.1 * attempt)
                self.move_to(above, 0.0, g_open, strict=True)
                fb("DESCEND", 0.3)
                self.move_to(pick + z * dz, 0.0, g_open, strict=True)
                fb("GRASP", 0.4)
                if self.close_and_verify():
                    return True
                self.set_gripper(g_open, settle=0.4)
                self.move_to(above, 0.0, g_open, strict=True)
            return False
        # 재시도마다: 그대로 / 옆(왼) / 옆(오른) / 조금 더 깊이 — 2026-10-22: 빗나감은 대개 옆으로
        # (팔 A 가 깊이·위·아래 재시도로도 못 잡음)
        t = np.array([-u[1], u[0], 0.0])
        nudges = [np.zeros(3), t * off, -t * off, u * off]
        for attempt in range(int(self.get_parameter("max_grasp_retries").value)):
            if self._cancel:
                return False
            target = pick + nudges[min(attempt, len(nudges) - 1)]
            if attempt == 0:
                fb("APPROACH_PICK", 0.1)
                self.move_to(pre_high, pp, g_open, strict=True)
                if getattr(self, "_grasp_calib", False):
                    # 2026-10-22: 내려가기 **전에** 멈춰 맞춘다 — 내려가다 손가락이 컵 옆을 쳤다 (사용자)
                    off = self._grasp_calib_stage(pre_high, pp)
                    if off is None:
                        return False
                    d3 = np.array([off[0], off[1], 0.0])
                    pick, pre_low, pre_high = pick + d3, pre_low + d3, pre_high + d3
                    target = pick + nudges[0]
                fb("DESCEND", 0.2)
                self._via_mid(pre_low, pp, g_open)
                self.move_to(pre_low, pp, g_open, strict=True)
            else:
                self.get_logger().warn(f"파지 재시도 {attempt + 1}")
                fb(f"RETRY_{attempt + 1}", 0.3)
                self.move_to(pre_low + (target - pick), pp, g_open, strict=True)
            fb("INSERT", 0.3)
            self.move_to(target, pp, g_open, strict=True)
            fb("GRASP", 0.4)
            if self.close_and_verify():
                return True
            self.set_gripper(g_open, settle=0.4)        # 실패: 열고 뒤로 빠졌다가 다시
            self.move_to(pre_low + (target - pick), pp, g_open, strict=True)
        return False

    def _execute_side(self, pick, place, fb, goal_handle, result, grasp_calib=False):
        self._grasp_calib = grasp_calib          # 집기 보정은 execute 에서 이미 더함
        g_open = float(self.get_parameter("gripper_open").value)
        g_close = float(self.get_parameter("gripper_closed").value)
        _, _, _, pick_above = self._side_points(pick, "집을")
        st = getattr(self, "_place_standoff", None)
        _, place_pre_low, place_pre_high, place_above = self._side_points(place, "놓을", st)
        pp = self._side_pitch(pick)
        lp = self._side_pitch(place, "놓을", standoff=st)

        picked = self._side_pick(pick, fb)
        g_close = self.q[5]           # 실제로 쥔 명령을 유지 (적응형 닫기 — 물체마다 다르다)
        if not picked:
            if self._cancel or goal_handle.is_cancel_requested:
                self._end_cancel(goal_handle)
                result.success = False
                result.message = "사용자 취소 — 그 자리에서 정지했습니다."
                return result
            self._side_retreat_home(pick, g_open)
            goal_handle.abort()
            result.success = False
            result.message = "파지 실패 — 컵 위치(탑다운 클릭)나 grasp_height 를 확인하세요."
            self.get_logger().error(result.message)
            return result

        fb("LIFT", 0.5)
        self.move_to(pick_above, pp, g_close, strict=True)
        if not self.close_and_verify(reclose=False):
            goal_handle.abort()
            result.success = False
            result.message = "들어올리는 중에 컵을 놓쳤습니다."
            self.get_logger().error(result.message)
            self.go_end()
            return result

        fb("TRANSFER", 0.65)
        self.move_to(place_above, lp, g_close, strict=True)
        fb("PLACE_DESCEND", 0.8)
        self.move_to(place, lp, g_close, strict=True)
        fb("RELEASE", 0.9)
        self.set_gripper(g_open)
        fb("RETREAT", 0.95)
        if self._is_vertical(place):
            self.move_to(place_above, 0.0, g_open, strict=True)     # 위에서 내려놨으면 위로 빠진다
        else:
            self.move_to(place_pre_low, lp, g_open, strict=True)
            self._via_mid(place_pre_low, lp, g_open)
            self.move_to(place_pre_high, lp, g_open, strict=True)
        self.go_end()

        if self._cancel or goal_handle.is_cancel_requested:
            self._end_cancel(goal_handle)
            result.success = False
            result.message = "사용자 취소"
            return result
        goal_handle.succeed()
        result.success = True
        result.message = "집어서 옮기기 완료 (수평 옆 집기)"
        return result

    # ------------------------------------------------------------------
    # 물 따르기 (2026-10-14)
    # ------------------------------------------------------------------
    def _pour_plan(self, bottle, cup, raw=False, end_shift_override=None, angles=None):
        """따르기 계획. 끝 각도(보정 기록 150° 등)까지 그 컵 위치에서 팔이 안 닿으면 5° 씩 줄여
        닿는 데까지만 기울인다 (2026-10-14: 팔 B 148° 에서 쥔 곳 33.5cm -> IK 불가로 계속 거부)."""
        try:
            return self._pour_plan_once(bottle, cup, raw, end_shift_override, angles)
        except IKError as first:
            if "따르기 IK 불가" not in str(first):
                raise
            err = first
        for cap in range(150, 79, -5):
            try:
                plan = self._pour_plan_once(bottle, cup, raw, end_shift_override, angles, max_cap=cap)
            except IKError as exc:
                err = exc
                continue
            self.get_logger().warn(f"  이 컵 위치에서는 {plan['cfg'].max_deg:.0f}° 까지만 기울일 수 있어 "
                                   f"거기까지만 따릅니다")
            return plan
        raise err

    def _pour_plan_once(self, bottle, cup, raw=False, end_shift_override=None, angles=None,
                        max_cap=None):
        """경로 전체를 움직이기 **전에** 계획하고 IK 를 다 풀어 본다. 안 되면 IKError."""
        from gaze_hri.pour import PourConfig, PourError, plan_pour
        sign = int(self.get_parameter("roll_sign").value)
        if sign not in (-1, 1):
            raise IKError("roll_sign 이 설정되지 않았습니다 — 손목 roll 방향 확인부터 (따르기 거부)")
        gp = self.get_parameter
        cfg = PourConfig(
            bottle_height=float(gp("pour_bottle_height").value),
            grasp_height=float(gp("pour_grasp_height").value),
            cup_height=float(gp("pour_cup_height").value),
            cup_rim_radius=float(gp("pour_cup_rim_radius").value),
            lip_clear=float(gp("pour_lip_clear").value),
            pre_deg=float(gp("pour_pre_deg").value),
            onset_deg=float(gp("pour_onset_deg").value),
            max_deg=float(gp("pour_max_deg").value),
            fast_deg_s=float(gp("pour_fast_deg_s").value),
            slow_deg_s=float(gp("pour_slow_deg_s").value),
            hold_s=float(gp("pour_hold_s").value))
        if angles is not None:                 # 보정 중: 사용자가 정한 시작/종료 각도
            cfg.onset_deg, cfg.max_deg = angles
        cup_xy = np.asarray(cup[:2], float)
        end_shift = None
        if end_shift_override is not None:
            end_shift = np.asarray(end_shift_override, float)
        cm = None if raw else self._pour_calib_model(cup_xy)
        if cm:
            # 2026-10-14: 보정 기록 — 시작/종료 자세에서 따로 맞춘 입구 위치와 각도
            if cm.get("th_start") is not None:
                cfg.onset_deg = min(cm["th_start"], cfg.max_deg - 5.0)
            if cm.get("th_end") is not None:
                cfg.max_deg = max(cm["th_end"], cfg.onset_deg + 5.0)
            cs, ce = cm.get("start"), cm.get("end")
            if cs is None:
                cs = ce
            if cs is not None:
                end_shift = (ce - cs) if ce is not None else None
                self.get_logger().info(
                    f"  따르기 보정 적용: 시작 {np.round(cs * 100, 1).tolist()}cm"
                    + (f", 종료 {np.round(ce * 100, 1).tolist()}cm" if ce is not None else "")
                    + f", 각도 {cfg.onset_deg:.0f}° -> {cfg.max_deg:.0f}°")
                cup_xy = cup_xy + cs
        if max_cap is not None:
            cfg.max_deg = min(cfg.max_deg, float(max_cap))
            if cfg.max_deg < cfg.onset_deg + 5.0:
                raise IKError("따르기 IK 불가 (최대 각도가 시작 각도보다 작음)")
        try:
            plan = plan_pour(bottle, cup_xy, cfg, end_shift=end_shift)
            plan["corr_start"] = (cup_xy - np.asarray(cup[:2], float))
            plan["corr_end"] = plan["corr_start"] + (np.zeros(2) if end_shift is None
                                                     else np.asarray(end_shift, float)[:2])
        except PourError as exc:
            raise IKError(f"따르기 경로 불가: {exc}") from exc
        prev = None
        sign_s = sign * plan["s"]

        def to_q(G, th):
            nonlocal prev
            try:
                q = self._ik_continuous(G, 0.0, prev or list(self.home))
            except IKError as exc:
                raise IKError(f"따르기 IK 불가 (θ={math.degrees(th):.0f}°, 쥔 곳 "
                              f"{np.round(G, 3).tolist()})") from exc
            prev = list(q)
            return list(q[:4]) + [sign_s * th]
        start_q = to_q(plan["start"], 0.0)
        segs = {}
        for name in ("pre", "move", "onset", "pour"):
            segs[name] = [(to_q(G, th), dt) for G, th, dt in plan["segments"][name]]
        plan["start_q"], plan["qsegs"] = start_q, segs
        for p in (plan["carry"], plan["start"]):
            solve_with_fallback(list(p), self.geo, pitch_candidates=[0.0])
        plan["cfg"] = cfg
        return plan

    # ------------------------------------------------------------------
    # 따르기 보정: 리더 텔레옵으로 맞춘 값 기록 (2026-10-14)
    # ------------------------------------------------------------------
    def _on_teleop_cmd(self, msg):
        self._teleop_cmd = msg.data

    def _open_leader(self):
        if getattr(self, "_leader", None) is None:
            from gaze_hri.teleop import Leader
            try:
                self._leader = Leader(str(self.get_parameter("leader_port").value),
                                      os.path.expanduser(str(self.get_parameter("leader_map_file").value)))
            except Exception as exc:
                raise IKError(f"리더 팔을 열 수 없습니다 — {exc}") from exc
        return self._leader

    def _teleop_relative(self, label, goal_handle):
        """지금 자세에서 리더를 움직인 만큼만 팔로워를 움직인다. 그리퍼는 그대로 (병을 놓지 않게).
        GUI 에서 r 을 누르면 끝 (돌려줌: /teleop/cmd 문자열, 취소면 None)."""
        from gaze_hri.teleop import step_toward
        leader = self._open_leader()
        hz = float(self.get_parameter("stream_hz").value)
        step = math.radians(float(self.get_parameter("teleop_deg_s").value)) / hz
        base = list(self.q)
        l0 = leader.read()
        track = hasattr(self.backend, "track_step") and self.get_parameter("sag_correction").value
        cap = math.radians(float(self.get_parameter("pour_sag_max_deg").value))
        gain = float(self.get_parameter("pour_track_gain").value)
        self._teleop_cmd = None
        self.get_logger().info(f"[보정:{label}] 리더로 맞추고 GUI 에서 r — 리더를 움직인 만큼만 따라갑니다")
        t0, i = time.monotonic(), 0
        while True:
            if self._cancel or goal_handle.is_cancel_requested:
                return None
            cmd = self._teleop_cmd
            if cmd and cmd.startswith("record"):
                self._teleop_cmd = None
                return cmd
            i += 1
            lq = leader.read()
            target = [b + (l - l_0) for b, l, l_0 in zip(base[:5], lq[:5], l0[:5])] + [base[5]]
            self.q = step_toward(self.q, target, step)
            self.backend.write(self.q)
            if track:
                self.backend.track_step(gain, cap)
            time.sleep(max(0.0, t0 + i / hz - time.monotonic()))

    def _measured_q(self):
        """실제 서보 위치 (기구학 rad). dry-run 이면 명령값."""
        try:
            return list(self.backend.read())
        except Exception:
            return list(self.q)

    def _record_calib(self, rec):
        d = os.path.expanduser(str(self.get_parameter("pour_calib_dir").value))
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{self.get_parameter('arm_name').value or 'arm'}.jsonl")

        def conv(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            raise TypeError(type(v))
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=conv, ensure_ascii=False) + "\n")
        self.get_logger().info(f"[보정] 기록: {path} ({rec.get('stage')})")

    def _joint_record(self, planned_q):
        """관절 쪽 기록: 계획/실제 rad·deg·틱, 변화량, 부하, 처짐 보정값, 리더 틱."""
        meas = self._measured_q()
        rec = {"q_plan_rad": list(planned_q), "q_cmd_rad": list(self.q), "q_meas_rad": meas,
               "q_plan_deg": [math.degrees(v) for v in planned_q],
               "q_meas_deg": [math.degrees(v) for v in meas],
               "dq_deg": [math.degrees(m - p) for m, p in zip(meas, planned_q)]}
        if hasattr(self.backend, "map"):
            tp = self.backend.map.to_ticks(planned_q)
            tm = self.backend.map.to_ticks(meas)
            rec.update({"ticks_plan": tp, "ticks_meas": tm,
                        "dticks": [m - p for m, p in zip(tm, tp)],
                        "load": list(getattr(self.backend.bus, "last_load", None) or []),
                        "bias_deg": [math.degrees(b) for b in self.backend.bias]})
        ld = getattr(self, "_leader", None)
        if ld is not None and ld.last_ticks is not None:
            rec["leader_ticks"] = list(ld.last_ticks)
        return rec, meas

    def _pour_calib_model(self, cup_xy):
        """보정 기록 -> {start: xy, end: xy, th_start, th_end} (없는 건 빠짐).

        start/end 는 그 자세에서 사용자가 입구를 옮긴 양(카메라 컵 기준 오프셋)을 컵 위치에 대해
        맞춘 값 (3개 미만 평균, 이상 1차식). 예전 'high' 기록은 종료 자세로 친다."""
        if not self.get_parameter("pour_correction").value:
            return None
        path = os.path.join(os.path.expanduser(str(self.get_parameter("pour_calib_dir").value)),
                            f"{self.get_parameter('arm_name').value or 'arm'}.jsonl")
        if not os.path.exists(path):
            return None
        data = {"start": ([], [], []), "end": ([], [], [])}
        recs = []
        for line in open(path):
            try:
                recs.append(json.loads(line))
            except ValueError:
                continue
        if self.get_parameter("pour_calib_use").value == "latest" and recs:
            last = max((r.get("time", ""), r.get("session", "")) for r in recs)
            recs = [r for r in recs if (r.get("time", ""), r.get("session", "")) == last]
        for r in recs:
            st = {"high": "end"}.get(r.get("stage"), r.get("stage"))
            if st in data and "offset_cm" in r:
                data[st][0].append(np.array(r["cup_cam"][:2]))
                data[st][1].append(np.array(r["offset_cm"][:2]) / 100.0)
                if "theta_set_deg" in r:
                    data[st][2].append(float(r["theta_set_deg"]))

        def fit(X, Y):
            X, Y = np.array(X), np.array(Y)
            if len(X) < 3:
                return Y.mean(axis=0)
            A = np.hstack([X, np.ones((len(X), 1))])
            coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
            return np.array([cup_xy[0], cup_xy[1], 1.0]) @ coef
        out = {}
        for st, (X, Y, th) in data.items():
            if X:
                out[st] = fit(X, Y)
            if th:
                out[f"th_{st}"] = float(np.mean(th))
        return out or None

    def _stream_q_list(self, qs, g, duration):
        """관절각 목록(5개: 팔 4 + roll)을 순서대로 지나가며 스트리밍. 최소 저크 시간축."""
        # 지금 명령 자세에서 출발 (move_to 끝 해와 미세하게 달라도 튀지 않게)
        arr = np.asarray([list(self.q[:5])] + [list(q) for q in qs], dtype=float)
        n = len(arr) - 1
        vmax = math.radians(float(self.get_parameter("joint_speed_deg_s").value))
        travel = float(np.max(np.sum(np.abs(np.diff(arr, axis=0)), axis=0)))
        duration = max(duration, 1.875 * travel / vmax, 0.3)

        def q_of(s):
            x = s * n
            k = min(int(x), n - 1)
            return list(arr[k] + (arr[k + 1] - arr[k]) * (x - k)) + [g]
        track = hasattr(self.backend, "track_step") and self.get_parameter("sag_correction").value
        gain = float(self.get_parameter("pour_track_gain").value)
        cap = math.radians(float(self.get_parameter("pour_sag_max_deg").value))
        if not track:
            return self._stream(q_of, duration)
        return self._stream(q_of, duration,
                            after_write=lambda: self.backend.track_step(gain, cap))

    def _mouth_from_q(self, q):
        """관절각 -> 병 입구 위치 (모델). roll = roll_sign·s·θ 이므로 a = cosθ z + sin(roll_sign·roll) t."""
        from gaze_hri.pour import tangent
        G = forward_kinematics(q[:4], self.geo)
        L = float(self.get_parameter("pour_bottle_height").value) - \
            float(self.get_parameter("pour_grasp_height").value)
        r = q[4] * int(self.get_parameter("roll_sign").value)
        a = math.cos(r) * np.array([0.0, 0.0, 1.0]) + math.sin(r) * tangent(G)
        return np.asarray(G) + L * a, np.asarray(G)

    def _calib_table_stage(self, cup, session, base, goal_handle, fb):
        """책상 높이에서 그리퍼 가운데에 컵(바닥 쪽 몸통)이 오게 맞춘다 = 진짜 컵 위치.
        수평으로 안 닿는 거리면 조금씩 숙여 본다 (잡지 않으니 수평일 필요는 없다)."""
        g_open = float(self.get_parameter("gripper_open").value)
        z = float(self.get_parameter("pour_grasp_height").value)
        tgt = np.array([cup[0], cup[1], z])
        u = np.array([cup[0], cup[1], 0.0]) / max(float(np.linalg.norm(cup[:2])), 1e-6)
        d = float(self.get_parameter("side_standoff").value)
        z_hi = float(self.get_parameter("side_high_z").value)
        for deg in (0.0, -10.0, -20.0, -30.0):
            p = math.radians(deg)
            pts = [tgt - u * d + np.array([0, 0, z_hi - z]), tgt - u * d, tgt]
            try:
                for q in pts:
                    solve_with_fallback(list(q), self.geo, pitch_candidates=[p])
            except IKError:
                continue
            fb("CALIB_TABLE", 0.1)
            self.move_to(pts[0], p, g_open, strict=True)
            self.move_to(pts[1], p, g_open, strict=True)
            self.move_to(pts[2], p, g_open, strict=True)
            planned = list(self.q)
            cmd = self._teleop_relative("책상: 그리퍼 가운데에 컵", goal_handle)
            if cmd is None:
                return False
            rec, meas = self._joint_record(planned)
            rec.update(base, stage="table", pitch_deg=deg, tcp_plan=tgt,
                       tcp_meas=np.asarray(forward_kinematics(meas[:4], self.geo)),
                       snapshot=cmd.partition(":")[2] or None)
            rec["tcp_err_cm"] = ((rec["tcp_meas"] - tgt) * 100).tolist()
            self._record_calib(rec)
            # 컵을 쓸지 않게 반경 방향으로 뒤로 뺐다가 위로
            here = np.asarray(forward_kinematics(self.q[:4], self.geo))
            self.move_to(here - u * d, p, g_open, strict=False)
            self.move_to(here - u * d + np.array([0, 0, z_hi - here[2]]), p, g_open, strict=False)
            return True
        self.get_logger().warn("[보정] 책상 단계: 이 컵 위치는 어느 각도로도 안 닿아 건너뜁니다")
        return True

    def _nudge_stage(self, cup, plan, goal_handle, min_theta=None):
        """입구 위치를 화면 방향으로 1cm 씩 옮긴다 (GUI 8/4/2/6·화살표 -> /teleop/cmd "nudge:dx:dy",
        dx/dy 는 화면 픽셀 방향 부호). r -> (누적 오프셋 xy [m], 명령 문자열). 취소 -> None."""
        from gaze_hri.pour import grip_for_mouth
        from gaze_hri.table_view import TableHomography
        hom = TableHomography()
        ok, why = hom.load(os.path.expanduser(str(self.get_parameter("homography_file").value)))
        if not ok:
            raise IKError(f"입구 조정에 쓸 탑다운 호모그래피가 없습니다 ({why})")
        step = float(self.get_parameter("nudge_step").value)
        sign_s = int(self.get_parameter("roll_sign").value) * plan["s"]
        theta = abs(self.q[4])
        L = plan["cfg"].bottle_height - plan["cfg"].grasp_height
        mouth0, _ = self._mouth_from_q(list(self.q))
        offset = np.zeros(2)
        g = self.q[5]
        self._calib_theta = theta
        self._teleop_cmd = None
        t_hot = 0.0
        self.get_logger().info("[보정] 멈춤 — GUI 8/4/2/6(화살표) 입구 1cm, +/- 기울기 5°, 맞으면 r")
        while True:
            if self._cancel or goal_handle.is_cancel_requested:
                return None
            cmd, self._teleop_cmd = self._teleop_cmd, None
            if time.monotonic() - t_hot > 2.0:
                # 2026-10-14: 기울인 채 오래 멈추면 팔꿈치가 60°C 로 안전 정지했다 -> 55°C 부터 경고
                t_hot = time.monotonic()
                try:
                    tmax = max([t for t in self.backend.bus.temps() if 0 < t <= 70] or [0]) if hasattr(self.backend, "bus") else 0   # 60 넘으면 드라이버가 안전 정지 -> 70 넘는 값은 통신 오류 (97, 150 이 떴다)
                except Exception:
                    tmax = 0
                if tmax >= 55:
                    self.pub_phase.publish(String(data=f"HOT {tmax}C — 빨리 r"))
                    self.get_logger().warn(f"[보정] 서보 {tmax}°C — 60°C 면 안전 정지. 빨리 맞추고 r")
            if not cmd:
                time.sleep(0.05)
                continue
            if cmd.startswith("record"):
                self.get_logger().info(f"[보정] 오프셋 {np.round(offset * 100, 1).tolist()} cm, "
                                       f"기울기 {math.degrees(theta):.0f}° 로 확정")
                self._calib_theta = theta
                return offset, cmd
            if cmd.startswith("tilt:"):
                # 2026-10-14: +/- 로 기울기 조절 — 입구는 그 자리, 그 점을 축으로 돈다
                # 종료 자세 조정에서는 확정한 시작 각도 아래로 안 세운다 (사용자: + 로 따라 보다가
                # - 로 되돌릴 때 시작 각도에서 멈추게)
                lo = math.radians(plan["cfg"].pre_deg) if min_theta is None else min_theta
                if theta + math.radians(float(self.get_parameter("tilt_step_deg").value)) \
                        * float(cmd.split(":")[1]) < lo - 1e-6:
                    self.get_logger().info(f"[보정] {math.degrees(lo):.0f}° (시작 각도) 아래로는 안 세웁니다")
                hi = min(math.radians(150.0), abs(self.geo.limits["wrist_roll"][1]) - 0.05)
                th = min(hi, max(lo, theta + math.radians(
                    float(self.get_parameter("tilt_step_deg").value)) * float(cmd.split(":")[1])))
                m = mouth0 + np.array([offset[0], offset[1], 0.0])
                # 병 바닥이 컵 테두리 아래로 내려가는 각도는 안 된다 (컵을 친다)
                bottom_z = m[2] - plan["cfg"].bottle_height * math.cos(th)
                if bottom_z < plan["cfg"].cup_height + 0.005:
                    self.get_logger().warn(f"[보정] {math.degrees(th):.0f}° 면 병 바닥이 컵 높이 아래로 "
                                           f"({bottom_z * 100:.1f}cm) — 더 세울 수 없습니다")
                    continue
                try:
                    q = self._ik_continuous(grip_for_mouth(m, th, plan["s"], L), 0.0, list(self.q))
                except IKError:
                    self.get_logger().warn("[보정] 그 각도로는 못 갑니다 (IK)")
                    continue
                theta = th
                self._stream_q_list([list(q[:4]) + [sign_s * theta]], g, 0.5)
                self.get_logger().info(f"[보정] 기울기 {math.degrees(theta):.0f}°")
                continue
            if not cmd.startswith("nudge:"):
                continue
            _, dx, dy = cmd.split(":")
            px = hom.to_pixel(cup[0], cup[1])
            p2 = hom.to_robot(px[0] + 20.0 * float(dx), px[1] + 20.0 * float(dy)) if px else None
            if p2 is None:
                continue
            d = np.array(p2) - np.asarray(cup[:2], float)
            d = d / max(float(np.linalg.norm(d)), 1e-9) * step
            new = offset + d
            m = mouth0 + np.array([new[0], new[1], 0.0])
            G = grip_for_mouth(m, theta, plan["s"], L)
            try:
                q = self._ik_continuous(G, 0.0, list(self.q))
            except IKError:
                self.get_logger().warn("[보정] 그쪽으로는 더 못 갑니다 (IK)")
                continue
            offset = new
            self._stream_q_list([list(q[:4]) + [sign_s * theta]], g, 0.4)
            self.get_logger().info(f"[보정] 입구 오프셋 {np.round(offset * 100, 1).tolist()} cm")

    def _execute_pour(self, bottle, cup, fb, goal_handle, result, calib=False, wait=False):
        g_open = float(self.get_parameter("gripper_open").value)
        # 집기 보정(xy)·높이 보정은 execute 에서 이미 병 위치에 더했다
        # 2026-10-14: 보정 때도 지금까지의 보정을 적용해서 시작한다 (적용 안 하면 먼 컵이 IK 로 거부됐다).
        # 기록은 '카메라 컵 기준 누적 오프셋' 이라 적용한 보정 + 이번에 옮긴 양으로 남긴다.
        raw_cup = np.asarray(cup, float).copy()
        plan, cup = self._pour_plan_pulled(bottle, cup)   # 움직이기 전에 전부 검사 (도달 끝이면 조금 당김)
        pull_v = np.asarray(cup[:2], float) - raw_cup[:2]   # 당긴 양 — 보정 기록(카메라 컵 기준)에 더한다
        segs, cfg = plan["qsegs"], plan["cfg"]

        def path(name):
            return [q for q, _ in segs[name]]

        def dur(name):
            return sum(dt for _, dt in segs[name])
        self.get_logger().info(
            f"따르기 계획: 기울기 {'오른쪽' if plan['s'] < 0 else '왼쪽'}, 입구 "
            f"{np.round(plan['mouth'], 3).tolist()}, 옆에서 {cfg.pre_deg:.0f}° -> 컵 위로 "
            f"{dur('move'):.1f}s -> {cfg.onset_deg:.0f}° -> 천천히 {cfg.max_deg:.0f}° {dur('pour'):.1f}s")
        _, _, _, above = self._side_points(bottle, "병")
        up_here = np.array([bottle[0], bottle[1], plan["carry"][2]])
        # 2026-10-26: 멀리 놓인 병(받침대 42cm+)은 그 자리에서 23cm 로 들어 올리는 자세가 IK 불가라
        # 집었다가 그냥 내려놨다 -> 들어 올릴 점만 받침대 쪽으로 1cm 씩(최대 10cm) 당겨 풀리는 곳으로
        r_b = float(np.linalg.norm(bottle[:2]))
        for d_in in np.arange(0.0, 0.1001, 0.01):
            c = np.array([bottle[0] * (r_b - d_in) / r_b, bottle[1] * (r_b - d_in) / r_b, up_here[2]])
            try:
                solve_with_fallback(c, self.geo, pitch_candidates=[0.0])
            except IKError:
                continue
            if d_in > 0:
                self.get_logger().info(f"  병을 들어 올릴 점을 받침대 쪽으로 {d_in * 100:.0f}cm 당김 (그 자리 위는 IK 불가)")
            up_here = c
            break

        def cancelled():
            return self._cancel or goal_handle.is_cancel_requested

        base = {}
        if calib:
            base = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "session": time.strftime("%m%d_%H%M%S"),
                    "arm": self.get_parameter("arm_name").value,
                    "cup_cam": raw_cup, "bottle": np.asarray(bottle, float),
                    "s": plan["s"], "roll_sign": int(self.get_parameter("roll_sign").value),
                    "bottle_height": cfg.bottle_height, "cup_height": cfg.cup_height,
                    "lip_clear": cfg.lip_clear}

        if not self._side_pick(bottle, fb, max_tilt_deg=POUR_MAX_GRASP_TILT):
            if cancelled():
                self._end_cancel(goal_handle)
                result.success, result.message = False, "사용자 취소"
                return result
            self._side_retreat_home(bottle, g_open)
            goal_handle.abort()
            result.success, result.message = False, "병 파지 실패"
            return result
        g_hold = self.q[5]
        fb("LIFT", 0.4)
        self.move_to(above, 0.0, g_hold, strict=True)
        self.move_to(up_here, 0.0, g_hold, strict=True)
        if not self.close_and_verify(reclose=False):
            goal_handle.abort()
            result.success, result.message = False, "들어 올리다 병을 놓쳤습니다."
            self.go_end()
            return result
        if wait:
            got = self._wait_gate(goal_handle, fb)
            if got == "cancel":
                self._end_cancel(goal_handle)
                result.success, result.message = False, "사용자 취소 — 병을 든 채 정지했습니다."
                return result
            if got != "go":
                self.get_logger().warn(f"따르기 대기 끝 ({got}) — 병을 제자리에 둡니다.")
                self._put_bottle_back(bottle, up_here, above, g_hold, g_open, fb)
                goal_handle.abort()
                result.success, result.message = False, f"따르기 안 함 ({got}) — 병을 제자리에 두었습니다"
                return result
            if self._gate_cup is not None:
                go = self._grasp_offset("cup")
                new = np.array([self._gate_cup[0], self._gate_cup[1], cup[2]])
                if go is not None:
                    new[:2] += go[:2]
                if self.get_parameter("grasp_mode").value == "side":
                    new = self._reach_pull(new)        # 처음 계획과 같은 뻗음 보정
                self.get_logger().info(
                    f"  컵 다시 검출: {np.round(new[:2], 3).tolist()} (계획 {np.round(cup[:2], 3).tolist()}, "
                    f"{np.linalg.norm(new[:2] - np.asarray(cup[:2])) * 100:.1f}cm 차이) — 다시 계획")
                try:
                    raw_new = new.copy()
                    plan, new = self._pour_plan_pulled(bottle, new)
                    pull_v = np.asarray(new[:2], float) - raw_new[:2]
                    segs, cfg, cup = plan["qsegs"], plan["cfg"], new
                    if calib:
                        # 기록 기준은 실제 검출 컵 — 당긴 양은 보정 오프셋에 포함돼야 다음에 맞는다
                        base["cup_cam"] = raw_new
                except IKError as exc:
                    self.get_logger().warn(f"놓인 컵 위치로 따르기 계획 불가 ({exc}) — 병을 제자리에 둡니다.")
                    self._put_bottle_back(bottle, up_here, above, g_hold, g_open, fb)
                    goal_handle.abort()
                    result.success, result.message = False, (
                        "놓인 컵이 따르기 범위 밖입니다 — 병을 제자리에 두었습니다")
                    return result
        fb("CARRY", 0.5)
        self.move_to(plan["carry"], 0.0, g_hold, strict=True)
        self.move_to(plan["start"], 0.0, g_hold, strict=True)
        fb("TILT", 0.6)
        self._stream_q_list(path("pre"), g_hold, dur("pre"))        # 컵 옆: 물 안 나오는 각도까지
        self._stream_q_list(path("move"), g_hold, dur("move"))      # 그 각도로 컵 위로
        self._settle()                                              # 처짐 맞추고 나서 따른다
        fb("POUR", 0.7)
        self._stream_q_list(path("onset"), g_hold, dur("onset"))
        if calib:
            # 2026-10-14: 시작 자세와 종료 자세에서 각각 멈춰 입구 위치(8/4/2/6)와 각도(+/-)를
            # 맞춘다 -> r. 기울기에 따라 처짐이 달라 한 자세만 맞추면 다른 자세가 틀어진다(사용자).
            cup_xy0 = np.asarray(cup[:2], float)

            def stage(name, label, min_theta=None):
                fb("ADJUST_" + name.upper(), 0.72 if name == "start" else 0.78)
                planned = list(self.q)
                mouth_p, grip_p = self._mouth_from_q(planned)
                self.get_logger().info(f"[보정] {label} 자세 — 입구 위치/각도를 맞추고 r")
                got = self._nudge_stage(cup, plan, goal_handle, min_theta=min_theta)
                if got is None:
                    return None
                off, cmd = got
                rec, meas = self._joint_record(planned)
                mouth_m, grip_m = self._mouth_from_q(meas)
                rec.update(base, stage=name, theta_deg=math.degrees(abs(planned[4])),
                           theta_set_deg=math.degrees(self._calib_theta), mouth_plan=mouth_p,
                           grip_plan=grip_p, mouth_meas=mouth_m, grip_meas=grip_m,
                           snapshot=cmd.partition(":")[2] or None)
                return rec, off

            got = stage("start", "시작(물이 나오기 시작할 때)")
            if got is None:
                self._end_cancel(goal_handle)
                result.success, result.message = False, "보정 취소 — 그 자리에서 정지했습니다."
                return result
            rec_s, off_s = got
            c_s, c_e = plan["corr_start"], plan["corr_end"]     # 이번에 이미 적용한 보정
            rec_s["offset_cm"] = (c_s + off_s + pull_v) * 100
            rec_s["applied_corr_cm"] = c_s * 100
            self._record_calib(rec_s)
            th_s = self._calib_theta
            th_e_plan = max(math.radians(cfg.max_deg), th_s + math.radians(10))
            angles = (math.degrees(th_s), math.degrees(th_e_plan))
            try:
                plan = self._pour_plan(bottle, cup_xy0 + c_s + off_s, raw=True, angles=angles,
                                       end_shift_override=c_e - c_s)
                segs, cfg = plan["qsegs"], plan["cfg"]
            except IKError as exc:
                self.get_logger().warn(f"[보정] 맞춘 시작 위치로 다시 계획 못 함 ({exc}) — 원래 계획으로")
            self._stream_q_list(path("pour"), g_hold, dur("pour"))     # 종료 자세 쪽으로 천천히
            got = stage("end", "종료(다 따랐을 때)", min_theta=th_s)
            if got is None:
                self._end_cancel(goal_handle)
                result.success, result.message = False, "보정 취소 — 그 자리에서 정지했습니다."
                return result
            rec_e, off_e = got
            rec_e["offset_cm"] = (c_e + off_s + off_e + pull_v) * 100      # 카메라 컵 기준 누적 (+ 도달 끝 당김)
            rec_e["applied_corr_cm"] = c_e * 100
            self._record_calib(rec_e)
            try:   # 되돌아갈 경로도 맞춘 두 자세 기준으로
                plan = self._pour_plan(bottle, cup_xy0 + c_s + off_s, raw=True,
                                       end_shift_override=(c_e - c_s) + off_e,
                                       angles=(math.degrees(th_s), math.degrees(self._calib_theta)))
                segs, cfg = plan["qsegs"], plan["cfg"]
            except IKError as exc:
                self.get_logger().warn(f"[보정] 되돌아갈 경로 재계획 실패 ({exc}) — 이전 경로로")
        else:
            self._stream_q_list(path("pour"), g_hold, dur("pour"))
        time.sleep(cfg.hold_s)
        back = (path("pour")[::-1] + path("onset")[::-1])[1:] + [path("move")[-1]]
        fb("UPRIGHT", 0.8)
        # 컵 위에서 먼저 pre 까지 세운다 (빠르게 — 흐름을 끊는다). 그 다음에야 옆으로 뺀다.
        self._stream_q_list(back, g_hold,
                            math.radians(cfg.max_deg - cfg.pre_deg) / math.radians(cfg.fast_deg_s))
        self._stream_q_list(path("move")[::-1][1:] + [path("pre")[-1]], g_hold, dur("move"))
        self._stream_q_list(path("pre")[::-1][1:] + [plan["start_q"]], g_hold, dur("pre"))
        if cancelled():
            self._end_cancel(goal_handle)
            result.success, result.message = False, "사용자 취소 — 그 자리에서 정지했습니다."
            return result
        fb("RETURN", 0.85)
        self.move_to(plan["carry"], 0.0, g_hold, strict=True)
        self._put_bottle_back(bottle, up_here, above, g_hold, g_open, fb)
        if cancelled():
            self._end_cancel(goal_handle)
            result.success, result.message = False, "사용자 취소"
            return result
        goal_handle.succeed()
        result.success, result.message = True, ("따르기 보정 기록 완료" if calib else
                                                "따르기 시늉 완료 — 병을 제자리에 두었습니다")
        return result

    def _wait_gate(self, goal_handle, fb):
        """병을 든 채 pour_gate 를 기다린다. "go" | "abort" | "timeout" | "cancel"."""
        fb("WAIT", 0.45)
        timeout = float(self.get_parameter("pour_wait_timeout").value)
        t0, warned = time.monotonic(), False
        self.get_logger().info(f"  병을 들고 대기 — 컵 팔이 컵을 놓고 빠지면 들어갑니다 (최대 {timeout:.0f}s)")
        while True:
            if self._cancel or goal_handle.is_cancel_requested:
                return "cancel"
            gate = self._gate
            if gate:
                parts = gate.split(":")
                self._gate_cup = (float(parts[1]), float(parts[2])) if len(parts) == 3 else None
                return "go" if parts[0] == "go" else parts[0]
            if time.monotonic() - t0 > timeout:
                return "timeout"
            tmax = max([t for t in self.backend.bus.temps() if 0 < t <= 70] or [0]) if hasattr(self.backend, "bus") else 0   # 60 넘으면 드라이버가 안전 정지 -> 70 넘는 값은 통신 오류 (97, 150 이 떴다)
            if tmax >= 55 and not warned:   # 가득 찬 병을 들고 오래 기다리면 어깨가 데워진다
                warned = True
                self.get_logger().warn(f"  대기 중 서보 {tmax}°C")
                self.pub_phase.publish(String(data=f"HOT {tmax}C"))
            time.sleep(0.1)

    def _put_bottle_back(self, bottle, up_here, above, g_hold, g_open, fb):
        """병 위(들어 올린 높이)에서 원래 자리에 내려놓고 빠져 휴식자세로."""
        self.move_to(up_here, 0.0, g_hold, strict=True)
        bp = self._side_pitch(bottle, "병", max_tilt_deg=POUR_MAX_GRASP_TILT)
        self.move_to(above, bp, g_hold, strict=True)
        self.move_to(bottle, bp, g_hold, strict=True)
        fb("RELEASE", 0.9)
        self.set_gripper(g_open)
        _, pre_low, pre_high, b_above = self._side_points(bottle, "병")
        if self._is_vertical(bottle):
            self.move_to(b_above, 0.0, g_open, strict=True)
        else:
            self.move_to(pre_low, bp, g_open, strict=True)
            self._via_mid(pre_low, bp, g_open)
            self.move_to(pre_high, bp, g_open, strict=True)
        self.go_end()

    @staticmethod
    def _end_cancel(goal_handle):
        """/task/cancel(토픽)로 멈춘 경우는 액션 취소 요청이 없어 canceled() 가 예외를 낸다
        (2026-10-14 로그: invalid transition EXECUTING -> CANCELED). 그때는 abort."""
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()

    def _grasp_file(self):
        return os.path.join(os.path.expanduser(str(self.get_parameter("pour_calib_dir").value)),
                            f"{self.get_parameter('arm_name').value or 'arm'}_grasp.jsonl")

    def _grasp_offset(self, label=None):
        """최신 집기 보정 (xy, m). 없으면 None.
        2026-10-26: 물체 종류별 — 컵은 잘 잡는데 물병만 한쪽으로 치우쳤다(투명 병이라 검출 중심이 밀림).
        label(없으면 지금 집는 물체)과 같은 라벨 기록만 쓴다. 라벨 없는 옛 기록은 안 쓴다."""
        if not self.get_parameter("grasp_correction").value or not os.path.exists(self._grasp_file()):
            return None
        label = self._pick_label if label is None else label
        last = None
        for line in open(self._grasp_file()):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("label") == label:
                last = r
        return None if last is None else np.array(last["offset_cm"][:2]) / 100.0

    def _grasp_calib_stage(self, target, pitch):
        """내려가기 전 멈춤 (컵 뒤 높은 곳): GUI 8/4/2/6 로 그리퍼를 1cm 씩 (화면 방향) -> r.
        보정 기록 후 이번에 옮긴 양(xy, m)을 돌려준다. 취소면 None."""
        from gaze_hri.table_view import TableHomography
        hom = TableHomography()
        ok, why = hom.load(os.path.expanduser(str(self.get_parameter("homography_file").value)))
        if not ok:
            self.get_logger().warn(f"[집기 보정] 호모그래피 없음 ({why}) — 건너뜀")
            return np.zeros(2)
        step = float(self.get_parameter("nudge_step").value)
        g = self.q[5]
        offset = np.zeros(2)
        self._teleop_cmd = None
        self.pub_phase.publish(String(data="ADJUST_GRASP"))
        self.get_logger().info("[집기 보정] 내려가기 전 멈춤 — 8/4/2/6(화살표) 로 그리퍼 가운데 선이 컵 중심을 "
                               "향하고 손가락 끝이 입구 뒤 ~2cm 가 되게 (1cm 씩), 맞으면 r")
        base = np.asarray(target, float)
        while True:
            if self._cancel:
                return None
            cmd, self._teleop_cmd = self._teleop_cmd, None
            if not cmd:
                time.sleep(0.05)
                continue
            if cmd.startswith("record"):
                break
            if not cmd.startswith("nudge:"):
                continue
            _, dx, dy = cmd.split(":")
            px = hom.to_pixel(base[0], base[1])
            p2 = hom.to_robot(px[0] + 20.0 * float(dx), px[1] + 20.0 * float(dy)) if px else None
            if p2 is None:
                continue
            d = np.array(p2) - base[:2]
            new = offset + d / max(float(np.linalg.norm(d)), 1e-9) * step
            try:
                self.move_to(base + np.array([new[0], new[1], 0.0]), pitch, g, strict=True)
            except IKError:
                self.get_logger().warn("[집기 보정] 그쪽으로는 더 못 갑니다 (IK)")
                continue
            offset = new
            self.get_logger().info(f"[집기 보정] 오프셋 {np.round(offset * 100, 1).tolist()} cm")
        prev = self._grasp_offset()
        total = offset + (prev if prev is not None else 0.0)
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": self.get_parameter("arm_name").value,
               "label": self._pick_label,
               "pick_cam": np.asarray(self._grasp_cam_pick, float), "offset_this_cm": offset * 100,
               "offset_cm": total * 100, "snapshot": cmd.partition(":")[2] or None}
        os.makedirs(os.path.dirname(self._grasp_file()), exist_ok=True)
        with open(self._grasp_file(), "a") as f:
            f.write(json.dumps(rec, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v),
                               ensure_ascii=False) + "\n")
        self.get_logger().info(f"[집기 보정] 기록: 이번 {np.round(offset * 100, 1).tolist()} cm, "
                               f"누적 {np.round(total * 100, 1).tolist()} cm -> {self._grasp_file()}")
        return offset

    def _side_retreat_home(self, p, g_open):
        """실패 후 복귀: 컵을 쓸지 않게 뒤로 빠졌다가 위로 올라간 다음 home."""
        try:
            _, pre_low, pre_high, above = self._side_points(p, "복귀")
            rp = self._side_pitch(p, "복귀")
            if self._is_vertical(p):
                self.move_to(above, 0.0, g_open, strict=True)
            else:
                self.move_to(pre_low, rp, g_open, strict=True)
                self._via_mid(pre_low, rp, g_open)
                self.move_to(pre_high, rp, g_open, strict=True)
        except IKError as exc:
            self.get_logger().warn(f"뒤로 빠지기 실패({exc}) — 바로 home")
        self.go_end()

    def _choose_task_pitch(self, pitch, pts, label=""):
        cand = pitch
        while cand <= math.radians(-30.0) + 1e-9:
            try:
                for p in pts:
                    solve_with_fallback(list(p), self.geo, pitch_candidates=[cand])
                if abs(cand - pitch) > 1e-6:
                    self.get_logger().info(
                        f"  {label} 접근각: {math.degrees(pitch):.0f}° 로는 안 닿아 "
                        f"{math.degrees(cand):.0f}° 로 그 구간 통일")
                return cand
            except IKError:
                cand += math.radians(5.0)
        return pitch          # 어느 각도도 전 구간이 안 되면 기존 방식(구간별 대체)으로

    def _safe_go_home(self):
        """복귀 시도. 복귀 자체가 실패해도 원래 오류를 덮지 않는다."""
        try:
            self.go_end()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"홈 복귀 실패: {exc}")


def main():
    rclpy.init()
    node = ArmServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.backend.close()
        node.destroy_node()
        if rclpy.ok():                # SIGINT/SIGTERM 이면 rclpy 가 이미 내렸다
            rclpy.shutdown()


if __name__ == "__main__":
    main()
