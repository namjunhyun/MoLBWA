#!/usr/bin/env python3
"""
[노드 4] task_manager  —  상태 기계 (시나리오의 뇌)

"컵을 보면 -> 그 컵이 선택되고 -> 옮길 자리를 보면 -> 로봇이 옮긴다"
이 흐름을 관리합니다.

상태 전이:

    IDLE ──(pick 목표 수신)──> PICK_SELECTED
      ▲                              │
      │                       (place 목표 수신)
      │                              ▼
      └──(동작 완료/실패/취소)──  EXECUTING

안전장치 (대회 데모에서 실제로 중요합니다):
  1. place 목표는 pick 목표와 min_separation(기본 10cm) 이상 떨어져야 함.
     -> 같은 곳을 계속 보다가 pick/place가 동시에 잡히는 사고 방지.
  2. PICK_SELECTED 상태가 select_timeout(기본 15초) 지나면 자동 IDLE 복귀.
     -> 잘못 선택했을 때 그냥 딴 데 보고 기다리면 취소됨.
  3. 신뢰도(confidence)가 낮은 목표는 거부.
  4. /task/cancel 로 언제든 중단 (실제로는 물리 비상정지 버튼도 반드시 두세요).

발행:
  /task/expected_role  std_msgs/String  -> target_resolver가 pick/place 해석을 바꿈
  /task/state          std_msgs/String  -> UI/RViz 표시용
액션 클라이언트:
  /pick_place          gaze_hri_msgs/action/PickPlace
"""

import math

import numpy as np
import rclpy
from gaze_hri_msgs.action import PickPlace
from gaze_hri_msgs.msg import GazeTarget
from geometry_msgs.msg import Point
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Empty, String

IDLE = "IDLE"
PICK_SELECTED = "PICK_SELECTED"
EXECUTING = "EXECUTING"


class TaskManager(Node):

    def __init__(self):
        super().__init__("task_manager")

        self.declare_parameter("min_separation", 0.10)
        self.declare_parameter("select_timeout", 15.0)
        self.declare_parameter("min_confidence", 0.35)
        self.declare_parameter("approach_pitch", -math.pi / 4)
        self.declare_parameter("dry_run", False)

        self.min_separation = float(self.get_parameter("min_separation").value)
        self.select_timeout = float(self.get_parameter("select_timeout").value)
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.approach_pitch = float(self.get_parameter("approach_pitch").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)

        self.state = IDLE
        self.pick_point = None
        self.state_entered = self.now()
        self._goal_handle = None
        # target_resolver 가 목표를 내보낸 좌표계. 액션 목표에 그대로 실어 보낸다.
        self.target_frame = "base_link"

        self.create_subscription(GazeTarget, "/gaze/target", self.on_target, 10)
        self.create_subscription(Empty, "/task/cancel", self.on_cancel, 10)

        self.pub_role = self.create_publisher(String, "/task/expected_role", 10)
        self.pub_state = self.create_publisher(String, "/task/state", 10)

        self.client = ActionClient(self, PickPlace, "/pick_place")

        self.create_timer(0.2, self.tick)
        self.set_state(IDLE)
        self.get_logger().info("task_manager 시작 — 컵을 응시하세요.")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------
    def set_state(self, state):
        self.state = state
        self.state_entered = self.now()
        self.pub_state.publish(String(data=state))

        role = {IDLE: "pick", PICK_SELECTED: "place", EXECUTING: "idle"}[state]
        self.pub_role.publish(String(data=role))

    def tick(self):
        # 주기적으로 현재 역할을 다시 알림 (노드가 늦게 떠도 동기화되도록)
        self.pub_state.publish(String(data=self.state))
        role = {IDLE: "pick", PICK_SELECTED: "place", EXECUTING: "idle"}[self.state]
        self.pub_role.publish(String(data=role))

        # 선택 타임아웃
        if self.state == PICK_SELECTED and \
                (self.now() - self.state_entered) > self.select_timeout:
            self.get_logger().warn("선택 타임아웃 — 초기 상태로 돌아갑니다.")
            self.pick_point = None
            self.set_state(IDLE)

    # ------------------------------------------------------------------
    def on_cancel(self, _msg):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
        self.pick_point = None
        self.set_state(IDLE)
        self.get_logger().warn("사용자 취소")

    def on_target(self, msg: GazeTarget):
        p = np.array([msg.point.x, msg.point.y, msg.point.z])

        # 안전장치 3: 신뢰도가 낮은 목표는 거부한다.
        # (dwell 산포가 크거나 깜빡임이 섞인 응시는 여기서 걸러진다)
        if msg.confidence < self.min_confidence:
            self.get_logger().warn(
                f"신뢰도가 낮아 목표를 무시합니다 "
                f"({msg.confidence:.2f} < {self.min_confidence:.2f}). "
                "조금 더 가만히 응시해 주세요."
            )
            return

        # 목표가 어느 좌표계인지 기억해 두었다가 액션 목표에 같이 실어 보낸다.
        # 이게 없으면 arm_server 가 map 기준으로 넘겨받았다고 착각해 변환을 한 번 더 건다.
        self.target_frame = msg.header.frame_id

        if self.state == IDLE and msg.role == "pick":
            self.pick_point = p
            self.set_state(PICK_SELECTED)
            self.get_logger().info(
                f"물체 선택됨: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}). "
                "이제 옮길 자리를 응시하세요."
            )

        elif self.state == PICK_SELECTED and msg.role == "place":
            sep = float(np.linalg.norm(p - self.pick_point))
            if sep < self.min_separation:
                self.get_logger().warn(
                    f"놓을 자리가 집을 자리와 너무 가깝습니다 ({sep * 100:.1f}cm). "
                    "조금 더 떨어진 곳을 보세요."
                )
                return
            self.send_goal(self.pick_point, p)

    # ------------------------------------------------------------------
    def send_goal(self, pick, place):
        if not self.client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("/pick_place 액션 서버가 없습니다. arm_server를 켜세요.")
            self.pick_point = None
            self.set_state(IDLE)
            return

        goal = PickPlace.Goal()
        goal.pick = Point(x=float(pick[0]), y=float(pick[1]), z=float(pick[2]))
        goal.place = Point(x=float(place[0]), y=float(place[1]), z=float(place[2]))
        goal.frame_id = self.target_frame
        goal.approach_pitch = self.approach_pitch
        goal.dry_run = self.dry_run

        self.set_state(EXECUTING)
        self.get_logger().info("로봇팔에 목표 전달 — 동작 시작")

        fut = self.client.send_goal_async(goal, feedback_callback=self.on_feedback)
        fut.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("액션 목표가 거부되었습니다.")
            self.pick_point = None
            self.set_state(IDLE)
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self.on_result)

    def on_feedback(self, fb):
        f = fb.feedback
        self.get_logger().info(f"  진행: {f.phase} ({f.progress * 100:.0f}%)")

    def on_result(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info(f"완료: {result.message}")
        else:
            self.get_logger().error(f"실패: {result.message}")
        self._goal_handle = None
        self.pick_point = None
        self.set_state(IDLE)


def main():
    rclpy.init()
    node = TaskManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
