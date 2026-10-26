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

두 팔 (2026-10-14, dual.launch.py):
  arms 에 팔이 2대 이상이면, 집을 곳과 놓을 곳 **둘 다** 닿는 팔이 움직인다 (받침대 중심 기준
  수평 거리가 pick_min_r~pick_max_r / place_min_r~place_max_r 안). 둘 다 닿는 팔이 여럿이면
  집을 컵에 더 가까운 팔. 놓을 곳에 닿는 팔이 없으면 거리를 억지로 당겨 엉뚱한 데 두지 않고,
  집기 선택은 그대로 둔 채 다른 놓을 곳을 기다린다 (2026-10-14, 팔 B 가 66cm 밖 놓을 곳을
  42cm 로 당겨 놓았다).

따르기 (2026-10-14): **물병**을 집을 것으로 고른 뒤 놓을 곳으로 **컵**을 고르면, 병을 들어
그 컵에 따르는 시늉을 하고 병을 제자리에 둔다 (arm_server task="pour", gaze_hri/pour.py).
빈 곳을 고르면 그냥 옮긴다. 다른 물체 위(컵->컵, 컵->병 등)는 부딪히므로 거부한다.
클래스는 YOLO 이름 (/objects/labels, /objects/poses 와 같은 순서).
컵은 병을 쥔 팔에서 pour_min_r~pour_max_r 안이어야 한다. 좌표는 target_resolver 가 첫 팔(arms[0]) 기준으로 낸 것을
  탑다운 호모그래피로 픽셀 -> 고른 팔 기준으로 옮긴다 (팔마다 같은 카메라로 따로 캘리브했다).
  놓을 자리 거리 보정(place_min_r~place_max_r)도 고른 팔 기준으로 여기서 한다.
"""

import math
import time

import json
import os

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from gaze_hri_msgs.action import PickPlace
from gaze_hri_msgs.msg import GazeTarget
from geometry_msgs.msg import Point, PoseArray
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String

from gaze_hri.table_view import TableHomography

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
        # 두 팔: 이름 / 호모그래피 / 액션. 비어 있으면 예전처럼 한 팔(/pick_place).
        self.declare_parameter("arms", [""])
        self.declare_parameter("arm_homographies", [""])
        self.declare_parameter("arm_actions", [""])
        self.declare_parameter("pick_min_r", 0.37)
        self.declare_parameter("pick_max_r", 0.42)
        self.declare_parameter("place_min_r", 0.37)
        self.declare_parameter("place_max_r", 0.42)
        # 2026-10-14(4): 팔 고르기는 거리로 대충 걸러 **순서만** 정하고, 되는지는 arm_server 가 실제
        # IK 로 판단한다. 움직이기 전에 거부되면([거부]) 다음 팔에게 넘긴다. (예전엔 여기 거리 띠가
        # 실제보다 좁아 "선택 안 되는 곳이 너무 많다" — 사용자)
        self.declare_parameter("hard_min_r", 0.25)
        self.declare_parameter("hard_max_r", 0.44)        # 수평 집기 IK 한계 (= arm_server workspace_radius)
        # 2026-10-26: 팔별 뻗음 보정 (arm_server reach_offset 과 같은 값, arms 순서). 수평 집기 범위는 모델 거리라
        # 실제(카메라) 거리로는 이만큼 바깥으로 밀린다 — 안 밀면 GUI 범위 안 물병(37.7cm)을 팔이 거부했다
        self.declare_parameter("arm_reach_offsets", [0.0])
        # 2026-10-22: 따르기는 컵을 늘 가운데 한 자리(두 팔 다 닿는 곳)에 두고 한다 (사용자 —
        # 위치가 매번 다르면 보정이 안 맞는다). 컵이 거기 없으면 먼저 옮긴 다음 따른다.
        # 첫 팔(a) 기준 컵 기준점(박스 아래 변) 좌표. 두 팔 모두 받침대 38.8cm, 컵 축 36cm.
        self.declare_parameter("pour_spot", [0.33, -0.205])
        self.declare_parameter("pour_spot_tol", 0.03)
        # 2026-10-22: 물체 종류별 잡는 높이 (그리퍼 중심, 책상 기준). 컵은 아래로 갈수록 좁아
        # 4cm 에서 지름 ~6cm -> 그리퍼 벌림 7.5cm 와 한쪽 여유 2.5mm -> 7mm (6cm 높이는 지름 7cm 라
        # 3mm 만 옆으로 틀려도 손가락이 컵을 밀어 빈손). 물병은 굵기 일정 + 따르기 계산이 6cm 전제.
        # 없는 종류는 target_resolver grasp_height 그대로.
        self.declare_parameter("grasp_height_labels", ["cup"])
        self.declare_parameter("grasp_height_values", [0.04])
        self.declare_parameter("place_clearance", 0.005)
        self.declare_parameter("pour_snap_r", 0.06)     # 놓을 곳이 물체에서 이만큼 안이면 따르기
        # 2026-10-22: 위치는 모두 '물체 바닥 중심' 하나로 쓴다 = 검출점(박스 아래 변) + 팔별 집기 보정
        # (arm_server). 2026-10-12 실측으로 아래 변이 이미 컵 중심에 가까웠고(3.5cm 당기면 4cm 짧게
        # 잡음), 남는 차이는 집기 보정이 맞춘다. 예전엔 따르기만 3cm 더 밀어(pour_cup_center_shift)
        # 컵을 놓은 점과 따르는 점이 달랐다.
        # 두 팔 동시 따르기: 컵 팔이 컵을 자리로 옮기는 동안 병 팔은 병을 들고 기다린다. 병이 따르기
        # 자리에서 이보다 가까우면(공용 구역 안) 동시에 하지 않고 차례로.
        self.declare_parameter("parallel_pour", True)
        self.declare_parameter("parallel_min_bottle_dist", 0.15)
        self.declare_parameter("redetect_r", 0.10)
        self.declare_parameter("redetect_delay", 2.0)   # 컵 팔 RETREAT 후 이만큼 뒤에 컵을 찾고 go [s]      # 컵 놓은 뒤 이 반경 안의 컵을 다시 찾아 따른다
        self.declare_parameter("pour_min_r", 0.25)      # 컵 거리 (pour.py 계획 + 팔 B 서보 한계 IK 로 확인, 컵 15cm·입구 +5cm)
        self.declare_parameter("pour_max_r", 0.40)      # 입구 25cm 기준 (0.41 부터 105° 에서 IK 불가)

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
        # 2026-10-26: 시선 모드 놓을 곳 — 광선-책상 교점은 머리가 낮아 초록 띠 밖(46~73cm)으로 찍혀 늘 거부됐다.
        # 머리 -> 응시점 방향(책상 위 직선)을 따라 집는 팔의 초록 띠 안에서 교점에 가장 가까운 점에 놓는다.
        self.declare_parameter("place_along_ray", False)
        self._head = None
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseStamped, "/head/pose",
                                 lambda m: setattr(self, "_head", (np.array([m.pose.position.x, m.pose.position.y]),
                                                                   time.time())), be)
        # 미리보기: 물체를 고른 뒤 보는 동안 실제로 놓일 자리를 GUI 에 (같은 계산 — 화면과 실제가 안 어긋나게)
        from geometry_msgs.msg import PointStamped
        self._ray_pt = None
        self._place_pref_arm = None
        self.create_subscription(PointStamped, "/gaze/ray_point",
                                 lambda m: setattr(self, "_ray_pt", (np.array([m.point.x, m.point.y, m.point.z]),
                                                                     time.time())), be)
        self.pub_place_prev = self.create_publisher(PointStamped, "/task/place_preview", 10)
        self.create_timer(0.1, self._publish_place_preview)
        self.create_subscription(Empty, "/task/cancel", self.on_cancel, 10)

        self.pub_role = self.create_publisher(String, "/task/expected_role", 10)
        self.pub_state = self.create_publisher(String, "/task/state", 10)
        # 2026-10-14: 거부/실패 이유를 GUI 화면에 띄운다 (로그에만 남아 '왜 안 되는지' 몰랐다)
        self.pub_msg = self.create_publisher(String, "/task/message", 10)

        self.arms = []                # [(이름, TableHomography, ActionClient)]
        names = [n for n in self.get_parameter("arms").value if n]
        homs = list(self.get_parameter("arm_homographies").value)
        acts = list(self.get_parameter("arm_actions").value)
        if len(names) >= 2:
            if not (len(homs) == len(acts) == len(names)):
                raise RuntimeError("arms / arm_homographies / arm_actions 길이가 다르다")
            for n, h, a in zip(names, homs, acts):
                H = TableHomography()
                ok, why = H.load(os.path.expanduser(h))
                if not ok:
                    raise RuntimeError(f"팔 {n} 호모그래피: {why}")
                self.arms.append((n, H, ActionClient(self, PickPlace, a)))
            self.get_logger().info(f"두 팔 모드: {names} — 컵에 가까운 팔이 움직입니다.")
        self.client = ActionClient(self, PickPlace, "/pick_place") if not self.arms else None
        self.pick_arm = None          # 두 팔 모드에서 고른 팔 (arms 인덱스)
        self.pick_cands = []          # 집을 곳에 닿는 팔들 (가까운 순)
        self.objects = []             # 검출 물체 (첫 팔 base_link)
        self.task = ""                # "" | "pour"
        self.labels = []              # objects 와 같은 순서의 클래스 이름
        self.pick_label = None
        self.create_subscription(PoseArray, "/objects/poses", self.on_objects, 10)
        self.create_subscription(String, "/objects/labels", self.on_labels, 10)
        self.pour_calib = False       # GUI p: 따르기 대신 보정 기록 (task "pour_calib")
        self.create_subscription(Bool, "/task/pour_calib",
                                 lambda m: setattr(self, "pour_calib", bool(m.data)), 10)
        self.grasp_calib = False      # GUI g: 집기 직전 멈춰 보정 (task "grasp_calib")
        self.create_subscription(Bool, "/task/grasp_calib",
                                 lambda m: setattr(self, "grasp_calib", bool(m.data)), 10)
        self.queue = []               # 여러 단계 태스크 (컵을 따르기 자리로 -> 따르기)
        self.par = None               # 두 팔 동시 따르기 진행 상태 (_start_parallel)
        self.pub_gate = {n: self.create_publisher(String, a.rsplit("/", 1)[0] + "/pour_gate", 10)
                         for n, a in zip(names, acts)} if self.arms else {}

        self.create_timer(0.2, self.tick)
        self.set_state(IDLE)
        self.get_logger().info("task_manager 시작 — 컵을 응시하세요.")

    def _say(self, m):
        self.get_logger().info(m)
        self.pub_msg.publish(String(data=m))

    def _next_step(self):
        """queue 의 다음 단계를 보낸다. 없으면 False."""
        if not self.queue:
            return False
        st = self.queue.pop(0)
        self.pick_point, self.pick_label, self.task = st["pick"], st["label"], st["task"]
        if self.task == "" and self.grasp_calib:
            self.task = "grasp_calib"
        self.try_order = self.order_for(st["target"], st["what"])
        if not self.try_order:
            self.queue = []
            self.pick_point = None
            self.set_state(IDLE)
            return True
        self.try_target, self.try_reasons = st["target"], []
        self.pick_arm = self.try_order.pop(0)
        self.send_goal_dual(self.pick_point, st["target"])
        return True

    # ------------------------------------------------------------------
    # 두 팔 동시 따르기 (2026-10-22)
    #   컵 팔: 컵 -> 따르기 자리.  병 팔: 병을 들고 자기 쪽에서 대기("pour_wait").
    #   공용 구역은 한 팔씩 — 컵 팔이 컵을 놓고 그리퍼를 열어 빠지기 시작하면(RETREAT) 병 팔에 go.
    #   병 팔 목표는 컵 팔이 실제로 움직이기 시작한 뒤(첫 피드백) 보낸다 — 컵 팔이 [거부]되면
    #   병을 괜히 들었다 내리지 않고 예전처럼 차례로 한다.
    # ------------------------------------------------------------------
    def _start_parallel(self, cup_step, pour_step):
        if not self.get_parameter("parallel_pour").value or len(self.arms) < 2:
            return False
        spot = np.array(cup_step["target"][:2], float)
        if np.linalg.norm(pour_step["pick"][:2] - spot) < float(
                self.get_parameter("parallel_min_bottle_dist").value):
            self.get_logger().info("물병이 공용 구역 안 — 동시에 하지 않고 차례로")
            return False
        cup_order = self.order_for(cup_step["target"], cup_step["what"], pick=cup_step["pick"],
                                   pour=False, quiet=True)
        bot_order = self.order_for(pour_step["target"], "컵", pick=pour_step["pick"],
                                   pour=True, quiet=True)
        pair = next(((i, j) for i in cup_order for j in bot_order if j != i), None)
        if pair is None:
            self.get_logger().info("컵 팔과 병 팔을 따로 고를 수 없음 — 차례로")
            return False
        ci, bi = pair
        self.queue = []
        self.par = {"cup": ci, "bot": bi, "cup_step": cup_step, "pour_step": pour_step,
                    "cup_h": None, "bot_h": None, "bot_sent": False, "gate": False,
                    "cup_res": None, "bot_res": None}
        self._say(f"두 팔 동시: 팔 {self.arms[ci][0]} 컵 -> 따르기 자리, "
                  f"팔 {self.arms[bi][0]} 물병 들고 대기 -> 컵 팔이 빠지면 따르기")
        self.set_state(EXECUTING)
        par = self.par
        goal = self._build_goal(ci, cup_step["pick"], cup_step["target"], "cup", "")
        fut = self.arms[ci][2].send_goal_async(
            goal, feedback_callback=lambda f: self._par_feedback(par, "cup", f))
        fut.add_done_callback(lambda f: self._par_accepted(par, "cup", f))
        return True

    def _par_send_bottle(self, par):
        par["bot_sent"] = True
        st = par["pour_step"]
        goal = self._build_goal(par["bot"], st["pick"], st["target"], "bottle", st["task"] + "_wait")
        fut = self.arms[par["bot"]][2].send_goal_async(
            goal, feedback_callback=lambda f: self._par_feedback(par, "bot", f))
        fut.add_done_callback(lambda f: self._par_accepted(par, "bot", f))

    def _par_accepted(self, par, who, future):
        h = future.result()
        if not h.accepted:
            self._par_done(par, who, False, "[거부] 액션 목표 거부")
            return
        par[who + "_h"] = h
        h.get_result_async().add_done_callback(
            lambda f: self._par_done(par, who, f.result().result.success, f.result().result.message))

    def _par_feedback(self, par, who, fb):
        f = fb.feedback
        self.get_logger().info(f"  진행[{self.arms[par[who]][0]}]: {f.phase} ({f.progress * 100:.0f}%)")
        if par is not self.par or who != "cup":
            return
        if not par["bot_sent"]:
            self._par_send_bottle(par)          # 컵 팔이 움직이기 시작함 = 사전 검사 통과
        if f.phase == "RETREAT" and not par["gate"] and not par.get("timer"):
            # 빠지기 시작한 순간엔 그리퍼가 컵을 가려 재검출이 3번 연속 실패 -> 조금 빠진 뒤 찾는다
            def fire():
                par["timer"].cancel()
                if par is self.par and not par["gate"]:
                    self._par_gate(par, "go")
            par["timer"] = self.create_timer(
                float(self.get_parameter("redetect_delay").value), fire)

    def _cup_near(self, spot):
        """놓인 컵을 다시 찾는다 (첫 팔 기준 xy). 없으면 None."""
        r = float(self.get_parameter("redetect_r").value)
        best = None
        for i, o in enumerate(self.objects):
            d = float(np.linalg.norm(o - spot))
            if self._label(i) == "cup" and d < r and (best is None or d < best[0]):
                best = (d, o)
        return None if best is None else np.array(best[1], float)

    def _par_gate(self, par, what):
        par["gate"] = True
        name = self.arms[par["bot"]][0]
        msg = what
        if what == "go":
            spot = np.array(par["pour_step"]["target"][:2], float)
            cup = self._cup_near(spot)
            if cup is None:
                self._warn("놓인 컵이 안 보입니다 — 따르기 자리 기준으로 따릅니다")
                cup = spot
            else:
                self.get_logger().info(
                    f"놓인 컵 검출 ({cup[0]:.3f}, {cup[1]:.3f}) — 따르기 자리에서 "
                    f"{np.linalg.norm(cup - spot) * 100:.1f}cm")
            c = self.to_arm(par["bot"], np.array([cup[0], cup[1], 0.0]))
            msg = f"go:{c[0]:.4f}:{c[1]:.4f}"
        self.get_logger().info(f"팔 {name} 에 {msg}")
        self.pub_gate[name].publish(String(data=msg))

    def _par_done(self, par, who, ok, message):
        if par is not self.par:
            return
        par[who + "_res"] = (ok, message)
        name = self.arms[par[who]][0]
        (self.get_logger().info if ok else self._warn)(f"결과[{name}]: {message}")
        if who == "cup" and not ok:
            if not par["bot_sent"]:
                # 컵 팔이 움직이기 전에 거부 -> 예전처럼 차례로 (다른 팔로 재시도 포함)
                self.par = None
                self.queue = [par["cup_step"], par["pour_step"]]
                self._next_step()
                return
            if not par["gate"]:
                self._par_gate(par, "abort")      # 병 팔: 병 제자리
        if who == "cup" and ok and not par["gate"]:
            self._par_gate(par, "go")             # RETREAT 피드백을 못 받았어도
        if who == "bot" and not ok and message.startswith("[거부]"):
            par["retry_pour"] = True              # 병 팔이 움직이기 전에 거부 -> 컵 놓은 뒤 따르기만 차례로
        if par["cup_res"] is None or (par["bot_sent"] and par["bot_res"] is None):
            return
        self.par = None
        if par.get("retry_pour") and par["cup_res"][0]:
            self.queue = [par["pour_step"]]
            self._next_step()
            return
        self.pick_point = None
        self.set_state(IDLE)
        if par["cup_res"][0] and par["bot_res"] and par["bot_res"][0]:
            self._say("두 팔 따르기 완료")

    def _warn(self, m):
        self.get_logger().warn(m)
        self.pub_msg.publish(String(data=m))

    def _err(self, m):
        self.get_logger().error(m)
        self.pub_msg.publish(String(data=m))

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
            self._warn("선택 타임아웃 — 초기 상태로 돌아갑니다.")
            self.pick_point = None
            self.set_state(IDLE)

    # ------------------------------------------------------------------
    def on_cancel(self, _msg):
        self.queue = []
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
        if self.par:
            for h in (self.par["cup_h"], self.par["bot_h"]):
                if h is not None:
                    h.cancel_goal_async()
            self.par = None
        self.pick_point = None
        self.set_state(IDLE)
        self._warn("사용자 취소")

    def on_target(self, msg: GazeTarget):
        p = np.array([msg.point.x, msg.point.y, msg.point.z])

        # 안전장치 3: 신뢰도가 낮은 목표는 거부한다.
        # (dwell 산포가 크거나 깜빡임이 섞인 응시는 여기서 걸러진다)
        if msg.confidence < self.min_confidence:
            self._warn(
                f"신뢰도가 낮아 목표를 무시합니다 "
                f"({msg.confidence:.2f} < {self.min_confidence:.2f}). "
                "조금 더 가만히 응시해 주세요."
            )
            return

        # 목표가 어느 좌표계인지 기억해 두었다가 액션 목표에 같이 실어 보낸다.
        # 이게 없으면 arm_server 가 map 기준으로 넘겨받았다고 착각해 변환을 한 번 더 건다.
        self.target_frame = msg.header.frame_id

        if self.state == IDLE and msg.role == "pick":
            if self.arms:
                self.pick_cands = self.pick_candidates(p, self.label_at(p))
                if not self.pick_cands:
                    return
            self.pick_point = p
            self.pick_label = self.label_at(p)
            self.set_state(PICK_SELECTED)
            self.get_logger().info(
                f"물체 선택됨: {self.pick_label or '물체'} ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}). "
                + ("컵을 고르면 따르기, 빈 곳을 고르면 옮기기." if self.pick_label == "bottle"
                   else "이제 옮길 자리를 응시하세요.")
            )

        elif self.state == PICK_SELECTED and msg.role == "place":
            sep = float(np.linalg.norm(p - self.pick_point))
            if sep < self.min_separation:
                self._warn(
                    f"놓을 자리가 집을 자리와 너무 가깝습니다 ({sep * 100:.1f}cm). "
                    "조금 더 떨어진 곳을 보세요."
                )
                return
            if self.arms:
                kind, cup = self.pour_target(p)
                if kind == "blocked":
                    self._warn(
                        f"놓을 곳에 {cup} 이(가) 있습니다 — 부딪힙니다. 빈 곳을 고르세요"
                        + (" (물병을 고른 뒤 컵을 고르면 따르기)" if self.pick_label != "bottle" else ""))
                    return
                if kind == "pour":
                    spot = np.array(self.get_parameter("pour_spot").value, float)
                    anchor = self._cup_anchor
                    pour_task = "pour_calib" if self.pour_calib else "pour"
                    bottle = (self.pick_point, self.pick_label)
                    pour_step = {"pick": bottle[0], "label": "bottle", "task": pour_task,
                                 "target": np.array([spot[0], spot[1], 0.0]), "what": "컵"}
                    tol = float(self.get_parameter("pour_spot_tol").value)
                    # 2026-10-22: 따르기 자리가 다른 물건(컵·물병 말고)으로 차 있으면 거부 — 자리를 자동으로
                    # 옮기면 '늘 같은 자세로 따른다'는 장점이 사라진다 (사용자 결정)
                    others = [o for o in self.objects
                              if np.linalg.norm(o - anchor) > 0.03
                              and np.linalg.norm(o - self.pick_point[:2]) > 0.03]
                    blocking = [o for o in others if np.linalg.norm(o - spot) < 0.08]
                    if blocking:
                        self._warn("따르기 자리에 다른 물건이 있습니다 → 따르기 자리(점선 원)를 비우고 다시 고르세요")
                        return
                    if np.linalg.norm(self.pick_point[:2] - spot) < 0.08:
                        self._warn("물병이 따르기 자리에 있습니다 → 물병을 옆으로 옮기고 다시 고르세요")
                        return
                    if np.linalg.norm(anchor - spot) > tol:
                        # 먼저 컵을 따르기 자리로
                        cup_step = {"pick": np.array([anchor[0], anchor[1], 0.06]), "label": "cup",
                                    "task": "", "target": np.array([spot[0], spot[1], 0.065]),
                                    "what": "따르기 자리"}
                        self.queue = [cup_step, pour_step]
                        self._say(f"컵을 가운데 따르기 자리로 먼저 옮긴 뒤 따릅니다 "
                                  f"(컵이 자리에서 {np.linalg.norm(anchor - spot) * 100:.0f}cm 떨어짐)")
                        if self._start_parallel(cup_step, pour_step):
                            return
                    else:
                        # 컵이 이미 자리에 있으면 검출된 컵 바닥 중심으로 (동시 경로의 재검출과 같은 기준)
                        pour_step["target"] = np.array([anchor[0], anchor[1], 0.0])
                        self.queue = [pour_step]
                    self._next_step()
                    return
                self.task = "grasp_calib" if self.grasp_calib else ""
                target, what = p, "놓을 곳"
                pref = None
                if bool(self.get_parameter("place_along_ray").value):
                    got = self._place_along_ray(p)
                    if got is not None:
                        target, pref = got
                self.try_order = self.order_for(target, what)
                if not self.try_order:
                    return            # 집기 선택은 유지 — 다른 놓을 곳을 보면 된다
                if pref is not None and pref in self.try_order:
                    # 놓을 곳을 정한 팔(바라본 곳에 가까운 팔)이 맡는다
                    self.try_order.remove(pref)
                    self.try_order.insert(0, pref)
                self.try_target, self.try_reasons = target, []
                self.pick_arm = self.try_order.pop(0)
                self.send_goal_dual(self.pick_point, target)
            else:
                self.send_goal(self.pick_point, p)

    # ------------------------------------------------------------------
    def send_goal(self, pick, place):
        if not self.client.wait_for_server(timeout_sec=2.0):
            self._err("/pick_place 액션 서버가 없습니다. arm_server를 켜세요.")
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

    # ------------------------------------------------------------------
    # 두 팔
    # ------------------------------------------------------------------
    def to_arm(self, i, p):
        """첫 팔 기준 점 -> i 번째 팔 기준. 같은 픽셀을 각 팔의 호모그래피로 푼다 (z 는 그대로)."""
        if i == 0:
            return np.array(p, dtype=float)
        px = self.arms[0][1].to_pixel(p[0], p[1])
        xy = self.arms[i][1].to_robot(*px) if px else None
        if xy is None:
            raise RuntimeError("호모그래피 변환 실패")
        return np.array([xy[0], xy[1], p[2]])

    def _dists(self, p):
        return [float(np.linalg.norm(self.to_arm(i, p)[:2])) for i in range(len(self.arms))]

    def _desc(self, rs):
        return ", ".join(f"{n} {r * 100:.1f}cm" for (n, _, _), r in zip(self.arms, rs))

    def pick_candidates(self, p, label=None):
        """집을 곳에 닿는 팔들, 가까운 순.

        2026-10-14: 물체 종류로 가까운 쪽 한계를 바로 건다 — 컵은 위가 넓어 뒤에서 수평으로 밀고
        들어가야 해서 36cm~, 물병은 위에서 내려와 27cm~ (arm_server side_tilt/vertical_min_radius).
        예전엔 놓을 곳까지 고른 뒤에야 거부돼 '놓을 곳이 문제' 처럼 보였다."""
        lo = float(self.get_parameter("pick_min_r").value)    # 컵·물병 모두 뒤에서 수평 접근 (36cm~)
        hi = float(self.get_parameter("hard_max_r").value)
        rs = self._dists(p)
        cands = sorted((i for i in range(len(rs)) if lo + self._ro(i) <= rs[i] <= hi + self._ro(i)),
                       key=lambda i: rs[i])
        if not cands:
            # 가장 가까운 팔 기준으로 '가깝다/멀다' (예전엔 한 팔은 가깝고 다른 팔은 멀면 '멉니다' 로 잘못 나왔다)
            i0 = int(np.argmin(rs))
            lo, hi = lo + self._ro(i0), hi + self._ro(i0)
            near = rs[i0] < lo
            name = {"cup": "컵", "bottle": "물병"}.get(label, "물체")
            self._warn(
                f"{name}이(가) 팔 {self.arms[i0][0]} 받침대에 너무 "
                f"{'가깝습니다 → 팔에서 ' + format((lo - rs[i0]) * 100 + 1, '.0f') + 'cm 이상 멀리' if near else '멉니다 → 초록 띠 안으로'} 옮기세요 "
                f"({self._desc(rs)}; {lo * 100:.0f}~{hi * 100:.0f}cm)")
        else:
            self.get_logger().info(
                f"집을 곳 거리: {self._desc(rs)} -> 닿는 팔 "
                f"{[self.arms[i][0] for i in cands]}")
        return cands

    def _ro(self, i):
        """팔 i 의 뻗음 보정 [m] (없으면 0)."""
        offs = list(self.get_parameter("arm_reach_offsets").value)
        return float(offs[i]) if i < len(offs) else 0.0

    def order_for(self, p, what, pick=None, pour=None, quiet=False):
        """2026-10-14(5): 팔 고르기 — 집을 곳과 놓을 곳(또는 컵)을 **둘 다** 본다 (사용자).
        팔마다 둘 다 그 팔 범위 안인지 보고, 되는 팔 중 두 거리 합이 작은 팔부터 (arm_server 가
        실제 IK 로 최종 확인, 안 되면 다음 팔)."""
        hi = float(self.get_parameter("hard_max_r").value)
        cup_lo = float(self.get_parameter("pick_min_r").value)
        pick_lo = cup_lo
        if pour is None:
            pour = self.task in ("pour", "pour_calib")
        pr, lr = self._dists(self.pick_point if pick is None else pick), self._dists(p)
        ok, why = [], []
        for i, (n, _, _) in enumerate(self.arms):
            bad = []
            ro = self._ro(i)
            if not pick_lo + ro <= pr[i] <= hi + ro:
                bad.append(f"집기 {pr[i] * 100:.0f}cm")
            if pour:
                # 컵 중심 거리 (pour.py + IK 로 확인한 범위). 2026-10-26: 상한은 pour_max_r 설정 (옛 0.40 고정)
                if not 0.23 <= lr[i] <= float(self.get_parameter("pour_max_r").value):
                    bad.append(f"컵 {lr[i] * 100:.0f}cm")
            elif not pick_lo + ro <= lr[i] <= hi + ro:  # 같은 물체를 놓으므로 같은 범위
                bad.append(f"놓기 {lr[i] * 100:.0f}cm")
            if bad:
                why.append(f"팔 {n}: " + ", ".join(bad))
            else:
                ok.append(i)
        order = sorted(ok, key=lambda i: pr[i] + lr[i])
        rng = (f"집기 {pick_lo * 100:.0f}~{hi * 100:.0f}cm, "
               + (f"컵 23~{float(self.get_parameter('pour_max_r').value) * 100:.0f}cm" if pour
                  else f"놓기 {pick_lo * 100:.0f}~{hi * 100:.0f}cm"))
        if quiet:
            return order
        if not order:
            self._warn(f"집기와 {what} 둘 다 닿는 팔이 없습니다 → {what} 다시 고르세요 ("
                       + " / ".join(why) + f"; {rng})")
        else:
            self.get_logger().info(
                f"팔 고르기: " + ", ".join(f"{self.arms[i][0]} 집기 {pr[i] * 100:.0f}+{what} "
                                         f"{lr[i] * 100:.0f}cm" for i in order)
                + (f" (안 됨: {' / '.join(why)})" if why else "") + f" -> {self.arms[order[0]][0]}")
        return order

    def on_objects(self, msg):
        self.objects = [np.array([q.position.x, q.position.y]) for q in msg.poses]

    def on_labels(self, msg):
        try:
            self.labels = list(json.loads(msg.data))
        except ValueError:
            self.labels = []

    def _label(self, i):
        return self.labels[i] if i < len(self.labels) and len(self.labels) == len(self.objects) \
            else None

    def label_at(self, p, r=0.06):
        best = None
        for i, o in enumerate(self.objects):
            d = float(np.linalg.norm(o - p[:2]))
            if d < r and (best is None or d < best[0]):
                best = (d, i)
        return self._label(best[1]) if best else None

    def pour_target(self, p):
        """놓을 곳 판정. 돌려주는 것: ("pour", 컵 축) | ("move", None) | ("blocked", 이름).

        물병 -> 컵 이면 따르기. 빈 곳이면 옮기기. 다른 물체 위면 부딪히므로 막는다."""
        snap = float(self.get_parameter("pour_snap_r").value)
        best = None
        for i, o in enumerate(self.objects):
            if np.linalg.norm(o - self.pick_point[:2]) < 0.08:
                continue                       # 집은 물체 자신
            d = float(np.linalg.norm(o - p[:2]))
            if d < snap and (best is None or d < best[0]):
                best = (d, i)
        if best is None:
            return "move", None
        o, lab = self.objects[best[1]], self._label(best[1])
        if not (self.pick_label == "bottle" and lab == "cup"):
            return "blocked", lab or "물체"
        self._cup_anchor = np.array(o[:2], float)
        self.get_logger().info(f"물병 -> 컵: 따르기. 컵 바닥 중심 ({o[0]:.3f}, {o[1]:.3f})")
        return "pour", np.array([o[0], o[1], 0.0])

    def _publish_place_preview(self):
        from geometry_msgs.msg import PointStamped
        if (self.state != PICK_SELECTED or self.task in ("pour", "pour_calib")
                or not bool(self.get_parameter("place_along_ray").value)
                or self._ray_pt is None or time.time() - self._ray_pt[1] > 0.5):
            return
        got = self._place_along_ray(self._ray_pt[0], quiet=True)
        m = PointStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.target_frame or "base_link"
        if got is None:
            m.point.x = m.point.y = m.point.z = float("nan")      # 놓을 수 있는 곳 없음
        else:
            m.point.x, m.point.y, m.point.z = float(got[0][0]), float(got[0][1]), float(got[1])  # z = 팔 번호
        self.pub_place_prev.publish(m)

    def _place_along_ray(self, p, quiet=False):
        """머리 -> 응시점 방향 직선 위에서, 물체를 집을 수 있는 팔의 초록 띠(뻗음 보정 포함) 안이면서
        다른 물체와 8cm 이상 떨어진 점 중 원래 응시점에 가장 가까운 점 -> (점, 팔 번호). 없으면 None.
        2026-10-26: 팔은 바라본 곳(원래 응시점)에 가까운 팔부터 — 그 팔 띠에 없을 때만 다른 팔."""
        if self._head is None or time.time() - self._head[1] > 1.0:
            return None
        h = self._head[0]
        d = np.asarray(p[:2], float) - h
        if float(np.linalg.norm(d)) < 0.05:
            return None
        lo = float(self.get_parameter("pick_min_r").value)
        hi = float(self.get_parameter("hard_max_r").value)
        pr = self._dists(self.pick_point)
        arms_ok = [i for i in range(len(self.arms)) if lo + self._ro(i) <= pr[i] <= hi + self._ro(i)]
        gaze_d = self._dists(np.array([p[0], p[1], 0.0]))
        arms_ok.sort(key=lambda i: gaze_d[i])               # 바라본 곳에 가까운 팔부터
        others = [o for o in self.objects if np.linalg.norm(np.asarray(o[:2]) - self.pick_point[:2]) > 0.03]
        for i in arms_ok:
            best = None
            for t in np.linspace(0.3, 1.6, 131):
                q = h + d * t
                q3 = np.array([q[0], q[1], p[2]])
                if np.linalg.norm(q - self.pick_point[:2]) < max(self.min_separation, 0.05):
                    continue
                if any(np.linalg.norm(q - np.asarray(o[:2])) < 0.08 for o in others):
                    continue
                r = self._dists(q3)[i]
                if not lo + self._ro(i) + 0.005 <= r <= hi + self._ro(i) - 0.005:
                    continue
                if best is None or abs(t - 1.0) < abs(best[0] - 1.0):
                    best = (t, q3)
            if best is not None:
                q3 = best[1]
                if not quiet:
                    self.get_logger().info(
                        f"놓을 곳을 시선 방향 팔 {self.arms[i][0]} 초록 띠 안으로: ({p[0]:.3f}, {p[1]:.3f}) -> "
                        f"({q3[0]:.3f}, {q3[1]:.3f})")
                return q3, i
        return None

    def clamp_place(self, q):
        lo = float(self.get_parameter("place_min_r").value)
        hi = float(self.get_parameter("place_max_r").value)
        r = float(np.linalg.norm(q[:2]))
        r_c = min(max(r, lo), hi)
        if r > 1e-6 and abs(r_c - r) > 1e-4:
            self.get_logger().info(
                f"놓을 자리 범위 보정(팔 {self.arms[self.pick_arm][0]} 기준): "
                f"r={r:.3f} -> {r_c:.3f}")
            q = np.array([q[0] * r_c / r, q[1] * r_c / r, q[2]])
        return q

    def send_goal_dual(self, pick, place):
        i = self.pick_arm
        name, _, client = self.arms[i]
        if not client.wait_for_server(timeout_sec=2.0):
            self._err(f"팔 {name} 액션 서버가 없습니다.")
            self.pick_point = None
            self.set_state(IDLE)
            return
        goal = self._build_goal(i, pick, place, self.pick_label, self.task)
        self.set_state(EXECUTING)
        fut = client.send_goal_async(goal, feedback_callback=self.on_feedback)
        fut.add_done_callback(self.on_goal_response)

    def _build_goal(self, i, pick, place, label, task):
        name = self.arms[i][0]
        pk = self.to_arm(i, pick)
        pl = self.to_arm(i, place)        # 거리 보정 안 함 — 되는지는 arm_server 가 판단
        pour = task.startswith("pour")
        gh = dict(zip(self.get_parameter("grasp_height_labels").value,
                      self.get_parameter("grasp_height_values").value)).get(label)
        if gh is not None:
            pk[2] = float(gh)
            if not pour:                                          # 따르기면 place 는 컵 바닥 중심(높이 무관)
                pl[2] = float(gh) + float(self.get_parameter("place_clearance").value)
        goal = PickPlace.Goal()
        goal.pick = Point(x=float(pk[0]), y=float(pk[1]), z=float(pk[2]))
        goal.place = Point(x=float(pl[0]), y=float(pl[1]), z=float(pl[2]))
        goal.frame_id = "base_link"          # 고른 팔의 base_link (arm_server 는 변환 안 함)
        goal.approach_pitch = self.approach_pitch
        goal.dry_run = self.dry_run
        goal.task = task
        goal.pick_label = label or ""
        kind = {"pour": "따르기", "pour_calib": "따르기 보정"}.get(task.removesuffix("_wait"), "")
        self.get_logger().info(
            f"팔 {name} 에 {kind}{'(대기)' if task.endswith('_wait') else ''} 목표 전달 — "
            f"집기 {np.round(pk, 3)}, 놓기 {np.round(pl, 3)}")
        return goal

    def on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self._err("액션 목표가 거부되었습니다.")
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
        if (not result.success and result.message.startswith("[거부]") and self.arms
                and getattr(self, "try_order", None)):
            # 움직이기 전에 거부 -> 다음 팔
            self.try_reasons.append(f"팔 {self.arms[self.pick_arm][0]}: {result.message[5:]}")
            self.pick_arm = self.try_order.pop(0)
            self._warn(f"{self.try_reasons[-1]} -> 팔 {self.arms[self.pick_arm][0]} 로 다시")
            self._goal_handle = None
            self.send_goal_dual(self.pick_point, self.try_target)
            return
        if not result.success and result.message.startswith("[거부]") and self.arms:
            # 모든 팔이 움직이기 전에 거부 -> 집기 선택은 유지, 다른 놓을 곳을 고르게
            reasons = getattr(self, "try_reasons", []) + [
                f"팔 {self.arms[self.pick_arm][0]}: {result.message[5:]}"]
            self.queue = []          # 여러 단계 중이었으면 남은 단계도 취소
            pick_bad = all("집을 위치" in r or "병 위치" in r for r in reasons)
            if pick_bad:
                # 집을 물체 자체가 어느 팔로도 안 잡힘 -> 놓을 곳을 바꿔도 소용없다 (2026-10-14)
                self._err("집을 물체를 못 잡습니다 → 물체를 초록 띠 안으로 옮기고 다시 고르세요 ("
                          + " / ".join(reasons) + ")")
                self._goal_handle = None
                self.pick_point = None
                self.set_state(IDLE)
                return
            self._err("놓을 곳(컵)에 닿는 팔이 없습니다 → 다시 고르세요 (" + " / ".join(reasons) + ")")
            self._goal_handle = None
            self.set_state(PICK_SELECTED)
            return
        if result.success:
            self.get_logger().info(f"완료: {result.message}")
            if self.queue:
                self._goal_handle = None
                self._say("다음 단계: " + self.queue[0]["what"] + " 쪽 작업")
                self._next_step()
                return
        else:
            self._err(f"실패: {result.message}")
            self.queue = []
        self._goal_handle = None
        self.pick_point = None
        self.set_state(IDLE)


def main():
    rclpy.init()
    node = TaskManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():                # SIGINT/SIGTERM 이면 rclpy 가 이미 내렸다
            rclpy.shutdown()


if __name__ == "__main__":
    main()
