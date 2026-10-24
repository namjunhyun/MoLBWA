#!/usr/bin/env python3
"""
[노드 1] gaze_bridge  —  "발행(publish) 쪽"

기존 MoLBWA 융합 파이프라인(docs/03_fusion.md의 gaze_point_world)의 출력을
ROS2 토픽으로 흘려보내는 다리 역할입니다.

발행하는 것:
  /gaze/point_raw   geometry_msgs/PointStamped   시선 3D점 p_W (frame: map)
  /gaze/valid       std_msgs/Bool                깜빡임/추적실패 여부
  /head/pose        geometry_msgs/PoseStamped    SLAM 헤드 pose T_WS
  TF: map -> head   (RViz 시각화 및 tf2 변환용)

입력 소스(--source):
  fake  : 하드웨어 없이 합성 데이터. 지금 당장 전체 파이프라인 테스트 가능.
  udp   : 융합 프로세스가 UDP JSON으로 쏘는 걸 받음. (권장: 기존 코드 안 건드림)
  inproc: 이 노드 안에서 직접 융합 코드를 import 해서 돌림.

왜 UDP를 권하냐면, ORB-SLAM3 + pye3d + OpenCV 파이프라인은 보통
자기만의 파이썬 환경/의존성을 갖고 있어서 rclpy와 한 프로세스에 묶으면
의존성 지옥에 빠지기 쉽습니다. 프로세스를 분리하면 각자 편하게 돌릴 수 있습니다.
"""

import json
import math
import socket
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Int32
from tf2_ros import TransformBroadcaster


class GazeBridge(Node):

    def __init__(self):
        super().__init__("gaze_bridge")

        self.declare_parameter("source", "fake")        # fake | udp | inproc
        self.declare_parameter("udp_port", 55055)
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("head_frame", "head")
        self.declare_parameter("publish_rate", 30.0)
        # 2026-10-22 광선 스냅: 머리 -> 응시점 광선이 물체(파지 높이 중심) 옆을 ray_snap_radius 안으로
        # 지나가면 응시점을 그 물체 중심으로 바꿔 낸다 = "광선이 지나는 물체를 자동 클릭".
        # 머리가 낮아(0.3~0.4m) 광선-책상 교점은 앞뒤로 50cm 까지 튀어 dwell(3cm)이 안 모인다.
        # 물체에 붙이면 점이 고정돼 2초 응시가 그 물체로 확정된다. 1, 2등 차이가 margin 보다
        # 작으면(어느 물체인지 모호) 붙이지 않는다.
        self.declare_parameter("ray_snap", False)
        self.declare_parameter("ray_snap_radius", 0.15)
        self.declare_parameter("ray_snap_height", 0.06)      # target_resolver grasp_height 와 같게
        self.declare_parameter("ray_snap_margin", 0.05)

        self.source = self.get_parameter("source").value
        self.world_frame = self.get_parameter("world_frame").value
        self.head_frame = self.get_parameter("head_frame").value

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_point = self.create_publisher(PointStamped, "/gaze/point_raw", qos)
        self.pub_valid = self.create_publisher(Bool, "/gaze/valid", qos)
        self.pub_head = self.create_publisher(PoseStamped, "/head/pose", qos)
        self.tf_bc = TransformBroadcaster(self)
        self.ray_snap = bool(self.get_parameter("ray_snap").value)
        self._objects, self._objects_t = [], 0.0
        if self.ray_snap:
            self.create_subscription(PoseArray, "/objects/poses", self._on_objects, 10)
            # 화면용: 스냅 전 응시점(광선 그리기)과 붙은 물체 번호(-1 = 없음)
            self.pub_ray_point = self.create_publisher(PointStamped, "/gaze/ray_point", qos)
            self.pub_snapped = self.create_publisher(Int32, "/gaze/snapped", qos)

        # 최신 샘플 (스레드 간 공유)
        self._lock = threading.Lock()
        self._latest = None          # dict: point, valid, head_pose
        self._t0 = time.time()

        if self.source == "udp":
            self._start_udp_listener()
        elif self.source == "inproc":
            self._start_inproc()

        period = 1.0 / float(self.get_parameter("publish_rate").value)
        self.create_timer(period, self._tick)
        self.get_logger().info(f"gaze_bridge 시작 (source={self.source})")

    # ------------------------------------------------------------------
    # 소스 1: UDP JSON 수신
    # ------------------------------------------------------------------
    def _start_udp_listener(self):
        port = int(self.get_parameter("udp_port").value)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))
        sock.settimeout(0.5)
        self.get_logger().info(f"UDP {port} 포트에서 시선 데이터 대기 중")

        def loop():
            while rclpy.ok():
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    msg = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                with self._lock:
                    self._latest = msg

        threading.Thread(target=loop, daemon=True).start()

    # ------------------------------------------------------------------
    # 소스 2: 융합 코드를 직접 import (같은 환경에서 돌 때만)
    # ------------------------------------------------------------------
    def _start_inproc(self):
        # TODO: 팀 저장소의 실제 융합 모듈로 교체하세요.
        #
        #   from molbwa.fusion import GazeFusionPipeline
        #   pipeline = GazeFusionPipeline(config_path="...")
        #   def loop():
        #       for sample in pipeline.stream():
        #           p_W, T_WS, valid = sample.point_world, sample.pose, sample.valid
        #           with self._lock:
        #               self._latest = {
        #                   "point": p_W.tolist(),
        #                   "valid": bool(valid),
        #                   "head": pose_to_dict(T_WS),
        #               }
        #   threading.Thread(target=loop, daemon=True).start()
        self.get_logger().warn(
            "inproc 모드는 아직 연결되지 않았습니다. "
            "gaze_bridge_node.py 의 _start_inproc()에 융합 파이프라인을 연결하세요."
        )

    # ------------------------------------------------------------------
    # 소스 3: 합성 데이터 — 하드웨어 없이 전체 파이프라인 검증용
    # ------------------------------------------------------------------
    def _fake_sample(self):
        """두 개의 컵(A, B)과 놓을 자리(C)를 차례로 응시하는 시나리오.

        0~4초  : 컵 A 응시
        4~6초  : 이동 중 (시선 흔들림)
        6~10초 : 자리 C 응시
        반복
        """
        t = (time.time() - self._t0) % 14.0
        cup_a = (0.30, -0.12, 0.05)
        cup_b = (0.30, 0.12, 0.05)
        drop_c = (0.42, 0.00, 0.00)

        if t < 4.0:
            base, jitter, valid = cup_a, 0.004, True
        elif t < 6.0:
            frac = (t - 4.0) / 2.0
            base = tuple(a + (c - a) * frac for a, c in zip(cup_a, drop_c))
            jitter, valid = 0.05, True       # 사코드 구간: 산포가 커서 dwell 미발생
        elif t < 10.0:
            base, jitter, valid = drop_c, 0.004, True
        elif t < 10.4:
            base, jitter, valid = drop_c, 0.004, False   # 깜빡임
        else:
            base, jitter, valid = cup_b, 0.05, True

        import random
        pt = [c + random.gauss(0, jitter) for c in base]
        head = {"position": [0.0, 0.0, 0.45],
                "orientation": [0.0, 0.0, 0.0, 1.0]}
        return {"point": pt, "valid": valid, "head": head}

    # ------------------------------------------------------------------
    def _on_objects(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.world_frame:
            return
        self._objects = [(p.position.x, p.position.y) for p in msg.poses]
        self._objects_t = time.time()

    def _snap(self, pt, head):
        """광선(머리 -> pt)에 가장 가까운 물체 번호. 없거나 모호하면 -1."""
        if head is None or not self._objects or time.time() - self._objects_t > 1.0:
            return -1
        o = [float(v) for v in head.get("position", [0, 0, 0])]
        d = [pt[i] - o[i] for i in range(3)]
        dd = sum(v * v for v in d)
        if dd < 1e-9:
            return -1
        h = float(self.get_parameter("ray_snap_height").value)
        dists = []
        for x, y in self._objects:
            c = (x, y, h)
            t = sum((c[i] - o[i]) * d[i] for i in range(3)) / dd
            if t <= 0:
                dists.append(float("inf"))
                continue
            dists.append(math.sqrt(sum((o[i] + t * d[i] - c[i]) ** 2 for i in range(3))))
        order = sorted(range(len(dists)), key=lambda i: dists[i])
        best = order[0]
        if dists[best] > float(self.get_parameter("ray_snap_radius").value):
            return -1
        if len(order) > 1 and dists[order[1]] - dists[best] < float(self.get_parameter("ray_snap_margin").value):
            return -1
        return best

    def _tick(self):
        if self.source == "fake":
            sample = self._fake_sample()
        else:
            with self._lock:
                sample = self._latest
            if sample is None:
                return

        now = self.get_clock().now().to_msg()

        valid = bool(sample.get("valid", True))
        self.pub_valid.publish(Bool(data=valid))

        pt = sample.get("point")
        if self.ray_snap and pt is not None and valid and all(math.isfinite(v) for v in pt):
            raw = PointStamped()
            raw.header.stamp = now
            raw.header.frame_id = self.world_frame
            raw.point.x, raw.point.y, raw.point.z = float(pt[0]), float(pt[1]), float(pt[2])
            self.pub_ray_point.publish(raw)
            k = self._snap(pt, sample.get("head"))
            self.pub_snapped.publish(Int32(data=k))
            if k >= 0:
                x, y = self._objects[k]
                pt = [x, y, float(self.get_parameter("ray_snap_height").value)]
        if pt is not None and valid and all(math.isfinite(v) for v in pt):
            msg = PointStamped()
            msg.header.stamp = now
            msg.header.frame_id = self.world_frame
            msg.point.x, msg.point.y, msg.point.z = float(pt[0]), float(pt[1]), float(pt[2])
            self.pub_point.publish(msg)

        head = sample.get("head")
        if head is not None:
            pos = head.get("position", [0, 0, 0])
            ori = head.get("orientation", [0, 0, 0, 1])

            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = self.world_frame
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos)
            (ps.pose.orientation.x, ps.pose.orientation.y,
             ps.pose.orientation.z, ps.pose.orientation.w) = map(float, ori)
            self.pub_head.publish(ps)

            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = self.world_frame
            tf.child_frame_id = self.head_frame
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, pos)
            (tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w) = map(float, ori)
            self.tf_bc.sendTransform(tf)


def main():
    rclpy.init()
    node = GazeBridge()
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
