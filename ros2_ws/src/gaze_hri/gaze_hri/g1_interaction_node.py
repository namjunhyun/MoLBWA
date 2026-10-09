#!/usr/bin/env python3
"""
[데모 2] g1_interaction  —  "G1 의 어느 부위를 3초 봤는가?" -> Jetson 에 동작 명령.

    g1_gaze_bridge ─UDP 55057─> 이 노드 ─UDP 55070─> g1_motion_server (Jetson)
                                       <─UDP 55071─ {state, seq}

  * LabelDwell 로 부위 라벨 확정 (같은 라벨 dwell_time 동안 min_ratio 이상).
  * 확정 시 Jetson state 가 idle 이고 1초 안에 받은 것일 때만 act 를 보낸다.
    bearing_deg = int(round(최근 dwell_time 안의 마지막 유효 bearing)), 없으면 null(허리 정면 y0 동작만).
    어떤 세그먼트를 재생할지는 Jetson 이 g1_protocol.plan() 으로 정한다(허리 ±60°, 넘으면 turn_l180).
  * seq 는 밀리초 시각 기반 int(next_seq), 직전·Jetson 보고값보다 항상 큼 — 노드를 재시작해도 거부되지 않는다.
  * 1 Hz ping.

발행:
  /g1/event          std_msgs/String  JSON {t, part, bearing_deg, segments, seq}  (보낸 명령만)
  /g1/state          std_msgs/String  idle | turning | acting | offline
  /g1/dwell_progress std_msgs/Float32 0~1
"""

import importlib
import json
import math
import os
import socket
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32, String

from gaze_hri.g1_dwell import LabelDwell, next_seq

STATE_TIMEOUT_S = 1.0


def _find_g1_dir():
    """이 파일에서 위로 올라가며 g1/g1_protocol.py 를 찾는다 (src 실행·install 둘 다)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "g1", "g1_protocol.py")):
            return os.path.join(d, "g1")
        d = os.path.dirname(d)
    return ""


def _udp_rx(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    s.setblocking(False)
    return s


def _drain(sock):
    out = []
    while True:
        try:
            out.append(sock.recvfrom(65535)[0])
        except BlockingIOError:
            return out


class G1Interaction(Node):

    def __init__(self):
        super().__init__("g1_interaction")
        self.declare_parameter("jetson_host", "192.168.50.119")
        self.declare_parameter("dwell_time", 3.0)
        self.declare_parameter("min_ratio", 0.7)
        self.declare_parameter("cooldown", 2.0)
        self.declare_parameter("g1_dir", "")         # 비면 저장소 g1/ 을 자동으로 찾는다
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        g1_dir = p("g1_dir") or _find_g1_dir()
        sys.path.insert(0, os.path.expanduser(g1_dir))
        self.proto = importlib.import_module("g1_protocol")   # 못 찾으면 여기서 죽는다

        self.dwell_time = float(p("dwell_time"))
        self.dwell = LabelDwell(self.dwell_time, float(p("min_ratio")), float(p("cooldown")))
        self.jetson = (p("jetson_host"), self.proto.PORT_CMD)
        self.rx_bridge = _udp_rx(self.proto.PORT_BRIDGE)
        self.rx_state = _udp_rx(self.proto.PORT_STATE)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.seq = 0                 # 마지막으로 보낸 act seq
        self.jetson_seq = 0          # Jetson 이 마지막으로 수락했다고 보고한 seq
        self.jetson_state, self.state_t = "offline", -1e9
        self.bearing, self.bearing_t = None, -1e9
        self.bridge_t = -1e9

        self.pub_event = self.create_publisher(String, "/g1/event", 10)
        self.pub_state = self.create_publisher(String, "/g1/state", 10)
        self.pub_progress = self.create_publisher(Float32, "/g1/dwell_progress", 10)
        self.create_timer(0.02, self.tick)
        self.create_timer(1.0, self.ping)
        self.get_logger().info(f"g1_protocol: {self.proto.__file__}, Jetson {self.jetson[0]}:{self.jetson[1]}, "
                               f"수신 {self.proto.PORT_BRIDGE}/{self.proto.PORT_STATE}")

    def _send(self, msg):
        try:
            self.tx.sendto(self.proto.encode(msg), self.jetson)
        except OSError as e:          # 와이파이 끊김 등 — 노드는 살아 있어야 한다
            self.get_logger().warn(f"명령 송신 실패: {e}", throttle_duration_sec=5.0)

    def ping(self):
        self._send({"seq": self.seq, "cmd": "ping"})

    def tick(self):
        now = time.monotonic()
        for data in _drain(self.rx_state):
            m = self.proto.decode(data)
            if m and m.get("state") in ("idle", "turning", "acting"):
                self.jetson_state, self.state_t = m["state"], now
                if type(m.get("seq")) is int:
                    self.jetson_seq = m["seq"]
        online = now - self.state_t <= STATE_TIMEOUT_S
        self.pub_state.publish(String(data=self.jetson_state if online else "offline"))

        for data in _drain(self.rx_bridge):
            m = self.proto.decode(data)
            if m is None:
                continue
            self.bridge_t = now
            b = m.get("bearing_deg")
            if isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(b):   # NaN 이면 int(round()) 가 죽는다
                self.bearing, self.bearing_t = float(m["bearing_deg"]), now
            label = m.get("label") if m.get("valid") else None
            part = self.dwell.update(now, label if label in self.proto.PART_TO_SEGMENT else None)
            if part is not None:
                self.on_confirm(part, now, online)
        fresh = now - self.bridge_t <= 0.5
        self.pub_progress.publish(Float32(data=float(self.dwell.progress(now)) if fresh else 0.0))

    def on_confirm(self, part, now, online):
        if not online or self.jetson_state != "idle":
            self.get_logger().info(f"{part} 확정 — Jetson {self.jetson_state if online else 'offline'} 이라 무시")
            return
        # 확정 순간 한 프레임만 태그를 놓쳐도 회전하도록 dwell 창 안의 마지막 유효 bearing 을 쓴다.
        # 평균은 내지 않는다(뒤쪽 ±180 래핑에서 평균이 0 근처로 무너진다).
        bearing = int(round(self.bearing)) if now - self.bearing_t <= self.dwell_time else None
        self.seq = next_seq(self.seq, time.time(), self.jetson_seq)
        self._send({"seq": self.seq, "cmd": "act", "part": part, "bearing_deg": bearing})
        ev = {"t": time.time(), "part": part, "bearing_deg": bearing,
              "segments": self.proto.plan(part, bearing), "seq": self.seq}
        self.pub_event.publish(String(data=json.dumps(ev)))
        self.get_logger().info(f"act {ev}")


def main():
    rclpy.init()
    node = G1Interaction()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
