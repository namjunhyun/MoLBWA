#!/usr/bin/env python3
"""Pi 씬 영상 중계 — WiFi 로 한 번만 받아 PC 안에서 나눠 준다.

DDS 는 원격 구독자마다 따로 보내므로, 뷰어·태그 브리지·캘리브 도우미·SLAM 이
/camera/left/compressed 를 각각 구독하면 WiFi 로 같은 영상이 4번 간다(왼쪽이 0.4~0.5s 밀림).
이 노드만 Pi 토픽을 구독하고 /pc/camera/{left,right}/compressed 로 다시 내보낸다.
헤더 스탬프는 그대로 둔다(SLAM 좌우/IMU 동기화용).

    python3 -u tools/cam_relay.py            # 기본: 수신 RELIABLE depth 1
    python3 -u tools/cam_relay.py --best-effort
"""
import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage


class CamRelay(Node):
    def __init__(self, sides, src_prefix, dst_prefix, best_effort):
        super().__init__("cam_relay")
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE)
        pub_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.counts = {s: 0 for s in sides}
        self.lag = {s: [] for s in sides}
        self._subs = []
        for s in sides:
            src = f"{src_prefix}/camera/{s}/compressed"
            dst = f"{dst_prefix}/camera/{s}/compressed"
            pub = self.create_publisher(CompressedImage, dst, pub_qos)
            self._subs.append(self.create_subscription(
                CompressedImage, src, lambda m, s=s, p=pub: self._on(m, s, p), sub_qos))
            self.get_logger().info(f"중계: {src} -> {dst}")
        self.create_timer(5.0, self._report)

    def _on(self, msg, side, pub):
        pub.publish(msg)
        self.counts[side] += 1
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.lag[side].append(time.time() - st)

    def _report(self):
        parts = []
        for s, n in self.counts.items():
            lg = self.lag[s]
            avg = sum(lg) / len(lg) if lg else float("nan")
            parts.append(f"{s} {n / 5.0:.1f}Hz 지연 {avg * 1000:.0f}ms")
            self.counts[s] = 0
            self.lag[s] = []
        self.get_logger().info(" | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", default="left,right")
    ap.add_argument("--src-prefix", default="")
    ap.add_argument("--dst-prefix", default="/pc")
    ap.add_argument("--best-effort", action="store_true")
    a = ap.parse_args()
    rclpy.init()
    node = CamRelay(a.sides.split(","), a.src_prefix, a.dst_prefix, a.best_effort)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
