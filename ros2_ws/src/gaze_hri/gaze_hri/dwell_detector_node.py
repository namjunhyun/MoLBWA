#!/usr/bin/env python3
"""
[노드 2] dwell_detector  —  "일정 시간 한 곳을 보고 있는가?"

/gaze/point_raw 를 구독해서, 시선이 충분히 오래 + 충분히 좁은 범위에 머무르면
확정 응시점(Fixation)을 발행합니다.

이 노드가 하는 일은 사실 시선 인터페이스의 심장입니다.
"눈으로 클릭한다"는 개념 자체가 여기서 구현되기 때문입니다.

알고리즘: I-DT (Dispersion Threshold Identification)
  - 최근 dwell_time초 동안의 점들을 버퍼에 유지
  - 그 점들의 무게중심으로부터 최대 거리 < dispersion_radius 이면 "고정"
  - 유효 샘플 비율이 min_valid_ratio 이상일 때만 인정 (깜빡임 견디기)
  - 한 번 발행하면 cooldown 동안, 그리고 시선이 exit_radius 밖으로
    나가기 전까지는 재발행 안 함 (연타 방지)

발행:
  /gaze/fixation         gaze_hri_msgs/Fixation
  /gaze/dwell_progress   std_msgs/Float32   0~1, 사용자 피드백 UI용
  /gaze/markers          visualization_msgs/MarkerArray  RViz 시각화
"""

import math
from collections import deque

import numpy as np
import rclpy
from gaze_hri_msgs.msg import Fixation
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32
from visualization_msgs.msg import Marker, MarkerArray


class DwellDetector(Node):

    def __init__(self):
        super().__init__("dwell_detector")

        # --- 튜닝 파라미터 ---
        # dwell_time: 짧으면 오작동(미다스의 손 문제), 길면 답답함. 1.0~1.5초가 적당.
        self.declare_parameter("dwell_time", 1.2)
        # dispersion_radius: 3cm. 시선 추적 정확도(~1도, 60cm에서 약 1cm)를 고려한 값.
        self.declare_parameter("dispersion_radius", 0.030)
        self.declare_parameter("min_valid_ratio", 0.7)
        self.declare_parameter("cooldown", 1.5)
        self.declare_parameter("exit_radius", 0.10)
        self.declare_parameter("max_sample_age", 0.35)   # 이보다 오래 끊기면 버퍼 리셋

        self.dwell_time = float(self.get_parameter("dwell_time").value)
        self.dispersion_radius = float(self.get_parameter("dispersion_radius").value)
        self.min_valid_ratio = float(self.get_parameter("min_valid_ratio").value)
        self.cooldown = float(self.get_parameter("cooldown").value)
        self.exit_radius = float(self.get_parameter("exit_radius").value)
        self.max_sample_age = float(self.get_parameter("max_sample_age").value)

        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PointStamped, "/gaze/point_raw", self.on_point, qos)
        self.create_subscription(Bool, "/gaze/valid", self.on_valid, qos)

        self.pub_fix = self.create_publisher(Fixation, "/gaze/fixation", 10)
        self.pub_progress = self.create_publisher(Float32, "/gaze/dwell_progress", qos)
        self.pub_marker = self.create_publisher(MarkerArray, "/gaze/markers", 10)

        self.buf = deque()            # (t, np.array([x,y,z]))
        self.valid_buf = deque()      # (t, bool)
        self.last_fix_time = -1e9
        self.last_fix_point = None
        self.left_since_fix = True
        self.frame_id = "map"
        self.last_stamp = None

        self.create_timer(0.05, self.evaluate)
        self.get_logger().info(
            f"dwell_detector 시작 (dwell={self.dwell_time}s, "
            f"반경={self.dispersion_radius * 100:.1f}cm)"
        )

    # ------------------------------------------------------------------
    def on_valid(self, msg: Bool):
        t = self.now()
        self.valid_buf.append((t, msg.data))
        self._trim(self.valid_buf, t)

    def on_point(self, msg: PointStamped):
        t = self.now()
        self.frame_id = msg.header.frame_id or "map"
        self.last_stamp = msg.header.stamp
        p = np.array([msg.point.x, msg.point.y, msg.point.z])

        # 데이터가 오래 끊겼다면 이전 버퍼는 신뢰할 수 없으니 리셋
        if self.buf and (t - self.buf[-1][0]) > self.max_sample_age:
            self.buf.clear()

        self.buf.append((t, p))
        self._trim(self.buf, t)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _trim(self, buf, t):
        while buf and (t - buf[0][0]) > self.dwell_time:
            buf.popleft()

    # ------------------------------------------------------------------
    def evaluate(self):
        t = self.now()
        self._trim(self.buf, t)
        self._trim(self.valid_buf, t)

        if len(self.buf) < 5:
            self.pub_progress.publish(Float32(data=0.0))
            return

        pts = np.array([p for _, p in self.buf])
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        max_dist = float(dists.max())
        rms = float(np.sqrt((dists ** 2).mean()))

        # 이전 fixation 지점에서 충분히 벗어났는지 추적
        if self.last_fix_point is not None:
            if np.linalg.norm(centroid - self.last_fix_point) > self.exit_radius:
                self.left_since_fix = True

        # 진행률 = (안정적으로 머문 시간) / (필요 시간)
        span = self.buf[-1][0] - self.buf[0][0]
        stable = max_dist < self.dispersion_radius
        progress = min(1.0, span / self.dwell_time) if stable else 0.0
        self.pub_progress.publish(Float32(data=float(progress)))
        self._publish_marker(centroid, progress, stable)

        if not stable:
            return
        if span < self.dwell_time * 0.95:
            return
        if t - self.last_fix_time < self.cooldown:
            return
        if not self.left_since_fix:
            return

        # 유효 샘플 비율 확인
        if self.valid_buf:
            ratio = sum(1 for _, v in self.valid_buf if v) / len(self.valid_buf)
        else:
            ratio = 1.0
        if ratio < self.min_valid_ratio:
            return

        # --- Fixation 확정 ---
        confidence = float(
            ratio * max(0.0, 1.0 - rms / self.dispersion_radius)
        )

        msg = Fixation()
        msg.header.stamp = self.last_stamp or self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.point.x, msg.point.y, msg.point.z = map(float, centroid)
        msg.duration = float(span)
        msg.dispersion = rms
        msg.confidence = confidence
        self.pub_fix.publish(msg)

        self.last_fix_time = t
        self.last_fix_point = centroid
        self.left_since_fix = False
        self.buf.clear()

        self.get_logger().info(
            f"응시 확정: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}) "
            f"산포 {rms * 1000:.1f}mm, 신뢰도 {confidence:.2f}"
        )

    # ------------------------------------------------------------------
    def _publish_marker(self, centroid, progress, stable):
        """RViz에 응시 커서를 띄웁니다. 발표 데모에서 효과가 큽니다."""
        arr = MarkerArray()

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "gaze_cursor"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, centroid)
        m.pose.orientation.w = 1.0
        d = self.dispersion_radius * 2
        m.scale.x = m.scale.y = m.scale.z = d
        # 진행률에 따라 파랑 -> 초록
        m.color.r = 0.1
        m.color.g = float(0.3 + 0.7 * progress)
        m.color.b = float(1.0 - 0.8 * progress)
        m.color.a = 0.55 if stable else 0.25
        arr.markers.append(m)

        self.pub_marker.publish(arr)


def main():
    rclpy.init()
    node = DwellDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
