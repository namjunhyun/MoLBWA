"""브리지가 낸 토픽을 N 초 받아 본다: 초당 쌍 수, 좌우 스탬프가 같은지, 영상 크기.
    python3 check_topics.py /pc/camera 8
"""
import sys
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

prefix = sys.argv[1] if len(sys.argv) > 1 else "/pc/camera"
sec = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
rclpy.init()
n = rclpy.create_node("uvc_stereo_check")
got = {"left": {}, "right": {}}
qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
for s in got:
    n.create_subscription(CompressedImage, f"{prefix}/{s}/compressed",
                          lambda m, s=s: got[s].__setitem__((m.header.stamp.sec, m.header.stamp.nanosec), len(m.data)), qos)
t_end = time.time() + sec
while time.time() < t_end:
    rclpy.spin_once(n, timeout_sec=0.1)
L, R = set(got["left"]), set(got["right"])
both = L & R
print(f"{sec:.0f}초: 왼 {len(L)} · 오 {len(R)} · 같은 스탬프 쌍 {len(both)} ({len(both) / sec:.1f}/s)")
if both:
    ts = sorted(s + ns * 1e-9 for s, ns in both)
    print(f"스탬프와 지금 시각 차(마지막) {time.time() - ts[-1]:.3f} s · 쌍 간격 중앙 {1000 * sorted(b - a for a, b in zip(ts, ts[1:]))[len(ts) // 2]:.1f} ms")
n.destroy_node(); rclpy.shutdown()
sys.exit(0 if len(both) > sec * 20 else 1)
