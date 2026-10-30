#!/usr/bin/env python3
"""씬 카메라 영상 품질 진단 — SLAM 추적 실패 원인 가리기 (HANDOFF_2026-09-23 §2).

2026-09-22 측정: ORB 485개(목표 1200), 선명도 20.1(정상 100+), 과노출 24%, 빈 구역 5/16.
선명도가 40프레임 내내 20.0~20.2 로 안 변했다 = 모션 블러가 아니라 고정 원인(초점/압축/노출).

    python3 tools/scene_quality.py                          # /camera/left/compressed, 5초
    python3 tools/scene_quality.py --topic /camera/left     # 원본(비압축) — 압축 탓인지 비교
    python3 tools/scene_quality.py --both                   # 압축/원본 나란히 -> JPEG 품질 영향

해석:
  * 선명도가 원본은 높은데 압축본만 낮다 -> Pi 의 JPEG 품질(image_transport compressed) 문제
  * 둘 다 낮고 프레임마다 거의 같다     -> 렌즈 초점
  * 과노출 > 5%                          -> 자동노출이 조명에 속음 (노출/게인 고정 검토)
  * 빈 구역(특징점 0) 많음               -> 텍스처 부족한 장면/흐림
"""
import argparse
import sys
import threading
import time

import cv2
import numpy as np


def metrics(img, orb):
    kps = orb.detect(img, None)
    sharp = float(cv2.Laplacian(img, cv2.CV_64F).var())
    over = float((img >= 250).mean() * 100)
    under = float((img <= 5).mean() * 100)
    h, w = img.shape
    empty = 0
    pts = np.array([k.pt for k in kps]) if kps else np.zeros((0, 2))
    for i in range(4):
        for j in range(4):
            x0, x1, y0, y1 = w * j / 4, w * (j + 1) / 4, h * i / 4, h * (i + 1) / 4
            if not ((pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)).any():
                empty += 1
    return len(kps), sharp, over, under, empty


def collect(topic, secs):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    compressed = topic.endswith("compressed")
    if compressed:
        from sensor_msgs.msg import CompressedImage as Msg
    else:
        from sensor_msgs.msg import Image as Msg
    if not rclpy.ok():
        rclpy.init()
    node = Node("scene_quality_" + str(abs(hash(topic)) % 10000))
    frames, lock = [], threading.Lock()

    def cb(m):
        if compressed:
            img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width]
        if img is not None:
            with lock:
                frames.append(img.copy())
    node.create_subscription(Msg, topic, cb, QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
    t_end = time.time() + secs
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    return frames


def report(name, frames):
    if not frames:
        print(f"{name}: 프레임 없음 — 토픽/Pi 확인")
        return
    orb = cv2.ORB_create(1200)
    M = np.array([metrics(f, orb) for f in frames[::2]])
    print(f"{name}: {len(frames)}프레임")
    print(f"  ORB 특징점  {np.median(M[:, 0]):6.0f}   (목표 1200, 2026-09-22: 485)")
    print(f"  선명도      {np.median(M[:, 1]):6.1f}   범위 {M[:, 1].min():.1f}~{M[:, 1].max():.1f} "
          "(100+ 선명, 50 미만 흐림, 09-22: 20.1 고정)")
    print(f"  과노출      {np.median(M[:, 2]):5.1f}%   (5% 미만 정상, 09-22: 24%)")
    print(f"  과소노출    {np.median(M[:, 3]):5.1f}%")
    print(f"  빈 구역     {np.median(M[:, 4]):4.0f}/16  (0~1 정상, 09-22: 5)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/camera/left/compressed")
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    topics = ["/camera/left/compressed", "/camera/left"] if a.both else [a.topic]
    for t in topics:
        report(t, collect(t, a.secs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
