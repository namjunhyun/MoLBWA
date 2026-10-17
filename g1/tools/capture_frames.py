#!/usr/bin/env python3
"""씬 카메라 CompressedImage 를 N 프레임마다 파일로 저장한다 (G1 부위 YOLO 학습 데이터용).

    python3 g1/tools/capture_frames.py --scene-topic /pc/camera/left/compressed --out DIR --every 10

압축 바이트를 그대로 쓴다(재인코딩 없음) — g1_gaze_bridge 가 YOLO 에 넣는 것과 같은 원본 영상이다.
Ctrl-C 로 끝내면 저장 수를 찍는다. 기존 파일은 덮어쓰지 않는다(이름에 시작 시각을 넣는다).
"""
import argparse
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-topic", default="/pc/camera/left/compressed")
    ap.add_argument("--out", required=True, help="저장 폴더 (없으면 만든다)")
    ap.add_argument("--every", type=int, default=10, help="N 프레임마다 1장")
    args = ap.parse_args()
    if args.every < 1:
        ap.error("--every 는 1 이상")
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    n_seen = n_saved = 0

    def on_image(msg):
        nonlocal n_seen, n_saved
        n_seen += 1
        if (n_seen - 1) % args.every:
            return
        ext = "png" if "png" in msg.format.lower() else "jpg"
        path = os.path.join(args.out, f"{stamp}_{n_seen:06d}.{ext}")
        with open(path, "wb") as f:
            f.write(bytes(msg.data))
        n_saved += 1

    rclpy.init()
    node = Node("g1_capture_frames")
    # BEST_EFFORT + depth 1: gaze_on_scene / run_demo 와 같은 정책
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(CompressedImage, args.scene_topic, on_image, qos)
    print(f"{args.scene_topic} → {args.out} ({args.every} 프레임마다). Ctrl-C 로 종료")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"수신 {n_seen} 프레임, 저장 {n_saved} 장 → {args.out}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
