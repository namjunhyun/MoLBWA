#!/usr/bin/env python3
"""Measure temporal stability of StereoSGBM depth from rectified ROS 2 images."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
import ocams_calib  # noqa: E402


def make_matcher():
    block_size = 7
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=block_size,
        P1=8 * block_size ** 2, P2=32 * block_size ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
    )


class StabilityProbe(Node):
    def __init__(self, args):
        super().__init__("ocams_depth_stability")
        self.args = args
        self.matcher = make_matcher()
        self.depths = []
        self.valid_ratios = []
        self.disparity_mads = []
        self.started = time.monotonic()
        left = Subscriber(self, Image, "/camera/left")
        right = Subscriber(self, Image, "/camera/right")
        self.sync = ApproximateTimeSynchronizer([left, right], queue_size=10, slop=0.03)
        self.sync.registerCallback(self.callback)

    def callback(self, left_msg, right_msg):
        left = np.frombuffer(left_msg.data, np.uint8).reshape(left_msg.height, left_msg.width)
        right = np.frombuffer(right_msg.data, np.uint8).reshape(right_msg.height, right_msg.width)
        disparity = self.matcher.compute(left, right).astype(np.float32) / 16.0
        h, w = disparity.shape
        cx = self.args.x if self.args.x is not None else w // 2
        cy = self.args.y if self.args.y is not None else h // 2
        radius = self.args.radius
        roi = disparity[max(0, cy-radius):min(h, cy+radius+1),
                        max(0, cx-radius):min(w, cx+radius+1)]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        self.valid_ratios.append(valid.size / roi.size)
        if valid.size:
            median = float(np.median(valid))
            mad = float(np.median(np.abs(valid - median)))
            self.disparity_mads.append(mad)
            self.depths.append(self.args.fx * self.args.baseline / median)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--baseline", type=float, default=ocams_calib.DEPTH_BASELINE_M)
    parser.add_argument("--fx", type=float, default=float(ocams_calib.RECTIFIED_K[0, 0]))
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--radius", type=int, default=10)
    args = parser.parse_args()

    rclpy.init()
    node = StabilityProbe(args)
    deadline = time.monotonic() + args.seconds
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    depths = np.asarray(node.depths)
    ratios = np.asarray(node.valid_ratios)
    mads = np.asarray(node.disparity_mads)
    if depths.size == 0:
        raise RuntimeError("No valid disparity samples received")
    print(f"frames={ratios.size} depth_frames={depths.size}")
    print(f"depth median={np.median(depths):.4f}m mean={depths.mean():.4f}m "
          f"std={depths.std():.4f}m min={depths.min():.4f}m max={depths.max():.4f}m")
    print(f"valid ratio median={np.median(ratios)*100:.1f}% "
          f"min={ratios.min()*100:.1f}%")
    print(f"disparity MAD median={np.median(mads):.3f}px max={mads.max():.3f}px")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
