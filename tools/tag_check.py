#!/usr/bin/env python3
"""AprilTag 번들 실기 점검 — 씬 카메라(/camera/left/compressed)로 N초 동안.

    python3 tools/tag_check.py --secs 8 --snap /tmp/tag.png

출력:
  * 태그별 검출률, 화면상 한 변 크기(px), decision margin
  * 번들 PnP 성공률, 재투영 오차, 거부 사유 (arm/anchor.TagBundleDetector 그대로 사용)
  * 헤드캠 위치 (팔 기준 base_link) — 배치 B 라면 x 음수(팔 뒤), z 는 눈높이 근처여야 정상
  * 프레임 간 자세 흔들림(표준편차) — 머리를 가만히 두고 재면 검출 잡음
"""
import argparse
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arm"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from anchor import TagBundleDetector  # noqa: E402
from ocams_calib import RECTIFIED_K  # noqa: E402


class Grab:
    def __init__(self, topic):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("tag_check")
        self.lock = threading.Lock()
        self.frame, self.seq = None, 0
        self.node.create_subscription(
            CompressedImage, topic, self._cb, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self):
        try:
            self.rclpy.spin(self.node)
        except Exception:
            pass

    def _cb(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            with self.lock:
                self.frame, self.seq = img, self.seq + 1

    def get(self):
        with self.lock:
            return self.frame, self.seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--topic", default="/camera/left/compressed")
    ap.add_argument("--snap", default="")
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="  [anchor] %(message)s")
    cfg = yaml.safe_load(open(os.path.join(HERE, "..", "arm", "config.yaml")))
    K = np.asarray(RECTIFIED_K, float)
    det = TagBundleDetector(cfg, K)
    want = sorted(det.obj_pts)
    g = Grab(a.topic)

    n = 0
    seen = {i: [] for i in want}            # 태그별 한 변 픽셀
    margins = {i: [] for i in want}
    poses, reproj = [], []
    last_seq, snap_img, t_end = 0, None, time.time() + a.secs
    while time.time() < t_end:
        img, seq = g.get()
        if img is None or seq == last_seq:
            time.sleep(0.005)
            continue
        last_seq = seq
        n += 1
        raw = [d for d in det.det.detect(img) if d.tag_id in det.obj_pts]
        for d in raw:
            c = np.asarray(d.corners)
            seen[d.tag_id].append(float(np.mean(np.linalg.norm(c - np.roll(c, 1, 0), axis=1))))
            margins[d.tag_id].append(float(d.decision_margin))
        T = det.detect(img)
        if T is not None:
            poses.append(np.linalg.inv(T))    # T_ab_hc
            reproj.append(det.last_reproj_px)
        if snap_img is None or len(raw) >= len([d for d in snap_img[1]]):
            snap_img = (img, raw)

    print(f"\n프레임 {n}개 ({a.secs:.0f}초)")
    if n == 0:
        print("  영상이 안 들어온다 — Pi 카메라 스택 확인")
        return 1
    for i in want:
        s = seen[i]
        if s:
            print(f"  태그 {i}: 검출 {len(s) / n * 100:5.1f}%  한 변 {np.median(s):5.1f}px  "
                  f"margin {np.median(margins[i]):5.1f}")
        else:
            print(f"  태그 {i}: 검출 0% — 화면 밖이거나 너무 작다/흐리다")
    print(f"  번들 자세 성공 {len(poses) / n * 100:.1f}%"
          + (f", 재투영 중앙값 {np.median(reproj):.2f}px (최대 {max(reproj):.2f})" if reproj else ""))
    if poses:
        P = np.array([T[:3, 3] for T in poses])
        m, sd = P.mean(0), P.std(0)
        print(f"  헤드캠 위치(팔 기준): x={m[0]:+.3f} y={m[1]:+.3f} z={m[2]:+.3f} m  "
              f"흔들림 σ=({sd[0] * 1000:.1f}, {sd[1] * 1000:.1f}, {sd[2] * 1000:.1f}) mm")
        fwd = np.mean([T[:3, 2] for T in poses], axis=0)
        print(f"  카메라가 보는 방향(팔 기준 단위벡터): {np.round(fwd, 2).tolist()}")
    if a.snap and snap_img is not None:
        img, raw = snap_img
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for d in raw:
            c = np.asarray(d.corners, int)
            cv2.polylines(vis, [c.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            cv2.putText(vis, str(d.tag_id), tuple(c[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(a.snap, vis)
        print(f"  스냅샷: {a.snap}")
    g.rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
