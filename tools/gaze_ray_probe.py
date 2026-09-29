#!/usr/bin/env python3
"""팔 없이 눈+태그 체인만 검증 — 알고 있는 3D 지점(태그 중심)을 응시해서 오차를 본다.

사용법: python3 tools/gaze_ray_probe.py --known 0.540 0.130 0.319
 (기본값은 태그 0 중심, arm/config.yaml 값)

/gaze/fixation 을 받아서 origin->point 광선을 known 점의 높이(z)와 교차시키고,
그 x,y 를 known x,y 와 비교한다. calibrate_world_to_base.py 가 캘리브 자세마다
하는 것과 똑같은 계산을 팔 없이 한 점에 대해서만 한다.
"""
import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node

from gaze_hri_msgs.msg import Fixation


class Probe(Node):
    def __init__(self, known):
        super().__init__("gaze_ray_probe")
        self.known = np.array(known, dtype=float)
        self.create_subscription(Fixation, "/gaze/fixation", self.on_fix, 10)
        self.n = 0

    def on_fix(self, msg: Fixation):
        point = np.array([msg.point.x, msg.point.y, msg.point.z])
        if not msg.has_origin:
            print("머리 위치 없음 — 태그 추적 확인할 것")
            return
        origin = np.array([msg.origin.x, msg.origin.y, msg.origin.z])
        direction = point - origin
        if abs(direction[2]) < 1e-9:
            print("광선이 수평 — 스킵")
            return
        t = (self.known[2] - origin[2]) / direction[2]
        p = origin + t * direction
        err = p - self.known
        self.n += 1
        print(f"[{self.n}] 계산={p.round(3)}  실제={self.known.round(3)}  "
              f"오차=({err[0]*100:+.1f}, {err[1]*100:+.1f})cm  "
              f"거리오차={np.linalg.norm(err[:2])*100:.1f}cm  신뢰도={msg.confidence:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=float, nargs=3, default=[0.540, 0.130, 0.319],
                     help="응시할 알려진 3D 지점 (armbase 기준, m). 기본값 = 태그0 중심")
    a = ap.parse_args()
    rclpy.init()
    node = Probe(a.known)
    print(f"알려진 지점 {a.known} 을 응시하세요 (예: 태그0 중심). Ctrl+C 로 종료.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
