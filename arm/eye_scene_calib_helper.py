#!/usr/bin/env python3
"""docs/12 눈-씬카메라 R+t 캘리브용 — 스테레오 깊이 대신 태그+로봇팔로 3D 캘리브 점을 만든다.

배경: docs/12는 여러 거리의 알려진 3D점(X_i, 씬카메라 기준)이 필요했는데, 원래 계획은
oCamS 스테레오로 깊이를 재는 거였다. 근데 2026-08-19에 그 스테레오 깊이가 검증 실패했다
(평면 위 5점인데 깊이가 0.52~1.38m로 제각각). 이 스크립트는 스테레오 대신, 이미 정확하게
동작하는 태그 추적(재투영 0.5px) + 팔 FK로 X_i를 만든다:

    X_i(씬카메라 기준) = T_hc_ab @ [p_B, 1]     (p_B = 팔 FK로 아는 그리퍼 위치, armbase 기준)

팔을 여러 자세로 보내면서(반경/높이를 다양하게) 이 스크립트를 계속 띄워두면, 매 프레임
현재 그리퍼 위치를 씬카메라 좌표계로 변환해서 UDP로 쏜다. gaze_on_scene.py의 새 캘리브
모드(M)가 이걸 받아서 그 순간의 눈 방향(d_i)과 짝지어 기록한다.

    python3 arm/eye_scene_calib_helper.py

송신: UDP 127.0.0.1:55058, JSON {"point": [x,y,z], "valid": bool} (씬카메라 기준, m)
"""
import argparse
import json
import logging
import math
import os
import socket
import sys
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ros2_ws", "src", "gaze_hri"))
from gaze_hri.kinematics import ArmGeometry, forward_kinematics  # noqa: E402

from anchor import TagBundleDetector, TagDirectAnchor  # noqa: E402
from run_demo import load_intrinsics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class Helper(Node):
    def __init__(self, udp_port, scene_topic, min_consecutive, max_jump_m):
        super().__init__("eye_scene_calib_helper")
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
        K = load_intrinsics()
        self.tags = TagBundleDetector(cfg, K)
        self.direct = TagDirectAnchor(min_consecutive=min_consecutive, max_jump_m=max_jump_m)
        self.geo = ArmGeometry()  # base_height 등은 gaze_hri.yaml과 별개, 기본값이 2026-09-27 실측값

        self.joint_pos = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = ("127.0.0.1", udp_port)

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CompressedImage, scene_topic, self.on_scene, qos)
        self.create_subscription(JointState, "/joint_states", self.on_joint, 10)

        self.n = self.n_valid = 0
        self.create_timer(2.0, self.report)
        log.info("씬 구독 %s, /joint_states 구독, 송신 %s:%d", scene_topic, *self.addr)

    def on_joint(self, msg: JointState):
        if len(msg.position) >= 4:
            self.joint_pos = list(msg.position[:4])

    def on_scene(self, msg: CompressedImage):
        self.n += 1
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if frame is None:
            return
        if frame.ndim == 2:
            frame_gray = frame
        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        T_hc_ab = self.tags.detect(frame_gray)
        if T_hc_ab is not None:
            self.direct.update_tag(T_hc_ab)
        else:
            self.direct.miss()

        T = self.direct.T_headcam_to_armbase()
        if T is None or self.joint_pos is None:
            self.sock.sendto(json.dumps({"point": [0.0, 0.0, 0.0], "valid": False}).encode(), self.addr)
            return

        # 2026-09-27: 휴식 자세는 shoulder_lift 가 3.26rad(187°)로 읽혀 모델 범위(±100°) 밖이다.
        # 거기서 FK 한 점으로 'o' 보정을 걸었다가 시선이 받침대 뒤로 날아갔다 -> 범위 밖이면 무효.
        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
        bad = [n for n, q in zip(names, self.joint_pos)
               if not (self.geo.limits[n][0] - 0.05 <= q <= self.geo.limits[n][1] + 0.05)]
        if bad:
            self.sock.sendto(json.dumps({"point": [0.0, 0.0, 0.0], "valid": False}).encode(), self.addr)
            self.n_out_of_range = getattr(self, "n_out_of_range", 0) + 1
            return
        p_B = forward_kinematics(self.joint_pos, self.geo)  # armbase 기준 그리퍼(TCP) 위치
        p_B_h = np.array([p_B[0], p_B[1], p_B[2], 1.0])
        p_hc = T @ p_B_h  # 씬카메라(헤드캠) 기준
        payload = {"point": [float(p_hc[0]), float(p_hc[1]), float(p_hc[2])], "valid": True}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)
        self.n_valid += 1

    def report(self):
        log.info("프레임 %d, 유효 %d (%.0f%%), 관절 범위 밖(휴식 자세 등) %d", self.n, self.n_valid,
                  100.0 * self.n_valid / max(self.n, 1), getattr(self, "n_out_of_range", 0))
        self.n_out_of_range = 0
        self.n = self.n_valid = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--udp-port", type=int, default=55058)
    ap.add_argument("--scene-topic", default="/camera/left/compressed")
    ap.add_argument("--min-consecutive", type=int, default=3)
    ap.add_argument("--max-jump-m", type=float, default=0.05)
    a = ap.parse_args()

    rclpy.init()
    node = Helper(a.udp_port, a.scene_topic, a.min_consecutive, a.max_jump_m)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
