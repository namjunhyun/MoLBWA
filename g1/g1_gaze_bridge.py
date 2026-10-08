#!/usr/bin/env python3
"""시선 픽셀 + G1 몸통 태그 + YOLO 부위 -> {t, label, bearing_deg, valid} (UDP 55057).

    gaze_on_scene.py --send-gaze-px ──UDP 55056──┐
    /pc/camera/left/compressed ──────────────────┴─> 이 스크립트 ──UDP 55057──> g1_interaction

label       : 시선 픽셀을 포함하는 YOLO 박스 (hand > face > torso, 손 박스가 몸통 박스 안이라)
bearing_deg : G1 torso 기준 헤드캠(=사용자 머리) 방위, + 왼쪽. 태그 번들이 안 보이면 null
valid       : 이번 프레임에 시선 픽셀이 있었는가

arm/gaze_tag_bridge.py 와 같은 구조: 순수 함수 + 얇은 루프. 무거운 의존(cv2, apriltag, rclpy)은 main 안에서.

실행:
    python3 -u g1/g1_gaze_bridge.py
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time

import numpy as np

from g1_protocol import PORT_BRIDGE, encode, nearest_bin  # noqa: F401  (nearest_bin 재노출)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
log = logging.getLogger("g1_gaze_bridge")

PART_PRIORITY = ("g1_hand", "g1_face", "g1_torso")


def pick_part(dets: list, uv: tuple[float, float] | None) -> str | None:
    """dets = [[cls, conf, x1, y1, x2, y2], ...] 중 uv 를 포함하는 박스의 부위. 우선순위 hand > face > torso."""
    if uv is None:
        return None
    u, v = uv
    hits = [d for d in dets if d[0] in PART_PRIORITY and d[2] <= u <= d[4] and d[3] <= v <= d[5]]
    if not hits:
        return None
    return min(hits, key=lambda d: (PART_PRIORITY.index(d[0]), -d[1]))[0]


def bearing_deg(T_hc_torso: np.ndarray) -> float:
    """T_hc_torso(p_hc = T @ p_torso) -> torso 기준 헤드캠 원점 방위 [deg], (-180, 180]. +x 정면, +y 왼쪽."""
    p = np.linalg.inv(T_hc_torso)[:3, 3]
    deg = math.degrees(math.atan2(p[1], p[0]))
    return 180.0 if deg == -180.0 else deg


class _Yolo:
    """gaze_hri/yolo_worker.py 를 자식 프로세스로 돌린다(동기: 프레임마다 보내고 결과를 기다림).
    프레임과 검출이 같은 순간이어야 시선 픽셀과 박스를 비교할 수 있다."""

    def __init__(self, ycfg):
        import subprocess
        worker = os.path.join(REPO, "ros2_ws", "src", "gaze_hri", "gaze_hri", "yolo_worker.py")
        cmd = [os.path.expanduser(ycfg["python"]), worker, os.path.expanduser(ycfg["model"]),
               str(ycfg["conf"]), ",".join(ycfg["classes"])]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self._read()                      # {"ready": true}

    def _read(self):
        import json
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("yolo_worker 가 종료됨 (모델 경로/ ~/yolo-env 확인)")
        return json.loads(line)

    def __call__(self, frame):
        import struct

        import cv2
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return []
        data = jpg.tobytes()
        self.proc.stdin.write(struct.pack("<I", len(data)) + data)
        return self._read().get("dets", [])

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--scene-topic", default="/pc/camera/left/compressed")
    ap.add_argument("--gaze-px-port", type=int, default=55056)
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=PORT_BRIDGE)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import socket

    import cv2
    import yaml
    sys.path.insert(0, os.path.join(REPO, "arm"))
    from anchor import TagBundleDetector
    from run_demo import GazeSource, load_intrinsics

    cfg = yaml.safe_load(open(args.config))
    tags = TagBundleDetector(cfg, load_intrinsics())
    yolo = _Yolo(cfg["yolo"])
    src = GazeSource(use_ros=True, tag_direct=True, scene_topic=args.scene_topic,
                     gaze_px_port=args.gaze_px_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (args.udp_host, args.udp_port)
    log.info("송신 %s:%d", *dst)

    n = n_label = n_bear = 0
    next_report = time.time() + 2.0
    try:
        while src.ok():
            frame, uv, _ = src.frame()
            if frame is None:
                time.sleep(0.005)
                continue
            T = tags.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            label = pick_part(yolo(frame), uv)
            bearing = None if T is None else bearing_deg(T)
            # 무효여도 매 프레임 보낸다 — dwell 이 '안 봄'을 알아야 리셋된다
            sock.sendto(encode({"t": time.time(), "label": label, "bearing_deg": bearing,
                                "valid": uv is not None}), dst)
            n += 1
            n_label += label is not None
            n_bear += bearing is not None
            if time.time() >= next_report:
                log.info("프레임 %d, 부위 %d%%, 방위 %d%% (최근 %s / %s)", n, 100 * n_label // n,
                         100 * n_bear // n, label, "-" if bearing is None else f"{bearing:+.0f}°")
                n = n_label = n_bear = 0
                next_report = time.time() + 2.0
    except KeyboardInterrupt:
        pass
    finally:
        sock.sendto(encode({"t": time.time(), "label": None, "bearing_deg": None, "valid": False}), dst)
        sock.close()
        yolo.close()
        src.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
