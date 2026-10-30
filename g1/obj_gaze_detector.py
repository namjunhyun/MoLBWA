"""시선이 향한 물체(화분·꽃병)를 YOLO 박스로 잡아 대화 kind 로 매핑한다.
pick_part(g1_gaze_bridge)와 같은 규약: dets 원소 = [cls, conf, x1, y1, x2, y2]."""
from __future__ import annotations

import json
import os
import struct
import subprocess

import numpy as np

from g1_gaze_bridge import object_bearing_deg

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# YOLO(COCO) 라벨 -> 대화 kind (spec §3.3). 물체 대화는 화분만 남김(2026-10-30 — 물병·가족사진·휴대폰 제외)
OBJ_TO_KIND = {"potted plant": "plant", "vase": "plant"}


def pick_object(dets: list, uv: tuple[float, float] | None) -> tuple[str, list[float]] | None:
    """uv 를 포함하고 OBJ_TO_KIND 에 있는 박스 중 conf 최고를 (kind, box_xyxy)로. 없으면 None."""
    if uv is None:
        return None
    u, v = uv
    hits = [d for d in dets if d[0] in OBJ_TO_KIND and d[2] <= u <= d[4] and d[3] <= v <= d[5]]
    if not hits:
        return None
    d = max(hits, key=lambda d: d[1])
    return OBJ_TO_KIND[d[0]], list(d[2:6])


def object_bearing(box_xyxy, K: np.ndarray, dist_m: float, T_hc_torso: np.ndarray) -> int:
    """박스 중심을 역투영한 torso 방위(도, 정수, +왼쪽). object_bearing_deg 재사용이라 T_hc_torso 인자가 추가됨."""
    uv = ((box_xyxy[0] + box_xyxy[2]) / 2, (box_xyxy[1] + box_xyxy[3]) / 2)
    return int(round(object_bearing_deg(T_hc_torso, uv, K, dist_m)))


class ObjectDetector:
    """gaze_hri/yolo_worker.py 를 자식 프로세스로 돌리는 래퍼. start() -> feed(jpeg) -> dets -> stop()."""

    def __init__(self, python: str, model: str = "yolo11l.pt", conf: float = 0.3, classes=None):
        self.cmd = [os.path.expanduser(python),
                    os.path.join(REPO, "ros2_ws", "src", "gaze_hri", "gaze_hri", "yolo_worker.py"),
                    os.path.expanduser(model), str(conf), ",".join(classes or OBJ_TO_KIND)]
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self._read()                      # {"ready": true}

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("yolo_worker 가 종료됨 (파이썬/모델 경로 확인)")
        return json.loads(line)

    def feed(self, jpeg: bytes) -> list:
        self.proc.stdin.write(struct.pack("<I", len(jpeg)) + jpeg)
        return self._read().get("dets", [])

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
