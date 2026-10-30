#!/usr/bin/env python3
"""시선 픽셀 + G1 몸통 태그 + YOLO 부위 -> {t, label, bearing_deg, valid} (UDP 55057).

    gaze_on_scene.py --send-gaze-px ──UDP 55056──┐
    /pc/camera/left/compressed ──────────────────┴─> 이 스크립트 ──UDP 55057──> g1_interaction

label       : 시선 픽셀을 포함하는 YOLO 박스 (hand > face > torso, 손 박스가 몸통 박스 안이라)
bearing_deg : G1 torso 기준 헤드캠(=사용자 머리) 방위, + 왼쪽. 태그 번들이 안 보이면 null
obj_bearing_deg : 착용자가 보는 점의 G1 torso 기준 방위(같이 보기용). 거리는 config attend.object_dist_m(기본 1.5 m) 가정
obj_kind    : 시선이 향한 물체의 대화 kind(plant/bottle/phone/photo) — 물체 YOLO + pick_object. 없으면 null
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

from g1_protocol import PORT_BRIDGE, encode

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


def object_bearing_deg(T_hc_torso: np.ndarray, uv: tuple[float, float], K: np.ndarray, dist_m: float) -> float:
    """착용자가 보는 점의 G1 torso 기준 방위 [deg] (+ 왼쪽) — '같이 보기'에서 지원이가 돌아볼 방향.
    헤드캠 광선(K⁻¹[u,v,1])을 dist_m 만큼 따라간 점을 torso 좌표로 옮긴다. 거리는 스테레오 깊이가 있으면 그 값,
    없으면 설정값(대충 1~2 m) — 지원이와 착용자가 다른 자리라 거리를 모르면 가까운 물체일수록 방향이 틀어진다."""
    ray = np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    p_cam = ray / np.linalg.norm(ray) * dist_m
    p_t = np.linalg.inv(T_hc_torso) @ np.append(p_cam, 1.0)
    deg = math.degrees(math.atan2(p_t[1], p_t[0]))
    return 180.0 if deg == -180.0 else deg


PART_COLOR = {"g1_hand": (0, 220, 0), "g1_face": (0, 200, 255), "g1_torso": (170, 170, 170)}   # BGR
OBJ_COLOR = (0, 140, 255)
GAZE_COLOR = (255, 0, 255)
PART_NAME = {"g1_hand": "HAND", "g1_face": "FACE", "g1_torso": "TORSO"}


def draw_overlay(frame, part_dets, obj_dets, uv, label, obj_kind, bearing):
    """확인 창(--show)용 그림: 부위·물체 박스, 시선 점, 위쪽에 판정·태그 방위. 원본은 건드리지 않고 복사본에 그린다.
    cv2 기본 글꼴은 한글을 못 그려서 화면 글자는 영어."""
    import cv2
    img = frame.copy()
    for cls, conf, x1, y1, x2, y2 in part_dets:
        c = PART_COLOR.get(cls, (255, 255, 255))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(img, f"{PART_NAME.get(cls, cls)} {conf:.2f}", (int(x1), max(52, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    for cls, conf, x1, y1, x2, y2 in obj_dets:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), OBJ_COLOR, 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (int(x1), max(52, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, OBJ_COLOR, 1)
    if uv is not None:
        p = (int(uv[0]), int(uv[1]))
        cv2.circle(img, p, 10, GAZE_COLOR, 2)
        cv2.circle(img, p, 3, GAZE_COLOR, -1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (30, 30, 30), -1)
    look = PART_NAME.get(label) or (f"OBJ {obj_kind}" if obj_kind else "-")
    tag = "TAG none" if bearing is None else f"TAG {bearing:+.0f}deg"
    cv2.putText(img, f"LOOK: {look}   {tag}   {'GAZE' if uv is not None else 'no gaze'}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


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
    ap.add_argument("--obj-model", default="",
                    help="물체 YOLO 모델(미지정 시 config obj_yolo.model 또는 ~/yolo-env/yolo11l.pt)")
    ap.add_argument("--no-obj", action="store_true", help="물체 검출 끄기(부위만) — yolo11l.pt 가 아직 없을 때")
    ap.add_argument("--show", action="store_true",
                    help="확인 창: 부위·물체 박스 + 시선 점 + 판정. 물체 YOLO를 시선 없을 때도 돌려 fps 가 떨어진다")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import socket

    import cv2
    import yaml
    sys.path.insert(0, os.path.join(REPO, "arm"))
    from anchor import TagBundleDetector
    from obj_gaze_detector import OBJ_TO_KIND, ObjectDetector, pick_object   # main 안에서 import(순환 import 회피)
    from run_demo import GazeSource, load_intrinsics

    cfg = yaml.safe_load(open(args.config))
    K = load_intrinsics()
    tags = TagBundleDetector(cfg, K)
    obj_dist = float(cfg.get("attend", {}).get("object_dist_m", 1.5))   # 스테레오 깊이 연결 전 가정 거리
    yolo = _Yolo(cfg["yolo"])                                           # 부위 YOLO(g1_parts.pt)
    obj_det = None                                                      # 물체 YOLO(yolo11l.pt) — 별도 subprocess
    if not args.no_obj:
        ocfg = cfg.get("obj_yolo", {})
        obj_model = args.obj_model or ocfg.get("model", "~/yolo-env/yolo11l.pt")
        obj_det = ObjectDetector(ocfg.get("python", cfg["yolo"]["python"]), model=obj_model,
                                 conf=float(ocfg.get("conf", 0.3)))
        obj_det.start()
        log.info("물체 YOLO: %s (classes=%s)", obj_model, ",".join(OBJ_TO_KIND))
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
            part_dets = yolo(frame)
            label = pick_part(part_dets, uv)
            bearing = None if T is None else bearing_deg(T)
            obj_b = None if (T is None or uv is None) else object_bearing_deg(T, uv, K, obj_dist)
            obj_kind, obj_dets = None, []
            if obj_det is not None and (uv is not None or args.show):   # 평소엔 시선 있을 때만 추론(부하 절반)
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    obj_dets = obj_det.feed(jpg.tobytes())
                    picked = pick_object(obj_dets, uv)
                    obj_kind = picked[0] if picked else None
            if args.show:
                cv2.imshow("gaze + YOLO", draw_overlay(frame, part_dets, obj_dets, uv, label, obj_kind, bearing))
                cv2.waitKey(1)
            # 무효여도 매 프레임 보낸다 — dwell 이 '안 봄'을 알아야 리셋된다
            sock.sendto(encode({"t": time.time(), "label": label, "bearing_deg": bearing,
                                "obj_bearing_deg": obj_b, "obj_kind": obj_kind, "valid": uv is not None}), dst)
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
        sock.sendto(encode({"t": time.time(), "label": None, "bearing_deg": None,
                            "obj_kind": None, "valid": False}), dst)
        sock.close()
        yolo.close()
        if obj_det is not None:
            obj_det.stop()
        src.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
