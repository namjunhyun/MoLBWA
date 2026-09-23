#!/usr/bin/env python
"""시선 픽셀 + AprilTag -> base_link 시선 광선 -> gaze_hri (UDP 55055).

SLAM 없이 gaze_hri(컵 옮기기) 스택에 시선을 넣는 다리. 2026-09-23 결정:
컵 위치는 탑다운 카메라가 정확히 잡고, 시선은 "어느 컵/어느 자리"만 고른다.
그 비교를 하려면 시선이 로봇 좌표(base_link)로 와야 하는데, 원래 설계의
스테레오 깊이 + SLAM + T_BW 대신 팔 뒤 태그 번들로 매 프레임 바로 구한다.

    gaze_on_scene.py --send-gaze-px ──UDP 55056──┐
    /camera/left/compressed ── 태그 검출 ────────┴─> 이 스크립트 ──UDP 55055──> gaze_bridge
                                                     (점 + 머리 위치, base_link 기준)

보내는 것:
  point : 시선 광선이 높이 --gaze-height(컵 몸통 중간) 평면을 뚫는 점
  head  : 헤드캠 위치/자세 (base_link 기준) -> target_resolver 가 광선으로 쓴다
          (스냅 = 광선과 컵 중심 거리, 놓을 자리 = 광선과 테이블 교차)
  point 를 어느 높이에 찍느냐는 head 가 있으면 결과에 영향이 없다. head 가 끊겨
  직교 투영으로 물러설 때 컵 쪽에 가깝도록 컵 중간 높이로 둔다.

gaze_hri 쪽은 gaze_frame:=base_link 로 띄운다 (map<->base_link 변환 없이 그대로):
    ros2 launch gaze_hri gaze_hri.launch.py source:=udp gaze_frame:=base_link

실행:
    python3 -u gaze_tag_bridge.py            # --table-z/--gaze-height 는 gaze_hri.yaml 과 맞출 것
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import yaml

from anchor import TagBundleDetector, TagDirectAnchor
from perception import gaze_ray_in_base, ray_hit_height
from run_demo import GazeSource, load_intrinsics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from gaze_udp_sender import GazeUdpSender  # noqa: E402

log = logging.getLogger("gaze_tag_bridge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--table-z", type=float, default=0.0,
                    help="테이블면의 base_link 높이 [m]. gaze_hri.yaml table_z 와 같게")
    ap.add_argument("--gaze-height", type=float, default=0.045,
                    help="테이블 위 이 높이에 응시점을 찍는다. gaze_hri.yaml grasp_height 와 같게")
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=55055)
    ap.add_argument("--scene-topic", default="/camera/left/compressed")
    ap.add_argument("--gaze-px-port", type=int, default=55056)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(args.config))
    K = load_intrinsics()
    tags = TagBundleDetector(cfg, K)
    anchor = TagDirectAnchor(**cfg["anchor"].get("direct", {}))
    src = GazeSource(use_ros=True, tag_direct=True, scene_topic=args.scene_topic,
                     gaze_px_port=args.gaze_px_port)
    sender = GazeUdpSender(args.udp_host, args.udp_port)
    plane_z = args.table_z + args.gaze_height
    log.info("송신 %s:%d, 응시점 높이 z=%.3f (테이블 %.3f + %.3f)",
             args.udp_host, args.udp_port, plane_z, args.table_z, args.gaze_height)

    import cv2
    n_frames = n_valid = 0
    why = {}
    next_report = time.time() + 2.0
    try:
        while src.ok():          # SIGTERM 이면 rclpy 가 컨텍스트를 내린다 -> 빠져나감
            frame, gaze_uv, _ = src.frame()
            if frame is None:
                time.sleep(0.005)
                continue
            n_frames += 1

            T = tags.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if T is not None:
                anchor.update_tag(T)
            else:
                anchor.miss()

            # 무효일 때도 매 프레임 보낸다 — 안 보내면 dwell_detector 가 마지막 유효점을 붙든다
            reason = None
            T_now = anchor.T_headcam_to_armbase()
            if T_now is None:
                reason = f"태그 {anchor.state.value}"
            elif gaze_uv is None:
                reason = "시선 없음"
            else:
                origin, direction = gaze_ray_in_base(gaze_uv, K, T_now)
                p = ray_hit_height(origin, direction, plane_z)
                if p is None:
                    reason = "광선이 테이블 쪽을 안 향함"
                else:
                    T_ab_hc = np.linalg.inv(T_now)
                    sender.send(p, valid=True, T_WS=T_ab_hc)
                    n_valid += 1
            if reason is not None:
                sender.send([0.0, 0.0, 0.0], valid=False)
                why[reason] = why.get(reason, 0) + 1

            if time.time() >= next_report:
                reproj = getattr(tags, "last_reproj_px", float("nan"))
                log.info("프레임 %d, 유효 송신 %d (%.0f%%), 무효 사유 %s, 태그 재투영 %.2fpx",
                         n_frames, n_valid, 100.0 * n_valid / max(n_frames, 1),
                         why or "-", reproj)
                n_frames = n_valid = 0
                why = {}
                next_report = time.time() + 2.0
    except KeyboardInterrupt:
        pass
    finally:
        sender.send([0.0, 0.0, 0.0], valid=False)     # 끊길 때 옛 시선을 붙들지 않게
        sender.close()
        src.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
