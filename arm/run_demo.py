#!/usr/bin/env python
"""메인 루프 — MoLBWA 시선 파이프라인 + SO-ARM101.

MoLBWA 쪽에서 가져와야 하는 것 (아래 GazeSource 를 채우면 된다):
  * scene 좌영상 (rectified BGR)   <- src/ocams_calib.py 의 rectify map
  * 시선점 (u,v)                   <- src/gaze_on_scene.py
  * depth [m] (없으면 테이블 평면 교차로 대체)
  * T_w_hc, map_id, tracking_ok    <- /orbslam3/pose (+ map_id 퍼블리시 패치 필요)

실행:
  python run_demo.py --sim              # 하드웨어 0개. 합성 장면으로 전 구간 검증
  python run_demo.py --dry-run --no-ros # 팔/카메라 없이 로직만
  python run_demo.py --no-arm           # 인식+anchor 만, 팔은 안 움직임
  python run_demo.py                    # 전체
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

import numpy as np
import yaml

from anchor import AnchorState, AnchorTracker, TagBundleDetector
from arm import Arm
from perception import CupDetector, GazeDwell, cup_position_on_table
from task import DrinkTask

log = logging.getLogger("molbwa")

# ★ intrinsics 는 하드코딩하지 않는다 (2026-08-19).
# 예전엔 여기 [[433.55,0,470.35],...] 가 박혀 있었는데, 그건 2026-07-14 의 낡은
# (순환) 캘리브레이션 값이었다. ORB-SLAM3 는 484.71/326.84 로 pose 를 만들고 있었으므로
# pose 좌표계와 역투영 좌표계가 서로 달랐고, 0.7m 거리에서 횡방향 20cm 넘게 틀어졌다.
# 이제 src/ocams_calib.py 를 유일한 출처로 쓴다. 그쪽이 ORB-SLAM3 설정과의 일치까지 검사한다.
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))


def load_intrinsics():
    """rectified K 를 ocams_calib 에서 가져온다. 실패하면 조용히 넘어가지 않고 죽는다."""
    from ocams_calib import RECTIFIED_K
    return np.asarray(RECTIFIED_K, dtype=float)


class GazeSource:
    """MoLBWA 파이프라인 어댑터.

    ★ 여기가 유일한 통합 지점이다. frame() 을 MoLBWA 코드로 채우면 끝.

    slam() 은 이미 동작한다. rclpy 스핀을 **별도 스레드**에서 돌리기 때문에,
    팔이 움직이느라 메인 루프가 블로킹된 동안에도 최신 pose 가 계속 갱신된다.
    (예전엔 메인 루프에서만 spin_once 를 불러서, 30초짜리 팔 동작 내내 pose 가
     멈춰 있었고 task.py 의 안전 게이트가 낡은 데이터를 검사하고 있었다.)
    """

    def __init__(self, use_ros: bool = True, pose_timeout_s: float = 0.5):
        self.use_ros = use_ros
        self.pose_timeout_s = pose_timeout_s
        self._node = None
        self._thread = None
        if use_ros:
            self._init_ros()

    def _init_ros(self):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from std_msgs.msg import Int32

        if not rclpy.ok():
            rclpy.init()

        class _Sub(Node):
            def __init__(self):
                super().__init__("molbwa_arm_bridge")
                self.lock = threading.Lock()
                self.T_w_hc = None
                self.map_id = 0
                self.last_pose_t = 0.0
                self.create_subscription(PoseStamped, "/orbslam3/pose", self._pose, 10)
                # ★ ORB-SLAM3 패치로 추가해야 하는 토픽. patches/orbslam3_ros2.patch 참고.
                self.create_subscription(Int32, "/orbslam3/map_id", self._map, 10)

            def _pose(self, msg):
                from scipy.spatial.transform import Rotation as R
                p, q = msg.pose.position, msg.pose.orientation
                T = np.eye(4)
                T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                T[:3, 3] = [p.x, p.y, p.z]
                with self.lock:
                    self.T_w_hc = T
                    self.last_pose_t = time.time()

            def _map(self, msg):
                with self.lock:
                    self.map_id = int(msg.data)

        self._rclpy = rclpy
        self._node = _Sub()
        self._thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._thread.start()
        log.info("ROS2 브릿지 시작 (spin 스레드 분리)")

    def slam(self):
        """-> (T_w_hc, map_id, tracking_ok)

        tracking_ok 는 "최근 pose_timeout_s 안에 pose 가 왔는가"로 판정한다.
        ORB_SLAM3_ROS2 는 트래킹을 놓치면 pose 발행을 멈추므로, 끊김 = 트래킹 유실이다.
        """
        if self._node is None:
            return None, 0, False
        with self._node.lock:
            T, mid, t_last = self._node.T_w_hc, self._node.map_id, self._node.last_pose_t
        ok = T is not None and (time.time() - t_last) < self.pose_timeout_s
        return T, mid, ok

    def frame(self):
        """-> (scene_bgr, gaze_uv or None, depth_m or None)

        TODO(MoLBWA): src/gaze_on_scene.py 의 루프를 여기에 연결.
        지금은 통합 전이라 None 을 돌려준다. --sim 으로 전 구간 검증은 가능하다.
        """
        return None, None, None

    def shutdown(self):
        if self._node is not None:
            self._rclpy.shutdown()


def build_pipeline(cfg, K, args):
    """anchor / 태그검출기 / 컵검출기 / dwell 을 만든다."""
    anchor = AnchorTracker(**{k: cfg["anchor"][k] for k in
                              ("stale_after_s", "drift_warn_m", "drift_max_m", "latch_ema_alpha")})

    # 무거운 서드파티(ultralytics / pupil_apriltags)는 없으면 없는 대로 간다.
    # 예전엔 여기서 바로 ModuleNotFoundError 로 죽어서, README 가 "바로 된다"고 적은
    # --dry-run --no-ros 조차 실행이 안 됐다.
    tags, cups_det = None, None
    if not args.sim:
        try:
            tags = TagBundleDetector(cfg, K)
        except ImportError:
            log.warning("pupil_apriltags 없음 -> 태그 검출 비활성 (pip install pupil-apriltags)")
        try:
            cups_det = CupDetector(cfg)
        except ImportError:
            log.warning("ultralytics 없음 -> 컵 검출 비활성 (pip install ultralytics)")
    dwell = GazeDwell(cfg["gaze"]["dwell_s"], cfg["gaze"]["dwell_release_s"],
                      track_radius_px=cfg["gaze"].get("track_radius_px", 60.0))
    return anchor, tags, cups_det, dwell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="팔 하드웨어 없이")
    ap.add_argument("--no-arm", action="store_true", help="팔은 절대 안 움직임 (인식만)")
    ap.add_argument("--no-ros", action="store_true", help="ROS2 없이 (SLAM 입력 없음)")
    ap.add_argument("--sim", action="store_true",
                    help="하드웨어 0개. 합성 시선/태그/컵으로 전 구간 검증 (--dry-run --no-ros 포함)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(args.config))

    if args.sim:
        args.dry_run = True
        args.no_ros = True

    K = load_intrinsics()
    log.info("rectified K: fx=%.2f cx=%.2f cy=%.2f", K[0, 0], K[0, 2], K[1, 2])

    anchor, tags, cups_det, dwell = build_pipeline(cfg, K, args)

    if args.sim:
        from sim_source import SimSource
        src = SimSource(cfg, K)
    else:
        src = GazeSource(use_ros=not args.no_ros)

    arm = Arm(cfg, dry_run=args.dry_run)
    arm.connect()
    arm.home()

    def refresh_slam():
        """★ 팔이 움직이는 도중에도 게이트가 최신 SLAM 상태를 보게 한다."""
        T_w_hc, map_id, ok = src.slam()
        if T_w_hc is not None:
            anchor.update_slam(T_w_hc, map_id, ok)

    task = DrinkTask(cfg, arm, anchor, refresh=refresh_slam)

    log.info("시작. 팔 쪽 태그를 한 번 봐 주세요.")
    last_state = None
    try:
        while True:
            refresh_slam()

            frame, gaze_uv, depth = src.frame()
            if frame is None:
                time.sleep(0.03)
                continue

            if args.sim:
                T_hc_ab = src.tag_pose()
                cups = src.cups()
            else:
                T_hc_ab, cups = None, []
                if tags is not None:
                    import cv2
                    T_hc_ab = tags.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if cups_det is not None:
                    cups = cups_det.detect(frame, depth, K)
            if T_hc_ab is not None:
                anchor.update_tag(T_hc_ab)

            if anchor.state is not last_state:
                drift_cm = anchor.last_drift_m * 100 if np.isfinite(anchor.last_drift_m) else 0.0
                log.info("anchor: %s (drift %.1fcm)", anchor.state.value, drift_cm)
                last_state = anchor.state

            # depth 가 없어도 컵이 테이블 위에 있다는 사실만으로 3D 위치가 나온다.
            T_hc_ab_now = anchor.T_headcam_to_armbase()
            if T_hc_ab_now is not None:
                for c in cups:
                    if c.p_cam is None:
                        c.p_cam = cup_position_on_table(c, T_hc_ab_now, cfg["task"]["table_z"], K)

            cup = dwell.update(cups, gaze_uv)
            if cup is None:
                continue

            if cup.p_cam is None:
                log.warning("컵 %d 의 3D 위치를 못 얻음 (depth 없음 + anchor 미확보) — 선택 무시", cup.idx)
                dwell.reset()
                continue

            p_ab = anchor.point_headcam_to_armbase(cup.p_cam)
            if p_ab is None:
                log.warning("anchor 미확보 — 팔 쪽 태그를 봐 주세요")
                dwell.reset()
                continue

            log.info("컵 %d -> armbase %s", cup.idx, np.round(p_ab, 3).tolist())
            if args.no_arm:
                dwell.reset()
                continue

            stage = task.run(p_ab)
            log.info("시퀀스 종료: %s", stage.value)
            dwell.reset()
            if args.sim:
                log.info("--sim: 1회 시퀀스 완료, 종료")
                return 0

    except KeyboardInterrupt:
        log.info("사용자 중단")
        task.abort()
    finally:
        arm.disconnect()
        if hasattr(src, "shutdown"):
            src.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
