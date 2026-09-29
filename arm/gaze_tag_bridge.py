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

--slam (2026-09-24, 시연에는 SLAM 도 쓴다): 태그가 이번 프레임에 보이면 태그 직결을 쓰고,
안 보이면 AnchorTracker 가 태그로 latch 해 둔 팔 자세 + SLAM 머리 자세로 이어 간다.
태그가 보일 때마다 두 경로의 차이(drift)가 기록된다 — SLAM 규약(T_w_hc)이 틀렸거나 발산하면
이 값이 바로 커진다. SLAM 포즈는 발산 방어(비현실적 위치/한 번에 큰 점프)를 거친다.

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

from anchor import AnchorTracker, TagBundleDetector, TagDirectAnchor
from perception import gaze_ray_in_base, gaze_ray_in_base_from_ray, ray_hit_height
from run_demo import GazeSource, load_intrinsics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from gaze_udp_sender import GazeUdpSender  # noqa: E402

log = logging.getLogger("gaze_tag_bridge")


class GazeToBase:
    """한 프레임 처리: 태그 관측(+SLAM) + 시선 픽셀 -> base_link 응시점과 헤드캠 자세.

    ROS/카메라 없이 테스트할 수 있게 루프에서 떼어 냈다.
    """

    def __init__(self, K, plane_z, direct_cfg=None, tracker_cfg=None, use_slam=False,
                 max_pose_jump_m=0.3, max_pose_norm_m=10.0, calib_mode=False, hold_s=0.0):
        self.K = K
        self.plane_z = plane_z
        self.direct = TagDirectAnchor(**(direct_cfg or {}))
        self.tracker = AnchorTracker(**(tracker_cfg or {})) if use_slam else None
        self.max_pose_jump_m = max_pose_jump_m
        self.max_pose_norm_m = max_pose_norm_m
        self._last_slam_pos = None
        self.slam_rejects = 0
        # ★ 2026-09-27: --calib-mode. T_BW 캘리브는 그리퍼 끝(공중, 다양한 높이)을 본다.
        # 평면 교차(plane_z)는 그 높이에서 대부분 "광선이 테이블 쪽을 안 향함"으로 막힌다.
        # 이 모드에서는 교차 대신 광선 위의 임의 점(1m 지점)을 그대로 보낸다.
        # 소비자(calibrate_world_to_base)가 origin+point로 광선을 재구성해 진짜 높이와
        # 교차시키므로, 어느 점이든 광선 위에 있기만 하면 된다.
        self.calib_mode = calib_mode
        self.last_ray = None
        self.hold_s = 0.0 if calib_mode else hold_s   # 캘리브 때는 옛 자세를 섞지 않는다
        self._last_T, self._last_T_t = None, 0.0

    def update_slam(self, T_w_hc, map_id, ok, t=None):
        """SLAM 포즈 입력. 발산 방어: ORB-SLAM3 는 발산 중에도 30Hz 로 포즈를 낸다
        (2026-09-22 실측 565m 가 '신선한' 포즈로 통과 — HANDOFF_2026-09-23 §6a)."""
        if self.tracker is None or T_w_hc is None:
            return
        pos = np.asarray(T_w_hc, float)[:3, 3]
        if ok:
            bad = (not np.all(np.isfinite(T_w_hc)) or np.linalg.norm(pos) > self.max_pose_norm_m
                   or (self._last_slam_pos is not None
                       and np.linalg.norm(pos - self._last_slam_pos) > self.max_pose_jump_m))
            if bad:
                ok = False
                self.slam_rejects += 1
                # ★ 튄 그 프레임만 거르면 안 된다. ORB-SLAM3 가 추적을 잃고 새 맵을 만들면
                # 이후 포즈는 새 원점 기준으로 '매끄럽게' 이어져 필터를 통과한다. map_id 패치가
                # 아직 없으니 latch 를 버린다 -> 태그를 다시 볼 때까지 SLAM 경로를 안 쓴다.
                self.tracker.T_w_ab = None
                self.tracker.latched_map_id = None
        self._last_slam_pos = pos if np.all(np.isfinite(pos)) else None
        self.tracker.update_slam(np.asarray(T_w_hc, float), map_id, ok, t)

    def step(self, T_tag, gaze_uv, t=None, gaze_ray=None):
        """-> (p_ab or None, T_origin or None, 출처 'tag'/'slam' 또는 무효 사유)

        gaze_ray: (origin_hc, direction_hc) 또는 None. docs/12 (R,p_eye) 실전 연결 —
        있으면 이걸 우선 쓴다(픽셀+K 역투영 없이 광선을 그대로 armbase로 옮김,
        gaze_ray_in_base_from_ray 참고). 없으면 옛 픽셀 경로(gaze_uv+K)로 대체한다.

        T_origin: 회전은 T_ab_hc, 이동은 **실제 광선 원점**. 하류(/head/pose →
        dwell_detector, target_resolver, calibrate_world_to_base)는 이 이동만 광선
        원점으로 쓰므로, 광선 경로에서 헤드캠 위치를 보내면 p_eye 오프셋만큼 다른
        직선이 재구성된다(코드리뷰 #2). 픽셀 경로에서는 원점 = 헤드캠 위치라 변화 없음.
        """
        if T_tag is not None:
            self.direct.update_tag(T_tag)
            if self.tracker is not None:
                self.tracker.update_tag(T_tag, t)       # latch 생성/갱신 + drift 기록
        else:
            self.direct.miss()

        T, src = self.direct.T_headcam_to_armbase(), "tag"
        # 2026-09-27: 태그가 한두 프레임 깜빡이면 즉시 무효 -> dwell 버퍼가 0.35초마다 리셋돼
        # 컵을 2초 봐도 응시 확정이 안 났다(실측 태그 유효 34~75%). 시선 '선택'용으로는 0.3초
        # 전 머리 자세를 써도 무해하다(팔 좌표는 탑다운 스냅이 정한다). 그 이상은 안 이어 쓴다.
        now = time.time() if t is None else t
        if T is not None:
            self._last_T, self._last_T_t = T, now
        elif (self._last_T is not None and self.hold_s > 0
              and now - self._last_T_t <= self.hold_s):
            T, src = self._last_T, "tag-hold"
        if T is None and self.tracker is not None:
            T, src = self.tracker.T_headcam_to_armbase(), "slam"
        if T is None:
            st = self.direct.state.value
            if self.tracker is not None:
                st += f"/SLAM {self.tracker.state.value}"
            return None, None, f"태그 {st}"
        if gaze_ray is not None:
            origin, direction = gaze_ray_in_base_from_ray(gaze_ray[0], gaze_ray[1], T)
        elif gaze_uv is not None:
            origin, direction = gaze_ray_in_base(gaze_uv, self.K, T)
        else:
            return None, None, "시선 없음"
        self.last_ray = (origin, direction)   # --debug-ray 진단용 (무효 판정 전 광선)
        if self.calib_mode:
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                return None, None, "광선 방향 계산 불가"
            p = origin + direction / norm * 1.0
        else:
            p = ray_hit_height(origin, direction, self.plane_z)
            if p is None:
                return None, None, "광선이 테이블 쪽을 안 향함"
        T_origin = np.linalg.inv(T).copy()
        T_origin[:3, 3] = origin      # 하류가 origin→p 로 광선을 재구성 — 진짜 원점을 싣는다
        return p, T_origin, src


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
    ap.add_argument("--gaze-ray-port", type=int, default=55059,
                    help="docs/12 (R,p_eye) 3D 광선 수신 포트. 안 켜져 있거나 아직 캘리브 "
                         "전이면 자동으로 옛 픽셀(--gaze-px-port) 경로로 대체됨")
    ap.add_argument("--slam", action="store_true",
                    help="태그가 안 보일 때 SLAM(/orbslam3/pose)으로 이어 간다")
    ap.add_argument("--calib-mode", action="store_true",
                    help="T_BW 캘리브용. 테이블 평면 교차 대신 광선 위 점을 그대로 보낸다 "
                         "(그리퍼 끝처럼 공중의 점을 볼 때 쓸 것)")
    ap.add_argument("--tag-hold-s", type=float, default=0.3,
                    help="태그를 놓친 프레임에 직전 머리 자세를 이어 쓰는 최대 시간 [s] (0=끔)")
    ap.add_argument("--debug-ray", action="store_true",
                    help="2초마다 armbase 기준 광선 원점/방향 중앙값과 평면까지 거리 출력")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(args.config))
    K = load_intrinsics()
    tags = TagBundleDetector(cfg, K)
    a = cfg["anchor"]
    g2b = GazeToBase(K, args.table_z + args.gaze_height, a.get("direct", {}),
                     {k: a[k] for k in ("stale_after_s", "drift_warn_m", "drift_max_m",
                                        "latch_ema_alpha")},
                     use_slam=args.slam, calib_mode=args.calib_mode, hold_s=args.tag_hold_s)
    src = GazeSource(use_ros=True, tag_direct=True, scene_topic=args.scene_topic,
                     gaze_px_port=args.gaze_px_port, gaze_ray_port=args.gaze_ray_port)
    sender = GazeUdpSender(args.udp_host, args.udp_port)
    log.info("송신 %s:%d, 응시점 높이 z=%.3f (테이블 %.3f + %.3f), SLAM %s",
             args.udp_host, args.udp_port, g2b.plane_z, args.table_z, args.gaze_height,
             "사용" if args.slam else "안 씀(태그 전용)")

    import cv2
    n_frames = n_valid = n_ray_input = 0
    why, srcs = {}, {}
    dbg_o, dbg_d = [], []
    next_report = time.time() + 2.0
    try:
        while src.ok():          # SIGTERM 이면 rclpy 가 컨텍스트를 내린다 -> 빠져나감
            frame, gaze_uv, _ = src.frame()
            if frame is None:
                time.sleep(0.005)
                continue
            n_frames += 1

            if args.slam:
                g2b.update_slam(*src.slam())
            T = tags.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            gaze_ray = src.gaze_ray()
            if gaze_ray is not None:
                n_ray_input += 1
            g2b.last_ray = None
            p, T_origin, info = g2b.step(T, gaze_uv, gaze_ray=gaze_ray)
            if args.debug_ray and g2b.last_ray is not None:
                dbg_o.append(g2b.last_ray[0]); dbg_d.append(g2b.last_ray[1])

            # 무효일 때도 매 프레임 보낸다 — 안 보내면 dwell_detector 가 마지막 유효점을 붙든다
            if p is not None:
                sender.send(p, valid=True, T_WS=T_origin)  # 이동 = 광선 원점(p_eye 반영)
                n_valid += 1
                srcs[info] = srcs.get(info, 0) + 1
            else:
                sender.send([0.0, 0.0, 0.0], valid=False)
                why[info] = why.get(info, 0) + 1

            if time.time() >= next_report:
                reproj = getattr(tags, "last_reproj_px", float("nan"))
                extra = ""
                if g2b.tracker is not None:
                    d = g2b.tracker.last_drift_m
                    extra = (f", SLAM drift {d * 100:.1f}cm" if np.isfinite(d) else ", SLAM drift -")
                    extra += f", SLAM 포즈 거부 {g2b.slam_rejects}"
                log.info("프레임 %d, 유효 송신 %d (%.0f%%, 출처 %s), 무효 사유 %s, 태그 재투영 %.2fpx, "
                         "시선 입력=%s (광선 수신 %d/%d프레임)%s",
                         n_frames, n_valid, 100.0 * n_valid / max(n_frames, 1),
                         srcs or "-", why or "-", reproj,
                         "광선(R,p_eye)" if n_ray_input else "픽셀(구형)", n_ray_input, n_frames, extra)
                if args.debug_ray and dbg_o:
                    o = np.median(dbg_o, axis=0); dd = np.median(dbg_d, axis=0)
                    dd = dd / np.linalg.norm(dd)
                    t = (g2b.plane_z - o[2]) / dd[2] if abs(dd[2]) > 1e-6 else float("inf")
                    elev = np.degrees(np.arcsin(dd[2]))
                    log.info("  [debug-ray] n=%d 원점=%s 방향=%s (앙각 %+.1f°, 평면까지 t=%.2fm, "
                             "방향 z 표준편차 %.3f)", len(dbg_o), np.round(o, 3), np.round(dd, 3),
                             elev, t, float(np.std(np.asarray(dbg_d)[:, 2])))
                dbg_o, dbg_d = [], []
                n_frames = n_valid = n_ray_input = 0
                why, srcs = {}, {}
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
