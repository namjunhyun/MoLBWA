#!/usr/bin/env python3
"""탑다운 카메라 호모그래피 캘리브레이션 — 안전 드라이버 위에서 직접 (ROS 불필요).

gaze_hri 의 topdown_click calib 모드를 대신한다. 그쪽을 그대로 쓰면 위험했다 (2026-09-24):
  * 파지점(TCP, l3=0.135)을 책상 위 5mm 에 두는데, 손가락 끝은 그보다 4.5cm 더 나가 있어
    45° 자세에서 **끝이 책상 아래 2.7cm** 로 가라는 명령이 된다. 게다가 화면에서 클릭하는 건
    손가락 끝이지 보이지 않는 파지점이 아니다.
  * 캘리브 지점 사이를 관절 보간(goto_joints)으로 옮겨 경로 중간 손가락 끝 검사가 없다.
그래서 여기서는
  * 기준점 = **두 손가락 끝의 가운데** (l3 + finger_tip_extra = 실측 0.18m 지점, 그리퍼 중심선 위).
    고정 손가락만 찍으면 중심선에서 1~2cm 옆이라 캘리브 전체가 그만큼 치우친다. 끝을 책상 위 tip_z 에.
  * 지점마다 "위로 들기(끝 +lift) -> 옆으로 -> 내려가기" 3단 이동, 각 구간 경로를 FK 로 샘플링해
    손가락 끝이 min_tip_z 아래로 가면 **움직이지 않는다**.
  * 결과는 topdown_click / control_panel 이 읽는 ~/.ros/topdown_homography.yaml 형식 그대로.
    H 는 화면 픽셀 -> **손가락 끝 XY** (책상 위 tip_z 평면). 로봇은 파지 때 같은 XY 로 TCP 를
    보내므로, 좌표 기준은 "손가락 끝이 그 XY 에 오는 곳"이 아니라 base_link XY 로 공통이다.

    python3 tools/topdown_calib.py --cam 0              # 캘리브 (창에서 손가락 끝 클릭)
    python3 tools/topdown_calib.py --cam 0 --check      # 경로 검사만, 팔 안 움직임

조작: 좌클릭 = 두 손가락 끝의 가운데 (한 번 클릭하면 바로 다음 점), s = 이번 점 건너뛰기,
      q = 중단(그 자리 정지). 잘못 찍은 점 하나는 RANSAC 이 걸러낸다.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "ros2_ws", "src", "gaze_hri")
sys.path.insert(0, PKG)
from gaze_hri.kinematics import ArmGeometry, IKError, inverse_kinematics, forward_kinematics  # noqa: E402
from gaze_hri.so101_driver import JointMap, SafetyStop, So101Bus, move_smooth  # noqa: E402

MAP = os.path.join(PKG, "config", "so101_follower.yaml")
PARAMS = os.path.join(PKG, "config", "gaze_hri.yaml")
OUT = os.path.expanduser("~/.ros/topdown_homography.yaml")
# 팔 앞 작업 영역 (base_link, m). 컵 배치(반경 0.30~0.34)를 감싸도록.
CALIB_XY = [(0.22, 0.12), (0.22, -0.12), (0.30, 0.16), (0.30, -0.16),
            (0.34, 0.08), (0.34, -0.08), (0.26, 0.00), (0.33, 0.00)]


def geometry():
    p = yaml.safe_load(open(PARAMS))["arm_server"]["ros__parameters"]
    g = ArmGeometry(base_height=p["base_height"], shoulder_offset=p["shoulder_offset"],
                    l1=p["l1"], l2=p["l2"], l3=p["l3"])
    tip = ArmGeometry(base_height=p["base_height"], shoulder_offset=p["shoulder_offset"],
                      l1=p["l1"], l2=p["l2"], l3=p["l3"] + p["finger_tip_extra"])
    return g, tip


def solve_tip(xyz, tip, pitches=(-0.785, -1.047, -0.524, -1.309)):
    """손가락 끝이 xyz 에 오는 관절각. 팔꿈치를 높이 드는 해 먼저."""
    for p in pitches:
        for up in (False, True):
            try:
                return list(inverse_kinematics(xyz, tip, approach_pitch=p, elbow_up=up)), p
            except IKError:
                continue
    raise IKError(f"손가락 끝 {xyz} 도달 불가")


def lowest_tip(q0, q1, tip, n=40):
    return min(float(forward_kinematics([a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * i / n))
                                         for a, b in zip(q0, q1)], tip)[2]) for i in range(n + 1))


def plan(tip_now_q, tip, tip_z, lift):
    """지점마다 [위, 아래] 관절각과 목표 XY. 경로 검사 포함."""
    legs, prev = [], tip_now_q
    for (x, y) in CALIB_XY:
        try:
            q_up, _ = solve_tip([x, y, tip_z + lift], tip)
            q_dn, _ = solve_tip([x, y, tip_z], tip)
        except IKError as e:
            print(f"  ({x:.2f},{y:+.2f}) 건너뜀: {e}")
            continue
        low1 = lowest_tip(prev, q_up, tip)          # 이전 -> 위
        low2 = lowest_tip(q_up, q_dn, tip)          # 위 -> 아래
        legs.append(((x, y), q_up, q_dn, min(low1, tip_z + lift - 0.005), low2))
        prev = q_up
    return legs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--tip-z", type=float, default=0.015, help="손가락 끝 높이 (책상 위, m)")
    ap.add_argument("--lift", type=float, default=0.06, help="지점 사이 이동 때 들어 올리는 높이")
    ap.add_argument("--min-tip-z", type=float, default=0.008)
    ap.add_argument("--speed", type=float, default=15.0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    g, tip = geometry()
    cfg = yaml.safe_load(open(MAP))
    m = JointMap(cfg["zero_ticks"], cfg["signs"])
    bus = So101Bus(cfg["port"], cfg["servo_ids"])
    try:
        q_now = m.to_rad(bus.ticks())
        legs = plan(q_now[:4], tip, a.tip_z, a.lift)
        bad = [(xy, l1, l2) for xy, _, _, l1, l2 in legs if min(l1, l2) < a.min_tip_z]
        print(f"캘리브 지점 {len(legs)}개, 손가락 끝 책상 위 {a.tip_z * 100:.1f}cm")
        for xy, l1, l2 in bad:
            print(f"  ✗ {xy}: 경로 중 손가락 끝 최저 {min(l1, l2) * 100:.1f}cm < {a.min_tip_z * 100:.1f}cm")
        if bad or len(legs) < 4:
            sys.exit("움직이지 않음 — 경로 검사 실패 또는 지점 부족")
        print("경로 검사 통과")
        if a.check:
            return 0

        cap = cv2.VideoCapture(a.cam, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            sys.exit(f"카메라 {a.cam} 열기 실패")
        win = "topdown calib - click midpoint of finger tips"
        cv2.namedWindow(win)
        click = {"p": None}
        cv2.setMouseCallback(win, lambda e, x, y, f, p: click.update(p=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

        bus.enable()
        grip = cfg["zero_ticks"][5] + int(0.8 / (2 * math.pi / 4096)) * cfg["signs"][5]
        pix, world = [], []
        for i, ((x, y), q_up, q_dn, _, _) in enumerate(legs):
            for q in (q_up, q_dn):
                target = m.to_ticks(list(q) + [0.0, 0.0])
                target[4] = bus.last_goal[4]                     # 롤은 그대로
                target[5] = grip                                 # 그리퍼 조금 열기 (끝이 잘 보이게)
                move_smooth(bus, target, max_deg_s=a.speed)
            time.sleep(0.8)                                      # 흔들림 가라앉기
            click["p"] = None
            print(f"[{i + 1}/{len(legs)}] 손가락 끝 = 로봇 ({x:.2f}, {y:+.2f}) — 화면에서 **두 손가락 끝의 가운데** 클릭")
            while True:
                ok, f = cap.read()
                if not ok:
                    continue
                vis = f.copy()
                for (u, v) in pix:
                    cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
                cv2.putText(vis, f"{i + 1}/{len(legs)} click MIDPOINT of finger tips  (s=skip q=quit)",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if click["p"]:
                    cv2.drawMarker(vis, click["p"], (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
                cv2.imshow(win, vis)
                k = cv2.waitKey(20) & 0xFF
                if k == ord("q"):
                    raise KeyboardInterrupt
                if k == ord("s"):
                    break
                if click["p"]:
                    pix.append(click["p"])
                    world.append((x, y))
                    print(f"    픽셀 {click['p']}")
                    break
            # 다음 지점으로 가기 전에 위로
            t = m.to_ticks(list(q_up) + [0.0, 0.0])
            t[4], t[5] = bus.last_goal[4], grip
            move_smooth(bus, t, max_deg_s=a.speed)

        if len(pix) < 4:
            sys.exit("클릭 4개 미만 — 저장 안 함")
        src = np.array(pix, np.float32)
        dst = np.array(world, np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.008)
        if H is None:
            sys.exit("호모그래피 계산 실패")
        pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        per = np.linalg.norm(pred - dst, axis=1) * 1000
        # 한 점씩 빼고 나머지로 예측 (처음 보는 점 오차)
        loo = []
        for j in range(len(src)):
            k = np.arange(len(src)) != j
            Hj, _ = cv2.findHomography(src[k], dst[k], 0)
            if Hj is not None:
                loo.append(float(np.linalg.norm(cv2.perspectiveTransform(src[j:j + 1].reshape(-1, 1, 2), Hj).ravel() - dst[j]) * 1000))
        print("점별 오차(mm):", np.round(per, 1).tolist(), f"| LOO 평균 {np.mean(loo):.1f}mm 최대 {max(loo):.1f}mm")
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        yaml.safe_dump({"H": H.tolist(), "reproj_error_mm": float(per.mean()), "loo_error_mm": float(np.mean(loo)),
                        "num_points": len(pix), "reference": "midpoint of finger tips",
                        "tip_z_m": a.tip_z, "camera_index": a.cam,
                        "pixels": [list(map(int, p)) for p in pix], "world_xy": [list(w) for w in world]},
                       open(OUT, "w"), default_flow_style=False, allow_unicode=True)
        print(f"저장: {OUT}")
    except SafetyStop as e:
        print(f"안전 정지: {e}")
        return 2
    except KeyboardInterrupt:
        bus.hold()
        print("중단 — 그 자리에 멈춤")
        return 1
    finally:
        cv2.destroyAllWindows()
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
