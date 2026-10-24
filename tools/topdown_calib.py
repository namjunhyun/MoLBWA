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
from gaze_hri import arms  # noqa: E402
from gaze_hri.so101_driver import JointMap, SafetyStop, So101Bus, move_smooth, settle_ticks  # noqa: E402

PARAMS = os.path.join(PKG, "config", "gaze_hri.yaml")
# 팔 앞 작업 영역 (base_link, m). 컵 배치(반경 0.30~0.34)를 감싸도록.
# 2026-09-29: 실제로 집고 놓는 범위(받침대에서 16~29cm, 좌우 ±20cm)로 넓게 다시 잡음.
# 예전 값(앞 22~34cm 에 몰림): [(0.22,0.12),(0.22,-0.12),(0.30,0.16),(0.30,-0.16),(0.34,0.08),(0.34,-0.08),(0.26,0),(0.33,0)]
# 2026-10-12: 컵 수평 옆 집기 — 컵은 37~42cm, 이동은 그 뒤 30cm 부근. 그 범위를 감싸게 다시 잡음.
# (그 전) [(0.18,0.14),(0.18,-0.14),(0.24,0.20),(0.24,-0.20),(0.30,0.10),(0.30,-0.10),(0.22,0),(0.29,0)]
CALIB_XY = [(0.27, 0.14), (0.27, -0.14), (0.33, 0.20), (0.33, -0.20),
            (0.40, 0.12), (0.40, -0.12), (0.30, 0.00), (0.42, 0.00)]
# 2026-10-22: 높은 점 4개 추가 (손가락 끝 높이 --high-z, 기본 15cm = 컵 입구 높이). 책상 높이 8점은 그대로
# 호모그래피(책상 면 -> 화면)에만 쓰고, 12점 전체(높이가 두 가지)로 카메라 투영 행렬 P 를 푼다
# -> 카메라 위치·각도를 알면 컵 중심을 '박스 아래 변 + 고정 3cm' 대신 기하학으로 구할 수 있다.
CALIB_XY_HIGH = [(0.28, 0.13), (0.28, -0.13), (0.34, 0.05), (0.34, -0.05)]   # 15cm+들기 6cm 에서 닿는 곳


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


def plan(tip_now_q, tip, tip_z, lift, high_z=None):
    """지점마다 [위, 아래] 관절각과 목표 XYZ. 경로 검사 포함."""
    legs, prev = [], tip_now_q
    pts = [(x, y, tip_z) for x, y in CALIB_XY]
    if high_z:
        pts += [(x, y, high_z) for x, y in CALIB_XY_HIGH]
    for (x, y, z) in pts:
        try:
            q_up, _ = solve_tip([x, y, z + lift], tip)
            q_dn, _ = solve_tip([x, y, z], tip)
        except IKError as e:
            print(f"  ({x:.2f},{y:+.2f}) 건너뜀: {e}")
            continue
        low1 = lowest_tip(prev, q_up, tip)          # 이전 -> 위
        low2 = lowest_tip(q_up, q_dn, tip)          # 위 -> 아래
        legs.append(((x, y, z), q_up, q_dn, min(low1, tip_z + lift - 0.005), low2))
        prev = q_up
    return legs


def solve_camera(pix, xyz):
    """3D 점(높이 두 가지) <-> 화면 점으로 카메라 투영 행렬 P (정규화 DLT). 렌즈 왜곡은 무시.
    돌려주는 것: P, K, 카메라 위치(팔 기준), 재투영 오차(px). 점이 6개 미만이거나 한 평면이면 빈 dict."""
    pix, xyz = np.asarray(pix, float), np.asarray(xyz, float)
    if len(pix) < 6 or np.ptp(xyz[:, 2]) < 0.03:
        print("카메라 자세: 높이가 다른 점이 부족해 건너뜀")
        return {}

    def norm(pts):
        c = pts.mean(axis=0)
        sc = math.sqrt(pts.shape[1]) / np.mean(np.linalg.norm(pts - c, axis=1))
        T = np.eye(pts.shape[1] + 1)
        T[:-1, :-1] *= sc
        T[:-1, -1] = -sc * c
        return T
    T2, T3 = norm(pix), norm(xyz)
    p2 = (T2 @ np.c_[pix, np.ones(len(pix))].T).T
    p3 = (T3 @ np.c_[xyz, np.ones(len(xyz))].T).T
    rows = []
    for (u, v, w), X in zip(p2, p3):
        rows.append(np.r_[np.zeros(4), -w * X, v * X])
        rows.append(np.r_[w * X, np.zeros(4), -u * X])
    _, _, vt = np.linalg.svd(np.array(rows))
    P = np.linalg.inv(T2) @ vt[-1].reshape(3, 4) @ T3
    P /= np.linalg.norm(P[2, :3])
    proj = (P @ np.c_[xyz, np.ones(len(xyz))].T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - pix, axis=1)
    K, R, C, *_ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2, 2]
    C = (C[:3] / C[3]).ravel()
    print(f"카메라 자세: 위치(팔 기준) {np.round(C, 3).tolist()} m, 초점 fx {K[0, 0]:.0f} fy {K[1, 1]:.0f}, "
          f"중심 ({K[0, 2]:.0f}, {K[1, 2]:.0f}) | 재투영 오차 평균 {err.mean():.1f}px 최대 {err.max():.1f}px")
    if err.max() > 6.0:
        print("  ★ 재투영 오차가 큼 — 렌즈 왜곡/클릭 실수 가능. 체커보드 렌즈 캘리브를 고려할 것")
    return {"P": P.tolist(), "K": K.tolist(), "camera_center": C.tolist(),
            "cam_reproj_px_mean": float(err.mean()), "cam_reproj_px_max": float(err.max()),
            "points3d": xyz.tolist(), "pixels3d": pix.astype(int).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--tip-z", type=float, default=0.015, help="손가락 끝 높이 (책상 위, m)")
    ap.add_argument("--lift", type=float, default=0.06, help="지점 사이 이동 때 들어 올리는 높이")
    ap.add_argument("--min-tip-z", type=float, default=0.008)
    ap.add_argument("--speed", type=float, default=15.0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--high-z", type=float, default=0.15,
                    help="높은 점 손가락 끝 높이 (0 이면 높은 점 안 찍음)")
    ap.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
    a = ap.parse_args()

    g, tip = geometry()
    global OUT
    OUT = arms.homography_file(a.arm)
    cfg = yaml.safe_load(open(arms.map_file(a.arm)))
    m = JointMap(cfg["zero_ticks"], cfg["signs"])
    bus = So101Bus(cfg["port"], cfg["servo_ids"])
    try:
        q_now = m.to_rad(bus.ticks())
        # 2026-10-22: 휴식자세(접힘)는 모델상 손가락 끝이 책상 아래라 첫 경로가 검사에 걸린다
        # -> 먼저 home 으로 펴고 거기서 계획 (arm_server 와 같은 처리)
        HOME_Q = [0.0, 1.83, -2.121, 0.291]
        start_from_home = float(forward_kinematics(q_now[:4], tip)[2]) < a.min_tip_z
        legs = plan(HOME_Q if start_from_home else q_now[:4], tip, a.tip_z, a.lift,
                    a.high_z or None)
        bad = [(xy, l1, l2) for xy, _, _, l1, l2 in legs if min(l1, l2) < a.min_tip_z]
        print(f"캘리브 지점 {len(legs)}개, 손가락 끝 책상 위 {a.tip_z * 100:.1f}cm")
        for xy, l1, l2 in bad:
            print(f"  ✗ {xy}: 경로 중 손가락 끝 최저 {min(l1, l2) * 100:.1f}cm < {a.min_tip_z * 100:.1f}cm")
        if bad or len(legs) < 4:
            sys.exit("움직이지 않음 — 경로 검사 실패 또는 지점 부족")
        print("경로 검사 통과")
        if a.check:
            return 0

        # 2026-10-22: GUI(control_panel)와 같은 1280x720 MJPG — 해상도가 다르면 화각이 달라 H 가 안 맞는다
        cap = cv2.VideoCapture(a.cam, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            sys.exit(f"카메라 {a.cam} 열기 실패")
        win = "topdown calib - click midpoint of finger tips"
        cv2.namedWindow(win)
        click = {"p": None}
        cv2.setMouseCallback(win, lambda e, x, y, f, p: click.update(p=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

        bus.enable()
        grip = cfg["zero_ticks"][5] + int(0.8 / (2 * math.pi / 4096)) * cfg["signs"][5]
        if start_from_home:
            print("휴식자세에서 시작 — 먼저 home 으로 폅니다")
            t = m.to_ticks(HOME_Q + [0.0, 0.0])
            t[4], t[5] = bus.last_goal[4], grip
            move_smooth(bus, t, max_deg_s=a.speed)
        pix, world = [], []
        pix_all, xyz_all = [], []
        for i, ((x, y, z), q_up, q_dn, _, _) in enumerate(legs):
            for q in (q_up, q_dn):
                target = m.to_ticks(list(q) + [0.0, 0.0])
                target[4] = bus.last_goal[4]                     # 롤은 그대로
                target[5] = grip                                 # 그리퍼 조금 열기 (끝이 잘 보이게)
                move_smooth(bus, target, max_deg_s=a.speed)
            # 2026-10-12: 처짐 보정 — arm_server 도 집을 때 같은 보정을 하므로, 캘리브 지점도
            # 보정된 자세에서 찍어야 클릭 좌표와 실제 집는 위치가 맞는다
            settle_ticks(bus, target)
            time.sleep(0.8)                                      # 흔들림 가라앉기
            click["p"] = None
            print(f"[{i + 1}/{len(legs)}] 손가락 끝 = 로봇 ({x:.2f}, {y:+.2f}, 높이 {z * 100:.0f}cm) — "
                  "화면에서 **두 손가락 끝의 가운데** 클릭")
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
                    if abs(z - a.tip_z) < 1e-6:          # 책상 높이 점만 호모그래피에
                        pix.append(click["p"])
                        world.append((x, y))
                    pix_all.append(click["p"])
                    xyz_all.append((x, y, z))
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
        if os.path.exists(OUT):                  # 기존 파일(촬영용 임시값 등) 보존
            bak = OUT.replace(".yaml", time.strftime("_bak_%Y%m%d_%H%M%S.yaml"))
            os.rename(OUT, bak)
            print(f"기존 파일 백업: {bak}")
        cam = solve_camera(pix_all, xyz_all)
        yaml.safe_dump({**cam, "image_size": [1280, 720],
                        "H": H.tolist(), "reproj_error_mm": float(per.mean()), "loo_error_mm": float(np.mean(loo)),
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
