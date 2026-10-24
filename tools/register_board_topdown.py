#!/usr/bin/env python3
"""탑다운 한 대로 책상 태그 + 클립보드 태그를 팔 A 좌표에 같이 등록 (2026-10-22).

왜 이렇게 하나
--------------
* 안경 씬 카메라로는 책상에 눕힌 태그가 너무 비스듬해(약 20°) 검출이 안 된다 -> 클립보드를
  안경으로 등록하는 길은 막혔다.
* 탑다운 호모그래피는 팔 캘리브점(0.30~0.42m, 8점) 밖을 ~9% 작게 본다. 렌즈 왜곡을 펴도
  그대로다 -> 원인은 좁은 구역 8점으로 원근 항을 못 잡는 것.
* 그래서 탑다운 카메라 모델(K, 왜곡 = tools/topdown_intrinsics.py)을 두고, 아래를 한꺼번에 맞춘다:
    - 탑다운 카메라 자세 (6)
    - 책상 태그 종이 조각마다 책상면 위 (x, y, yaw) — 크기·간격은 인쇄값 그대로
    - 클립보드 시트 자세 (6) — 크기·간격은 인쇄값 그대로
  관측: 팔 A 캘리브점(FK 3D <-> 픽셀) + 태그 모서리 픽셀.
  책상 태그가 먼 구역의 실제 크기를, 팔 A 점이 팔 A 좌표계를 잡아 준다.

    python3 tools/register_board_topdown.py          # 양팔 GUI 를 끄고 (카메라 점유)
결과: ~/.ros/dual_anchor.yaml 의 table_bundle / board_bundle / topdown_camera 갱신 (이전 파일 백업)
"""
import argparse
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register_table_tags import sheet_center_mm  # noqa: E402

ROS = os.path.expanduser("~/.ros")
ANCHOR = os.path.join(ROS, "dual_anchor.yaml")


def page_corners(sheet_ids, tid, size_mm, quiet):
    """태그 모서리의 종이 좌표 (x 오른쪽, y 위, m). 순서 (우상, 좌상, 좌하, 우하)."""
    c = sheet_center_mm(sheet_ids, tid, size_mm, quiet) / 1000.0
    h = size_mm / 2000.0
    return c + np.array([[+h, +h], [-h, +h], [-h, -h], [+h, -h]])


def rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--table-ids", type=int, nargs="+", default=[14, 15, 16, 17])
    ap.add_argument("--table-groups", nargs="+", default=["14,15", "16,17"])
    ap.add_argument("--board-ids", type=int, nargs="+", default=[10, 11, 12, 13])
    ap.add_argument("--quiet-cells", type=float, default=1.0)
    ap.add_argument("--arm-sigma-px", type=float, default=3.0, help="팔 FK 점의 픽셀 불확실도 (태그 모서리는 0.5)")
    ap.add_argument("--table-z", type=float, default=0.0)
    args = ap.parse_args()

    I = yaml.safe_load(open(os.path.join(ROS, "topdown_intrinsics.yaml")))
    K, D = np.array(I["K"], float), np.array(I["dist"], float)
    Ha = yaml.safe_load(open(os.path.join(ROS, "topdown_homography_a.yaml")))
    if [I["width"], I["height"]] != (Ha.get("image_size") or [640, 480]):
        raise SystemExit(f"내부 파라미터 해상도 {I['width']}x{I['height']} != 호모그래피 {Ha.get('image_size')}")
    arm_P = np.array(Ha["points3d"], float)
    arm_px = np.array(Ha["pixels3d"], float)
    anc = yaml.safe_load(open(ANCHOR))
    size_mm = float(anc["tag_size_m"]) * 1000

    # ---- 관측 ----
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, I["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, I["height"])
    det = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=True)
    acc, shape = {}, None
    for k in range(args.frames + 10):
        ok, f = cap.read()
        if not ok:
            continue
        shape = f.shape
        if k < 10:
            continue
        for t in det.detect(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)):
            acc.setdefault(t.tag_id, []).append(np.asarray(t.corners, float))
    cap.release()
    if shape is None or shape[:2] != (I["height"], I["width"]):
        raise SystemExit(f"카메라 해상도가 다르다: {shape} — GUI 가 카메라를 쥐고 있지 않은지 확인")
    obs = {i: np.mean(v, axis=0) for i, v in acc.items() if len(v) >= args.frames // 2}
    groups = [[int(x) for x in g.split(",")] for g in args.table_groups]
    # 조각 안 태그가 하나만 보여도 쓴다 (조각 배치는 인쇄값으로 고정) — 안 보인 태그도 조각에 따라 결과에 넣는다
    all_groups = [g for g in groups if any(i in obs for i in g)]
    groups = all_groups
    board = [i for i in args.board_ids if i in obs]
    print(f"검출: 책상 {[i for i in args.table_ids if i in obs]}, 클립보드 {board}")
    if len(groups) < 2 or len(board) < 3:
        raise SystemExit("책상 두 조각 다 + 클립보드 3장 이상이 보여야 한다")

    # ---- 초기값 ----
    ok, r0, t0 = cv2.solvePnP(arm_P, arm_px, K, D, flags=cv2.SOLVEPNP_SQPNP)
    tb = {b["id"]: b for b in anc["table_bundle"]}
    strip0 = []
    page = {}
    for g in groups:
        pc = np.concatenate([page_corners(args.table_ids, i, size_mm, args.quiet_cells) for i in g])
        page[tuple(g)] = pc - pc.mean(0)          # 조각 원점 = 조각 전체(보인 것+안 보인 것) 중심
        right = np.array(tb[g[0]]["right"][:2])
        c = np.mean([tb[i]["pos"][:2] for i in g], axis=0)
        strip0 += [c[0], c[1], math.atan2(right[1], right[0])]
    bpage = {i: np.c_[page_corners(args.board_ids, i, size_mm, args.quiet_cells), np.zeros(4)] for i in board}
    bo = np.concatenate([bpage[i] for i in board])
    bi = np.concatenate([obs[i] for i in board])
    T_ca = np.eye(4)
    T_ca[:3, :3], T_ca[:3, 3] = cv2.Rodrigues(r0)[0], t0.ravel()
    T_ac = np.linalg.inv(T_ca)
    n, rv, tv, _ = cv2.solvePnPGeneric(bo, bi, K, D, flags=cv2.SOLVEPNP_IPPE)
    best = None
    for r, t in zip(rv, tv):
        T_cb = np.eye(4)
        T_cb[:3, :3], T_cb[:3, 3] = cv2.Rodrigues(r)[0], t.ravel()
        T_ab = T_ac @ T_cb
        zmin = ((T_ab[:3, :3] @ bo.T).T + T_ab[:3, 3])[:, 2].min()
        score = abs(zmin - 0.05) + (0 if T_ab[0, 2] < 0 else 1.0)   # 아래 모서리가 책상 근처 + 팔/착용자 쪽(-x)을 봄
        if best is None or score < score_best:
            best, score_best = T_ab, score
    b0 = np.r_[cv2.Rodrigues(best[:3, :3])[0].ravel(), best[:3, 3]]
    x0 = np.r_[r0.ravel(), t0.ravel(), strip0, b0]

    def unpack(x):
        rc, tc = x[:3], x[3:6]
        strips = x[6:6 + 3 * len(groups)].reshape(-1, 3)
        rb, tb_ = x[6 + 3 * len(groups):9 + 3 * len(groups)], x[9 + 3 * len(groups):]
        return rc, tc, strips, rb, tb_

    def strip_world(s, g):
        return np.c_[(rz(s[2]) @ page[tuple(g)].T).T + s[:2], np.full(4 * len(g), args.table_z)]

    def residuals(x, parts=False):
        rc, tc, strips, rb, tb_ = unpack(x)
        out = {}
        p, _ = cv2.projectPoints(arm_P, rc, tc, K, D)
        out["arm"] = ((p.reshape(-1, 2) - arm_px) / args.arm_sigma_px).ravel()
        res = []
        for s, g in zip(strips, groups):
            W = strip_world(s, g)
            seen = [j for j, i in enumerate(g) if i in obs]
            p, _ = cv2.projectPoints(np.concatenate([W[4 * j:4 * j + 4] for j in seen]), rc, tc, K, D)
            res.append((p.reshape(-1, 2) - np.concatenate([obs[g[j]] for j in seen])) / 0.5)
        out["table"] = np.concatenate(res).ravel()
        Rb = cv2.Rodrigues(rb)[0]
        p, _ = cv2.projectPoints((Rb @ bo.T).T + tb_, rc, tc, K, D)
        out["board"] = ((p.reshape(-1, 2) - bi) / 0.5).ravel()
        return out if parts else np.concatenate(list(out.values()))

    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=2.0)
    parts = residuals(sol.x, parts=True)
    sig = {"arm": args.arm_sigma_px, "table": 0.5, "board": 0.5}
    for k, v in parts.items():
        px = np.linalg.norm(v.reshape(-1, 2), axis=1) * sig[k]
        print(f"[맞춤] {k:5s}: 재투영 평균 {px.mean():.2f}px 최대 {px.max():.2f}px ({len(px)}점)")
    rc, tc, strips, rb, tb_ = unpack(sol.x)
    Rc = cv2.Rodrigues(rc)[0]
    C = -Rc.T @ tc
    print(f"탑다운 카메라 (팔 A 기준): {np.round(C, 3).tolist()}")

    # ---- 결과 ----
    hm = size_mm / 2000.0
    table_bundle = []
    for s, g in zip(strips, groups):
        W = strip_world(s, g)
        R2 = rz(s[2])
        for j, i in enumerate(g):
            c = W[4 * j:4 * j + 4].mean(0)
            table_bundle.append({"id": i, "pos": [round(float(v), 4) for v in c],
                                 "right": [round(float(v), 4) for v in np.r_[R2[:, 0], 0]],
                                 "up": [round(float(v), 4) for v in np.r_[R2[:, 1], 0]]})
            print(f"   책상 {i}: ({c[0]:.3f}, {c[1]:+.3f})  rA={math.hypot(c[0], c[1]):.3f}"
                  f"{'' if i in obs else '  (안 보임 — 조각 배치로 계산)'}")
    Rb = cv2.Rodrigues(rb)[0]
    board_bundle = []
    for i in args.board_ids:
        c = Rb @ np.r_[sheet_center_mm(args.board_ids, i, size_mm, args.quiet_cells) / 1000, 0] + tb_
        board_bundle.append({"id": i, "pos": [round(float(v), 4) for v in c],
                             "right": [round(float(v), 4) for v in Rb[:, 0]],
                             "up": [round(float(v), 4) for v in Rb[:, 1]]})
        print(f"   클립보드 {i}: ({c[0]:.3f}, {c[1]:+.3f}, {c[2]:.3f})")
    normal = Rb[:, 2]
    tilt = math.degrees(math.acos(abs(normal[2])))
    zmin = ((Rb @ bo.T).T + tb_)[:, 2].min()
    print(f"   클립보드 기울기 {tilt:.0f}° (90=수직), 법선 {np.round(normal, 2).tolist()}, "
          f"태그 가장 낮은 모서리 {zmin * 100:.1f}cm")

    bak = ANCHOR.replace(".yaml", time.strftime("_bak_%H%M%S.yaml"))
    shutil.copy2(ANCHOR, bak)
    anc["table_bundle"] = table_bundle
    anc["board_bundle"] = board_bundle
    anc["topdown_camera"] = {"center": C.tolist(), "rvec": rc.tolist(), "tvec": tc.tolist()}
    anc["board_registration"] = {"method": "topdown", "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "tilt_deg": tilt, "zmin_m": float(zmin),
                                 **{f"reproj_{k}_px": float(np.linalg.norm(v.reshape(-1, 2), axis=1).mean() * sig[k])
                                    for k, v in parts.items()}}
    with open(ANCHOR, "w") as f:
        yaml.safe_dump(anc, f, sort_keys=False, allow_unicode=True)
    print(f"[저장] {ANCHOR} (이전 {bak})")


if __name__ == "__main__":
    main()
