#!/usr/bin/env python3
"""책상에 눕힌 AprilTag 를 탑다운 카메라로 팔 A 좌표에 등록 + 팔 B 받침대의 팔 A 기준 위치 (2026-10-22).

양팔 시선 캘리브의 기준 좌표는 팔 A base_link 다 (dual.launch.py 와 같음).
  * 책상 태그: 탑다운에서 모서리 4개를 검출 -> 팔 A 호모그래피 -> 책상면 위 모서리 좌표.
    모서리로 중심(pos)과 태그 축(right/up)을 같이 구하므로 자로 잴 것이 없다.
  * 팔 B: 작업 구역 픽셀을 팔 A/B 호모그래피로 각각 풀어 2D 강체변환(B -> A)을 맞춘다.
    두 팔의 탑다운 캘리브가 서로 맞는지도 이 잔차로 드러난다.

★ 해상도는 호모그래피 파일의 image_size 를 따른다 (2026-10-22 이후 1280x720 MJPG, 없으면 640x480 YUYV).
  해상도가 다르면 화각이 달라 좌표가 틀린다 (2026-10-22 이걸 틀려 거리가 20cm 짧게 나왔었다).
  양팔 GUI 를 끄고 실행할 것 (카메라 점유).

    python3 tools/register_table_tags.py              # 태그 14~17
결과: ~/.ros/dual_anchor.yaml (bundle 항목 = arm/config.yaml anchor.bundle 형식, + arm_b_in_a)
"""
import argparse
import math
import os
import time

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector


def load_H(arm):
    d = yaml.safe_load(open(os.path.expanduser(f"~/.ros/topdown_homography_{arm}.yaml")))
    return np.array(d["H"], float), d


def apply_H(H, uv):
    uv = np.atleast_2d(uv)
    p = (H @ np.c_[uv, np.ones(len(uv))].T).T
    return p[:, :2] / p[:, 2:3]


def sheet_center_mm(sheet_ids, tid, size_mm, quiet_cells):
    """tools/make_apriltag_sheet.py 배치 그대로 — 태그 중심의 종이 좌표 (오른쪽 +x, 위 +y, mm)."""
    k = sheet_ids.index(tid)
    block = size_mm + 2 * max(1.0, quiet_cells) * size_mm / 8
    cols = 2 if len(sheet_ids) > 1 else 1
    gap = (210.0 - block * cols) / (cols + 1)
    r, c = divmod(k, cols)
    return np.array([gap + c * (block + gap) + block / 2, -(22.0 + r * (block + 8.0) + block / 2)])


def rigid_2d(src, dst):
    """dst ≈ R @ src + t (Kabsch, 2D)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--ids", type=int, nargs="+", default=[14, 15, 16, 17])
    ap.add_argument("--tag-size-mm", type=float, default=70.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--table-z", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.expanduser("~/.ros/dual_anchor.yaml"))
    # 같은 종이 조각에 붙어 있는 태그 묶음. 묶음은 인쇄 배치(간격·크기)를 그대로 믿고 위치·방향만 맞춘다.
    # 탑다운 호모그래피는 캘리브점(0.30~0.42m) 밖을 ~9% 작게 봐서, 태그를 따로 재면 묶음 안 간격이 틀어진다.
    ap.add_argument("--sheet-ids", type=int, nargs="+", default=[14, 15, 16, 17],
                    help="인쇄한 시트의 id 순서 (make_apriltag_sheet --ids 와 같게)")
    ap.add_argument("--quiet-cells", type=float, default=1.0, help="인쇄 때 --quiet-cells")
    ap.add_argument("--groups", nargs="*", default=["14,15", "16,17"],
                    help="같은 종이 조각의 id 묶음 (쉼표). 비우면 태그마다 따로")
    args = ap.parse_args()

    Ha, da = load_H("a")
    Hb, db = load_H("b")

    w, h = da.get("image_size") or [640, 480]
    if (db.get("image_size") or [640, 480]) != [w, h]:
        raise SystemExit(f"팔 a/b 호모그래피 해상도가 다르다: {da.get('image_size')} / {db.get('image_size')}")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*("MJPG" if w > 640 else "YUYV")))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    det = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=True)
    acc = {i: [] for i in args.ids}
    shape = None
    for k in range(args.frames + 10):
        ok, f = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        shape = f.shape
        if k < 10:          # 노출 안정
            continue
        for t in det.detect(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)):
            if t.tag_id in acc:
                acc[t.tag_id].append(np.asarray(t.corners, float))
    cap.release()
    if shape is None or shape[:2] != (h, w):
        raise SystemExit(f"카메라 해상도가 {w}x{h} 가 아니다: {shape} — GUI 가 카메라를 쥐고 있지 않은지 확인")

    bundle, report = [], []
    for i in args.ids:
        n = len(acc[i])
        if n < args.frames // 3:
            print(f"[경고] 태그 {i}: {n}/{args.frames} 프레임만 검출 — 제외")
            continue
        c_px = np.mean(acc[i], axis=0)                 # (우상, 좌상, 좌하, 우하)
        c_xy = apply_H(Ha, c_px)
        right = (c_xy[0] + c_xy[3]) / 2 - (c_xy[1] + c_xy[2]) / 2
        up = (c_xy[0] + c_xy[1]) / 2 - (c_xy[3] + c_xy[2]) / 2
        side = float(np.mean([np.linalg.norm(c_xy[j] - c_xy[(j + 1) % 4]) for j in range(4)]))
        r = np.r_[right / np.linalg.norm(right), 0.0]
        u = np.r_[up / np.linalg.norm(up), 0.0]
        u = u - (u @ r) * r                           # 직교화 (build_bundle_obj_pts 가 1e-6 로 검사)
        u /= np.linalg.norm(u)
        skew = math.degrees(math.acos(np.clip(abs(right @ up) / (np.linalg.norm(right) * np.linalg.norm(up)), 0, 1)))
        pos = np.r_[c_xy.mean(0), args.table_z]
        normal = np.cross(r, u)
        if normal[2] < 0:
            print(f"[경고] 태그 {i}: 법선이 아래(-z) — 모서리 순서/거울 확인")
        bundle.append({"id": int(i), "pos": [round(float(v), 4) for v in pos],
                       "right": [round(float(v), 4) for v in r], "up": [round(float(v), 4) for v in u]})
        pb = apply_H(Hb, c_px.mean(0, keepdims=True))[0]
        report.append((i, n, pos, math.hypot(*pos[:2]), pb, math.hypot(*pb), side, 90 - skew))

    # 종이 조각 단위 강체 맞춤 — 인쇄 배치로 만든 모서리(종이 좌표)를 탑다운 모서리에 회전+이동만으로 맞춘다
    meas = {i: apply_H(Ha, np.mean(acc[i], axis=0)) for i in args.ids if len(acc[i]) >= args.frames // 3}
    hm = args.tag_size_mm / 2000.0
    local = np.array([[+hm, +hm], [-hm, +hm], [-hm, -hm], [+hm, -hm]])   # (우상, 좌상, 좌하, 우하)
    group_report = []
    for g in args.groups:
        ids = [int(x) for x in g.split(",")]
        if any(i not in meas for i in ids):
            print(f"[경고] 묶음 {ids}: 검출 안 된 태그가 있어 태그별 값 유지")
            continue
        src = np.concatenate([sheet_center_mm(args.sheet_ids, i, args.tag_size_mm, args.quiet_cells) / 1000.0
                              + local for i in ids])
        dst = np.concatenate([meas[i] for i in ids])
        R, t = rigid_2d(src, dst)
        if np.linalg.det(R) < 0:
            print(f"[경고] 묶음 {ids}: 거울상으로 맞춰짐 — 묶음/시트 id 순서 확인")
        fit = (R @ src.T).T + t
        res = np.linalg.norm(fit - dst, axis=1)
        # 측정이 인쇄보다 얼마나 작게/크게 보이는지 (참고용 — 맞춤엔 안 씀)
        scale = np.sqrt(((dst - dst.mean(0)) ** 2).sum() / ((src - src.mean(0)) ** 2).sum())
        for j, i in enumerate(ids):
            c = fit[4 * j:4 * j + 4].mean(0)
            old = next(b for b in bundle if b["id"] == i)
            shift = np.linalg.norm(c - np.array(old["pos"][:2]))
            old["pos"] = [round(float(c[0]), 4), round(float(c[1]), 4), args.table_z]
            old["right"] = [round(float(v), 4) for v in np.r_[R[:, 0], 0.0]]
            old["up"] = [round(float(v), 4) for v in np.r_[R[:, 1], 0.0]]
            group_report.append((i, c, shift))
        print(f"[묶음] {ids}: 모서리 잔차 평균 {res.mean() * 1000:.1f}mm / 최대 {res.max() * 1000:.1f}mm, "
              f"탑다운이 본 크기 = 인쇄의 {scale * 100:.0f}%")
    for i, c, shift in group_report:
        print(f"   태그 {i}: 중심 ({c[0]:.3f}, {c[1]:+.3f}) rA={math.hypot(*c):.3f}  (태그별 값에서 {shift * 1000:.0f}mm 이동)")

    # 팔 B -> 팔 A : 두 팔 캘리브 점이 있는 작업 구역 픽셀로 맞춘다 (외삽 구간은 쓰지 않음)
    px = np.array(da["pixels"] + db["pixels"], float)
    u0, v0 = px.min(0)
    u1, v1 = px.max(0)
    gu, gv = np.meshgrid(np.linspace(u0, u1, 12), np.linspace(v0, v1, 8))
    grid = np.c_[gu.ravel(), gv.ravel()]
    pa, pb = apply_H(Ha, grid), apply_H(Hb, grid)
    R, t = rigid_2d(pb, pa)
    res = np.linalg.norm((R @ pb.T).T + t - pa, axis=1)
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    print("\n태그 (팔 A 기준, 책상면)")
    print(" id  검출   A(x, y)            rA     B(x, y)            rB     한변(인쇄 %.0fmm)  직각" % args.tag_size_mm)
    for i, n, pos, ra, pb_, rb, side, sq in report:
        print(f" {i:2d}  {n:3d}   ({pos[0]:.3f}, {pos[1]:+.3f})   {ra:.3f}  ({pb_[0]:.3f}, {pb_[1]:+.3f})   {rb:.3f}"
              f"  {side * 1000:5.1f}mm         {sq:+.1f}°")
    print(f"\n팔 B 받침대 (팔 A 기준): x={t[0]:.3f} y={t[1]:+.3f} m, yaw={yaw:+.1f}°, "
          f"맞춤 잔차 평균 {res.mean() * 1000:.1f}mm / 최대 {res.max() * 1000:.1f}mm (작업 구역 96점)")

    out = {"stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "frame": "arm_a base_link",
           "tag_size_m": args.tag_size_mm / 1000.0,
           "table_bundle": bundle,
           "arm_b_in_a": {"x": float(t[0]), "y": float(t[1]), "yaw_deg": float(yaw),
                          "fit_mean_mm": float(res.mean() * 1000), "fit_max_mm": float(res.max() * 1000)}}
    old = {}
    if os.path.exists(args.out):
        old = yaml.safe_load(open(args.out)) or {}
    old.update(out)              # 클립보드 태그(board_bundle) 등 다른 키는 보존
    with open(args.out, "w") as f:
        yaml.safe_dump(old, f, sort_keys=False, allow_unicode=True)
    print(f"[저장] {args.out}")


if __name__ == "__main__":
    main()
