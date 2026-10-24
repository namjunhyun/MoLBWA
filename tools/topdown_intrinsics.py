#!/usr/bin/env python3
"""탑다운 카메라 내부 파라미터(K, 왜곡) 캘리브 — 체커보드 자동 수집 (2026-10-22).

왜 필요한가
-----------
탑다운 호모그래피는 팔이 닿는 0.30~0.42m 에서만 점을 찍었다. 렌즈 왜곡을 안 펴고
그 바깥(0.6m 대 태그)으로 외삽하면 거리가 ~20% 짧게 나왔다(인쇄 간격 95.5mm -> 75~80mm).
왜곡을 펴면 호모그래피가 화면 전체에서 맞는다. 저장된 캘리브 점 픽셀을 다시 쓰므로 팔은 안 움직인다.

사용:
    python3 tools/topdown_intrinsics.py            # 창에서 체커보드를 여기저기 비추면 자동 수집
    python3 tools/topdown_intrinsics.py --board tags   # 체커보드 대신 태그 시트(클립보드 10~13)를 들고 움직임
    키: space=강제 수집, c=계산·저장, u=왜곡 편 화면 보기, q=종료
결과: ~/.ros/topdown_intrinsics.yaml
★ 해상도는 GUI(control_panel)와 같은 1280x720 MJPG (2026-10-22 전환). 640x480 은 화각이 다르다.
"""
import argparse
import os
import time

import cv2
import numpy as np
import yaml

GRID = 3          # 화면을 3x3 칸으로 나눠 골고루 모였는지 본다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cols", type=int, default=9, help="내부 코너 가로")
    ap.add_argument("--rows", type=int, default=6, help="내부 코너 세로")
    ap.add_argument("--square-mm", type=float, default=25.0)
    # 2026-10-22: 체커보드 대신 인쇄한 AprilTag 시트(4장)로도 된다 — 시트를 들고 움직이면 같은 원리
    ap.add_argument("--board", choices=["chess", "tags"], default="chess")
    ap.add_argument("--tag-ids", type=int, nargs="+", default=[10, 11, 12, 13], help="시트 인쇄 순서")
    ap.add_argument("--tag-size-mm", type=float, default=70.0)
    ap.add_argument("--quiet-cells", type=float, default=1.0)
    ap.add_argument("--target", type=int, default=25, help="이만큼 모이면 자동 계산")
    ap.add_argument("--out", default=os.path.expanduser("~/.ros/topdown_intrinsics.yaml"))
    ap.add_argument("--save-dir", default=os.path.expanduser(
        "~/MoLBWA-gaze-hri/notes/topdown_intrinsics"))
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm / 1000.0

    if args.board == "tags":
        import sys
        from pupil_apriltags import Detector
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from register_table_tags import sheet_center_mm
        tag_det = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=True)
        hm = args.tag_size_mm / 2000.0
        local = np.array([[+hm, +hm, 0], [-hm, +hm, 0], [-hm, -hm, 0], [+hm, -hm, 0]])   # (우상, 좌상, 좌하, 우하)
        tag_obj = {i: np.r_[sheet_center_mm(args.tag_ids, i, args.tag_size_mm, args.quiet_cells) / 1000, 0] + local
                   for i in args.tag_ids}
    obj_pts = []

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # control_panel 과 같은 모드 (1280x720 MJPG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    img_pts, feats = [], []
    cover = np.zeros((GRID, GRID), int)
    K = dist = None
    show_undist = False
    last_t = 0.0
    size = None

    def feature(c, shape, obj=None):
        # 위치(중심), 크기(대각선), 기울기(가로/세로 변 길이 비) — 같은 자세 중복 수집 방지
        c = c.reshape(-1, 2)
        if obj is not None:
            # 태그 시트: 종이 -> 화면 호모그래피의 원근 항으로 기울기를 본다
            Hs, _ = cv2.findHomography(obj[:, :2], c)
            ctr = c.mean(0) / [shape[1], shape[0]]
            diag = np.linalg.norm(c.max(0) - c.min(0)) / np.hypot(*shape[:2])
            return np.array([ctr[0], ctr[1], diag, Hs[2, 0] / Hs[2, 2] * 0.3, Hs[2, 1] / Hs[2, 2] * 0.3])
        ctr = c.mean(0) / [shape[1], shape[0]]
        diag = np.linalg.norm(c[0] - c[-1]) / np.hypot(*shape[:2])
        top = np.linalg.norm(c[0] - c[args.cols - 1])
        bot = np.linalg.norm(c[-args.cols] - c[-1])
        left = np.linalg.norm(c[0] - c[-args.cols])
        right = np.linalg.norm(c[args.cols - 1] - c[-1])
        return np.array([ctr[0], ctr[1], diag, top / bot - 1, left / right - 1])

    def calibrate():
        nonlocal K, dist
        if len(img_pts) < 10:
            print(f"[계산] {len(img_pts)}장 — 10장 이상 필요")
            return
        objs = obj_pts if args.board == "tags" else [objp] * len(img_pts)
        # 태그 시트는 장마다 모서리가 12~16개뿐이라 k3 까지 풀면 불안정 -> 고정
        flags = cv2.CALIB_FIX_K3 if args.board == "tags" else 0
        rms, K, dist, _, _ = cv2.calibrateCamera(objs, img_pts, size, None, None, flags=flags)
        print(f"[계산] {len(img_pts)}장, RMS {rms:.3f}px")
        print("  K =", np.round(K, 1).tolist())
        print("  dist =", np.round(dist.ravel(), 4).tolist())
        with open(args.out, "w") as f:
            yaml.safe_dump({"width": size[0], "height": size[1], "K": K.tolist(),
                            "dist": dist.ravel().tolist(), "rms_px": float(rms),
                            "num_images": len(img_pts),
                            "stamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        print(f"[저장] {args.out}")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        obj = None
        if args.board == "tags":
            dets = [d for d in tag_det.detect(gray) if d.tag_id in tag_obj]
            found = len(dets) >= 3          # 3장 이상 = 모서리 12개 이상, 한 줄로 안 늘어섬
            for d in dets:
                cv2.polylines(vis, [d.corners.astype(int)], True, (0, 255, 0), 2)
            if found:
                corners = np.concatenate([d.corners for d in dets]).astype(np.float32).reshape(-1, 1, 2)
                obj = np.concatenate([tag_obj[d.tag_id] for d in dets]).astype(np.float32)
        else:
            found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE)
        key = cv2.waitKey(1) & 0xFF
        if found:
            if args.board == "chess":
                cv2.drawChessboardCorners(vis, pattern, corners, found)
            f = feature(corners, frame.shape, obj)
            novel = not feats or min(np.linalg.norm(f - g) for g in feats) > 0.08
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var() > 30
            if (key == ord(" ") or (novel and sharp and time.time() - last_t > 1.0)):
                img_pts.append(corners.astype(np.float32))
                if obj is not None:
                    obj_pts.append(obj)
                feats.append(f)
                gx = min(GRID - 1, int(f[0] * GRID))
                gy = min(GRID - 1, int(f[1] * GRID))
                cover[gy, gx] += 1
                cv2.imwrite(os.path.join(args.save_dir, f"cb_{len(img_pts):02d}.png"), frame)
                last_t = time.time()
                print(f"[수집] {len(img_pts)}장 (칸 {gy},{gx})")
                if len(img_pts) == args.target:
                    calibrate()
        if show_undist and K is not None:
            vis = cv2.undistort(vis, K, dist)
        # 3x3 수집 현황 — 빈 칸(빨강)으로 보드를 옮길 것
        h, w = vis.shape[:2]
        for gy in range(GRID):
            for gx in range(GRID):
                x0, y0 = gx * w // GRID, gy * h // GRID
                col = (0, 200, 0) if cover[gy, gx] >= 2 else (0, 0, 255)
                cv2.rectangle(vis, (x0 + 2, y0 + 2), (x0 + w // GRID - 2, y0 + h // GRID - 2), col, 1)
                cv2.putText(vis, str(cover[gy, gx]), (x0 + 8, y0 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(vis, f"{len(img_pts)}/{args.target}  {'FOUND' if found else '-'}"
                    f"  space:grab c:calib u:undist q:quit", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("topdown intrinsics", vis)
        if key == ord("c"):
            calibrate()
        elif key == ord("u"):
            show_undist = not show_undist
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
