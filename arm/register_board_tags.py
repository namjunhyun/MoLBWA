#!/usr/bin/env python3
"""클립보드(세운 판) 태그를 안경 씬 카메라로 팔 A 좌표에 등록 (2026-10-22).

책상 태그(14~17)는 tools/register_table_tags.py 가 탑다운으로 이미 팔 A 좌표에 올려 놨다
(~/.ros/dual_anchor.yaml 의 table_bundle). 클립보드는 책상 위로 서 있어서 탑다운 호모그래피
(책상면 전용)로는 못 잰다. 그래서 씬 카메라가 두 묶음을 **같은 프레임에서** 볼 때

    T_A_board = T_A_cam(책상 태그 PnP) @ T_cam_board(클립보드 PnP)

를 모은다. 두 묶음 다 평면이라 PnP 해가 각각 2개(거울상)씩 나온다 — 한 프레임으로는 못 고른다.
머리를 움직이면 옳은 조합의 T_A_board 만 프레임마다 같은 값으로 모이고, 거울상 조합은
시점마다 흩어진다. 그래서 프레임 전체에서 가장 많이 겹치는 후보를 고른다.
등록이 끝나면 책상+클립보드 8장이 서로 다른 두 평면이라 이후 PnP 는 거울상 문제가 없다.

    python3 arm/register_board_tags.py        # 안경 쓰고 책상 태그+클립보드가 같이 보이게 머리를 여러 각도로
    키: q = 그만 모으고 계산

결과: ~/.ros/dual_anchor.yaml 에 board_bundle(arm/config.yaml anchor.bundle 형식) 추가,
      수집 원자료 notes/board_reg/board_<시각>.npz
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import rclpy
import yaml
from pupil_apriltags import Detector
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from anchor import build_bundle_obj_pts  # noqa: E402
from register_table_tags import sheet_center_mm  # noqa: E402
from run_demo import load_intrinsics  # noqa: E402

ANCHOR = os.path.expanduser("~/.ros/dual_anchor.yaml")


def rt(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
    T[:3, 3] = np.asarray(tvec, float).ravel()
    return T


def rot_deg(Ra, Rb):
    return math.degrees(math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def ippe(obj, img, K):
    """평면 PnP 두 해 [(T_cam_obj, 재투영px)]. 카메라 뒤에 놓이는 해는 버린다."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    out = []
    for r, t, e in zip(rvecs, tvecs, np.ravel(errs) if errs is not None else [0.0] * n):
        T = rt(r, t)
        if ((T[:3, :3] @ obj.T).T + T[:3, 3])[:, 2].min() > 0:
            out.append((T, float(e)))
    return out


class Collector(Node):
    def __init__(self, args):
        super().__init__("register_board_tags")
        self.args = args
        self.K = np.asarray(load_intrinsics(), float)
        d = yaml.safe_load(open(ANCHOR))
        size = float(d["tag_size_m"])
        self.table_obj, *_ = build_bundle_obj_pts({"tag_size_m": size, "bundle": d["table_bundle"]})
        # 클립보드 시트: 종이 좌표계 (x=오른쪽, y=위, z=종이에서 나오는 쪽), 원점=종이 왼쪽 위 모서리
        h = size / 2
        local = np.array([[+h, +h, 0], [-h, +h, 0], [-h, -h, 0], [+h, -h, 0]])   # (우상, 좌상, 좌하, 우하)
        self.board_obj = {i: np.r_[sheet_center_mm(args.board_ids, i, size * 1000, args.quiet_cells) / 1000, 0]
                          + local for i in args.board_ids}
        # 책상 두 조각이 다 보여야 책상 PnP 가 넓게 잡힌다 (한 조각 두 장은 9cm 라 좁다)
        self.groups = [set(int(x) for x in g.split(",")) for g in args.table_groups]
        self.det = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=True)
        self.frames = []          # [(table_dets, board_dets)] — 모서리 원자료
        self.cands = []           # 프레임마다 [(T_A_board, 재투영합)]
        self.n_seen = 0
        self.last = None
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CompressedImage, args.scene_topic, self.on_scene, qos)
        self.get_logger().info(f"씬 구독 {args.scene_topic} — 책상 태그와 클립보드가 같이 보이게 머리를 여러 각도로")

    def on_scene(self, msg):
        f = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if f is None:
            return
        self.n_seen += 1
        dets = self.det.detect(f)
        tab = {d.tag_id: np.asarray(d.corners, float) for d in dets if d.tag_id in self.table_obj}
        brd = {d.tag_id: np.asarray(d.corners, float) for d in dets if d.tag_id in self.board_obj}
        ok = (len(tab) >= 3 and all(g & set(tab) for g in self.groups) and len(brd) >= 2)
        if ok:
            c = self.solve(tab, brd)
            if c:
                self.frames.append((tab, brd))
                self.cands.append(c)
        vis = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        for d in dets:
            col = (0, 200, 0) if d.tag_id in self.table_obj else (255, 120, 0) if d.tag_id in self.board_obj else (0, 0, 255)
            cv2.polylines(vis, [d.corners.astype(int)], True, col, 2)
            cv2.putText(vis, str(d.tag_id), tuple(d.center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(vis, f"used {len(self.frames)}/{self.args.target}  table {sorted(tab)} board {sorted(brd)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(vis, "q: compute", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        self.last = vis

    def solve(self, tab, brd):
        to = np.concatenate([self.table_obj[i] for i in tab])
        ti = np.concatenate([tab[i] for i in tab])
        bo = np.concatenate([self.board_obj[i] for i in brd])
        bi = np.concatenate([brd[i] for i in brd])
        out = []
        for T_ca, ea in ippe(to, ti, self.K):
            if ea > self.args.max_reproj_px:
                continue
            T_ac = np.linalg.inv(T_ca)
            if T_ac[2, 3] < 0.1:          # 카메라(머리)가 책상 아래 — 물리적으로 불가능
                continue
            for T_cb, eb in ippe(bo, bi, self.K):
                if eb > self.args.max_reproj_px:
                    continue
                out.append((T_ac @ T_cb, ea + eb))
        return out


def choose(cands, pos_tol, rot_tol):
    """프레임 대부분이 동의하는 T_A_board. 거울상 조합은 머리 각도마다 다른 데로 흩어진다."""
    flat = [(k, T) for k, c in enumerate(cands) for T, _ in c]
    best, best_sup = None, -1
    for _, T in flat:
        sup = sum(any(np.linalg.norm(T2[:3, 3] - T[:3, 3]) < pos_tol and rot_deg(T[:3, :3], T2[:3, :3]) < rot_tol
                      for T2, _ in c) for c in cands)
        if sup > best_sup:
            best, best_sup = T, sup
    # 지지하는 프레임마다 가장 가까운 후보로 평균
    ts, Rs = [], []
    for c in cands:
        near = [T2 for T2, _ in c if np.linalg.norm(T2[:3, 3] - best[:3, 3]) < pos_tol
                and rot_deg(best[:3, :3], T2[:3, :3]) < rot_tol]
        if near:
            T2 = min(near, key=lambda T2: np.linalg.norm(T2[:3, 3] - best[:3, 3]))
            ts.append(T2[:3, 3])
            Rs.append(T2[:3, :3])
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.mean(ts, axis=0)
    spread_t = float(np.std(ts, axis=0).max())
    spread_r = float(np.std([rot_deg(R, R2) for R2 in Rs]))
    return T, best_sup, spread_t, spread_r


def validate(node, T_ab, board_ids):
    """책상+클립보드 8장(두 평면)으로 프레임마다 PnP — 재투영 오차와 거울상 해 간격."""
    obj_all = dict(node.table_obj)
    for i in board_ids:
        P = node.board_obj[i]
        obj_all[i] = (T_ab[:3, :3] @ P.T).T + T_ab[:3, 3]
    errs, amb = [], []
    for tab, brd in node.frames:
        ids = list(tab) + list(brd)
        obj = np.concatenate([obj_all[i] for i in ids])
        img = np.concatenate([tab.get(i, brd.get(i)) for i in ids])
        ok, r, t = cv2.solvePnP(obj, img, node.K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        r, t = cv2.solvePnPRefineLM(obj, img, node.K, None, r, t)
        p, _ = cv2.projectPoints(obj, r, t, node.K, None)
        errs.append(float(np.sqrt(((p.reshape(-1, 2) - img) ** 2).sum(1).mean())))
    return np.array(errs), obj_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-topic", default="/pc/camera/left/compressed")
    ap.add_argument("--board-ids", type=int, nargs="+", default=[10, 11, 12, 13],
                    help="클립보드 시트 id (인쇄 순서)")
    ap.add_argument("--quiet-cells", type=float, default=1.0)
    ap.add_argument("--table-groups", nargs="+", default=["14,15", "16,17"])
    ap.add_argument("--target", type=int, default=150, help="이만큼 모이면 자동 계산")
    ap.add_argument("--max-reproj-px", type=float, default=2.0)
    ap.add_argument("--pos-tol", type=float, default=0.02)
    ap.add_argument("--rot-tol", type=float, default=3.0)
    args = ap.parse_args()

    rclpy.init()
    node = Collector(args)
    try:
        while rclpy.ok() and len(node.frames) < args.target:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.last is not None:
                cv2.imshow("register board tags", node.last)
                node.last = None
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    n = len(node.frames)
    print(f"\n받은 프레임 {node.n_seen}, 쓸 수 있는 프레임 {n}")
    if n < 20:
        print("20 프레임 이상 필요 — 책상 양쪽 조각 + 클립보드가 같이 보이게 다시")
        return 1

    T_ab, sup, sd_t, sd_r = choose(node.cands, args.pos_tol, args.rot_tol)
    R, t = T_ab[:3, :3], T_ab[:3, 3]
    normal = R[:, 2]                       # 종이에서 나오는 쪽
    tilt = math.degrees(math.acos(abs(normal[2])))   # 0 = 책상에 누움, 90 = 똑바로 섬
    print(f"[선택] 지지 프레임 {sup}/{n} ({100 * sup / n:.0f}%), 흩어짐 위치 {sd_t * 1000:.1f}mm / 회전 {sd_r:.1f}°")
    if sup < 0.6 * n:
        print("[경고] 지지가 60% 미만 — 머리 각도를 더 다양하게 해서 다시 할 것")

    board_bundle = []
    for i in args.board_ids:
        c = R @ node.board_obj[i].mean(0) + t
        board_bundle.append({"id": int(i), "pos": [round(float(v), 4) for v in c],
                             "right": [round(float(v), 4) for v in R[:, 0]],
                             "up": [round(float(v), 4) for v in R[:, 1]]})
        print(f"   태그 {i}: 중심 ({c[0]:.3f}, {c[1]:+.3f}, {c[2]:.3f}) m")
    print(f"   판 기울기 {tilt:.0f}° (90=수직), 법선 {np.round(normal, 2).tolist()} "
          f"({'팔/착용자 쪽(-x)' if normal[0] < 0 else '바깥(+x)'}을 봄)")
    zmin = min(((R @ P.T).T + t)[:, 2].min() for P in node.board_obj.values())
    print(f"   클립보드 태그 가장 낮은 모서리 높이 {zmin * 100:.1f}cm (책상 아래면 틀린 해)")

    errs, _ = validate(node, T_ab, args.board_ids)
    print(f"[검증] 8장 PnP 재투영 오차: 중앙값 {np.median(errs):.2f}px, 90% {np.percentile(errs, 90):.2f}px, "
          f"최대 {errs.max():.2f}px (TagBundleDetector 상한 3px)")

    d = yaml.safe_load(open(ANCHOR))
    d["board_bundle"] = board_bundle
    d["board_registration"] = {"stamp": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": n, "support": int(sup),
                               "spread_mm": sd_t * 1000, "spread_deg": sd_r, "tilt_deg": tilt,
                               "reproj_median_px": float(np.median(errs)),
                               "reproj_p90_px": float(np.percentile(errs, 90))}
    with open(ANCHOR, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)
    out_dir = os.path.join(HERE, "..", "notes", "board_reg")
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, time.strftime("board_%m%d_%H%M%S.npz"))
    np.savez(raw, frames=np.array(node.frames, dtype=object), T_ab=T_ab, K=node.K)
    print(f"[저장] {ANCHOR} (board_bundle), 원자료 {raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
