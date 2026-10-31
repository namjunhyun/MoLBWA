"""G1 부위 YOLO 합성 데이터: MuJoCo 세그멘테이션 렌더로 g1_face/g1_hand/g1_torso 박스를 자동 라벨.

    python3 synth_parts.py --lib ~/molbwa_g1/lib5/molbwa_library5.csv --out DIR --n 30 [--seed 0] [--preview grid.jpg]

- 카메라: 방위 0~360°, 고도 −15~35°(바닥 위 0.15 m 이상), 거리 1.0~3.5 m, 시선점 흔들림 → 정면·측면·후면·위아래 다양
- 자세: 라이브러리(대기·회전·손 흔들기·악수) 무작위 프레임
- 클래스: 0 g1_face(head_link) 1 g1_hand(wrist_yaw_link, 실기는 손 없는 목업이라 rubber_hand 숨김) 2 g1_torso(torso·logo·waist_support)
- 가려진 부위는 보이는 픽셀만으로 박스(가림 반영), 40 px² 미만은 버림
한계: 배경이 MuJoCo 바닥·하늘이라 실제 행사장과 다르다 → 실사 배경 합성·실사 소량 라벨로 보정 필요.
"""
import argparse, os, sys
import numpy as np
import mujoco as mj
import imageio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motion"))
from make_idle_csv import CSV_JOINTS  # noqa: E402

XML = os.environ.get("G1_MUJOCO_XML", os.path.expanduser("~/unitree_g1_vibes/RL-shenanigans/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"))
CLASSES = ["g1_face", "g1_hand", "g1_torso"]
MESH_CLASS = {"head_link": 0, "left_wrist_yaw_link": 1, "right_wrist_yaw_link": 1,
              "torso_link": 2, "logo_link": 2, "waist_support_link": 2}
# 같은 클래스라도 따로 박스를 그릴 인스턴스(왼손·오른손)
MESH_INST = {"left_wrist_yaw_link": "L", "right_wrist_yaw_link": "R"}


def main_run(counts: np.ndarray, gap: int = 3):
    """1차원 픽셀 수에서 (gap 이하 빈칸은 이어 붙인) 연속 구간 중 픽셀이 가장 많은 구간 [a, b)."""
    idx = np.where(counts > 0)[0]
    if len(idx) == 0:
        return None
    runs, a = [], idx[0]
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] > gap + 1:
            runs.append((a, idx[i - 1] + 1)); a = idx[i]
    runs.append((a, idx[-1] + 1))
    return max(runs, key=lambda r: counts[r[0]:r[1]].sum())


def robust_box(mask: np.ndarray):
    """동떨어진 조각(다른 부위 뒤로 비친 몇 픽셀 등)을 버리고 주 덩어리의 박스. 2026-10-10: 머리 픽셀 16개가
    몸통 위치에 떨어져 있어 얼굴 박스가 허리까지 늘어났던 문제."""
    r = main_run(mask.sum(1))
    if r is None:
        return None
    sub = mask[r[0]:r[1]]
    c = main_run(sub.sum(0))
    sub = sub[:, c[0]:c[1]]
    r2 = main_run(sub.sum(1))
    return c[0], c[1], r[0] + r2[0], r[0] + r2[1], int(sub.sum())
HIDE = {"left_rubber_hand", "right_rubber_hand"}


def randomize(img: np.ndarray, robot: np.ndarray, rng) -> np.ndarray:
    """배경 = 무작위 두 색 그라데이션 + 잡음 + 사각형 몇 개, 로봇 = 밝기·대비 흔들기."""
    h, w = robot.shape
    c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    t = np.linspace(0, 1, h if rng.random() < 0.5 else w)
    grad = (c0[None] * (1 - t[:, None]) + c1[None] * t[:, None])
    bg = np.broadcast_to(grad[:, None, :], (h, w, 3)) if len(t) == h else np.broadcast_to(grad[None, :, :], (h, w, 3))
    bg = bg + rng.normal(0, rng.uniform(2, 25), (h, w, 3))
    bg = bg.copy()
    for _ in range(rng.integers(0, 6)):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        bg[y0:y0 + rng.integers(10, h // 2), x0:x0 + rng.integers(10, w // 2)] = rng.uniform(0, 255, 3)
    out = img.astype(float)
    out = (out - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-40, 40)
    out[~robot] = bg[~robot]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=int, default=640); ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--preview", default="")
    ap.add_argument("--bg", choices=["sim", "random"], default="sim",
                    help="random: 로봇 밖 픽셀을 무작위 그라데이션·잡음·사각형으로, 로봇 밝기·대비도 흔든다(MuJoCo 바닥·하늘 과적합 방지)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(f"{a.out}/images", exist_ok=True); os.makedirs(f"{a.out}/labels", exist_ok=True)
    m = mj.MjModel.from_xml_path(XML.replace("g1_29dof.xml", "scene_29dof.xml"))
    m.vis.global_.offwidth, m.vis.global_.offheight = a.w, a.h
    d = mj.MjData(m)
    gclass = np.full(m.ngeom, -1)
    ginst = np.array([""] * m.ngeom, dtype=object)
    for g in range(m.ngeom):
        if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        mesh = mj.mj_id2name(m, mj.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
        if mesh in HIDE:
            m.geom_rgba[g, 3] = 0.0
        elif mesh in MESH_CLASS:   # 시각(group 1)·충돌(group 0) geom 이 겹쳐 그려진다 — 둘 다 라벨에 넣는다
            gclass[g] = MESH_CLASS[mesh]
            ginst[g] = MESH_INST.get(mesh, "")
    adr = [m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)] for n in CSV_JOINTS]
    lib = np.loadtxt(a.lib, delimiter=",")
    rgb_r = mj.Renderer(m, a.h, a.w); seg_r = mj.Renderer(m, a.h, a.w); seg_r.enable_segmentation_rendering()
    cam = mj.MjvCamera(); mj.mjv_defaultCamera(cam)
    previews = []
    for i in range(a.n):
        row = lib[rng.integers(len(lib))]
        d.qpos[:] = 0; d.qpos[2] = row[2]; d.qpos[3:7] = row[[6, 3, 4, 5]]; d.qpos[adr] = row[7:]
        mj.mj_forward(m, d)
        cam.lookat[:] = d.qpos[:3] + [rng.normal(0, 0.1), rng.normal(0, 0.1), rng.uniform(0.0, 0.5)]
        while True:   # 카메라가 바닥(0.15 m) 아래로 내려가지 않게
            cam.azimuth = rng.uniform(0, 360); cam.elevation = rng.uniform(-15, 35); cam.distance = rng.uniform(1.0, 3.5)
            if cam.lookat[2] - cam.distance * np.sin(np.radians(cam.elevation)) > 0.15:
                break
        rgb_r.update_scene(d, cam); img = rgb_r.render()
        seg_r.update_scene(d, cam); seg = seg_r.render()
        objid, objtype = seg[..., 0], seg[..., 1]
        if a.bg == "random":
            img = randomize(img, (objtype == int(mj.mjtObj.mjOBJ_GEOM)) & (m.geom_bodyid[np.clip(objid, 0, m.ngeom - 1)] > 0), rng)
        lines = []
        isgeom = objtype == int(mj.mjtObj.mjOBJ_GEOM)
        for c in range(3):
            for inst in sorted(set(ginst[gclass == c])):
                gids = np.where((gclass == c) & (ginst == inst))[0]
                b = robust_box(isgeom & np.isin(objid, gids))
                if b is None or b[4] < 40:
                    continue
                x0, x1, y0, y1, _ = b
                lines.append(f"{c} {(x0 + x1) / 2 / a.w:.6f} {(y0 + y1) / 2 / a.h:.6f} {(x1 - x0) / a.w:.6f} {(y1 - y0) / a.h:.6f}")
        name = f"g1_{a.seed:03d}_{i:05d}"
        imageio.imwrite(f"{a.out}/images/{name}.jpg", img)
        open(f"{a.out}/labels/{name}.txt", "w").write("\n".join(lines) + ("\n" if lines else ""))
        if a.preview and i < 24:
            pv = img.copy()
            col = [(255, 80, 80), (80, 255, 80), (80, 160, 255)]
            for ln in lines:
                c, cx, cy, w, h = ln.split(); c = int(c)
                x0 = int((float(cx) - float(w) / 2) * a.w); x1 = int((float(cx) + float(w) / 2) * a.w)
                y0 = int((float(cy) - float(h) / 2) * a.h); y1 = int((float(cy) + float(h) / 2) * a.h)
                pv[max(y0, 0):y0 + 3, x0:x1] = col[c]; pv[max(y1 - 3, 0):y1, x0:x1] = col[c]
                pv[y0:y1, max(x0, 0):x0 + 3] = col[c]; pv[y0:y1, max(x1 - 3, 0):x1] = col[c]
            previews.append(pv[::2, ::2])
    open(f"{a.out}/classes.txt", "w").write("\n".join(CLASSES) + "\n")
    if previews:
        rows = [np.concatenate(previews[k:k + 6], 1) for k in range(0, len(previews) - len(previews) % 6, 6)]
        imageio.imwrite(a.preview, np.concatenate(rows, 0))
    print(f"{a.n} 장 → {a.out} (클래스 {CLASSES})")


if __name__ == "__main__":
    main()
