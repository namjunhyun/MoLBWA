"""촬영 클립 csv 들을 하나의 연속 모션 라이브러리 csv 로 잇는다 (BeyondMimic 정책 1개 학습용).

    python3 build_motion_library.py --segments segments.yaml --out_csv library.csv --out_meta library_meta.json

csv 36열 = root pos 3 + root quat xyzw 4 + 관절 29, 30 fps. 이후 csv_to_npz.py 를 한 번 돌린다(50 fps 리샘플).
meta 의 segments 는 npz(50 fps) 프레임 인덱스 [start, end), segments_csv 는 csv 인덱스.
이음새 임계(관절 0.05 rad/프레임, root 0.02 m, yaw 0.05 rad) 초과 시 exit 1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

FPS_NPZ = 50
SEAM_MAX = {"joint": 0.05, "root": 0.02, "yaw": 0.05}


# ---- 쿼터니언 (xyzw) ----

def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_yaw(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw):
    yaw = np.asarray(yaw, dtype=float)
    return np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2), np.cos(yaw / 2)], axis=-1)


def quat_slerp(a, b, t):
    """행별 slerp. t 는 (T,) 또는 스칼라."""
    t = np.asarray(t, dtype=float)[..., None]
    d = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(d < 0, -b, b)  # 짧은 쪽
    d = np.clip(np.abs(d), 0.0, 1.0)
    th = np.arccos(d)
    s = np.sin(th)
    small = s < 1e-6
    s = np.where(small, 1.0, s)
    wa = np.where(small, 1 - t, np.sin((1 - t) * th) / s)
    wb = np.where(small, t, np.sin(t * th) / s)
    q = wa * a + wb * b
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- 입출력 ----

def load_csv(path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", ndmin=2)
    if arr.shape[1] != 36:
        raise ValueError(f"{path}: 열 {arr.shape[1]}개, 36 이어야 함")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path}: NaN/inf 포함")
    return arr


def turn_bin_of(name: str) -> int | None:
    """turn_l90 -> 90, turn_r45 -> -45 (+ 가 왼쪽)."""
    if not name.startswith("turn_"):
        return None
    deg = int(name[6:])
    return deg if name[5] == "l" else -deg


def npz_index(i: int, fps: int) -> int:
    return int(math.floor(i * FPS_NPZ / fps + 0.5))


def npz_frames(n_csv: int, fps: int) -> int:
    """csv_to_npz.py 출력 프레임 수 = len(arange(0, (N-1)/fps, 1/50))."""
    return len(np.arange(0.0, (n_csv - 1) / fps, 1.0 / FPS_NPZ))


# ---- 블렌드·이어 붙이기 ----

def _blend_to_standby(clip, stand_joints, stand_tilt, stand_z, nb):
    """앞 nb 프레임은 대기→클립, 뒤 nb 프레임은 클립→대기. 대기 자세의 yaw 는 클립 자신의 yaw 를 쓴다(yaw·xy 보존)."""
    out = clip.copy()
    if nb == 0:
        return out
    T = len(clip)
    if T < 2 * nb + 2:
        raise ValueError(f"클립 {T} 프레임이 블렌드 2×{nb} 보다 짧음")
    a = 0.5 - 0.5 * np.cos(np.pi * np.arange(nb + 1) / nb)  # 0 → 1, 끝에서 기울기 0
    for idx, w in ((np.arange(nb + 1), a), (np.arange(T - nb - 1, T), a[::-1])):
        q = clip[idx, 3:7]
        stand_q = quat_mul(yaw_quat(quat_yaw(q)), np.broadcast_to(stand_tilt, q.shape))
        out[idx, 3:7] = quat_slerp(stand_q, q, w)
        out[idx, 2] = stand_z + w * (clip[idx, 2] - stand_z)
        out[idx, 7:] = stand_joints + w[:, None] * (clip[idx, 7:] - stand_joints)
    return out


def _attach(clip, yaw_end, xy_end):
    """클립 시작 yaw·xy 를 0 으로 만든 뒤 앞 클립 끝 yaw·xy 로 합성 (z 축 회전 + xy 평행이동)."""
    out = clip.copy()
    dyaw = yaw_end - quat_yaw(clip[0, 3:7])
    c, s = math.cos(dyaw), math.sin(dyaw)
    rel = clip[:, :2] - clip[0, :2]
    out[:, 0] = xy_end[0] + c * rel[:, 0] - s * rel[:, 1]
    out[:, 1] = xy_end[1] + s * rel[:, 0] + c * rel[:, 1]
    q = quat_mul(np.broadcast_to(yaw_quat(dyaw), clip[:, 3:7].shape), clip[:, 3:7])
    out[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    return out


def build_library(clips: dict, fps: int = 30, blend_s: float = 0.5):
    """clips 는 순서대로 이어 붙인다(dict 삽입 순서). 첫 클립은 idle 이어야 한다. -> (library (N,36), meta)."""
    if not clips or next(iter(clips)) != "idle":
        raise ValueError("첫 세그먼트는 idle 이어야 함 (대기 자세 기준)")
    nb = int(round(blend_s * fps))
    idle0 = clips["idle"][0]
    stand_joints = idle0[7:].copy()
    q0 = idle0[3:7] / np.linalg.norm(idle0[3:7])
    stand_tilt = quat_mul(yaw_quat(-quat_yaw(q0)), q0)  # roll·pitch 만 남김
    stand_z = idle0[2]

    parts, seg_csv, n = [], {}, 0
    for name, clip in clips.items():
        clip = np.array(clip, dtype=float)
        clip[:, 3:7] /= np.linalg.norm(clip[:, 3:7], axis=1, keepdims=True)
        clip = _blend_to_standby(clip, stand_joints, stand_tilt, stand_z, nb)
        if parts:
            clip = _attach(clip, quat_yaw(parts[-1][-1, 3:7]), parts[-1][-1, :2])
        parts.append(clip)
        seg_csv[name] = [n, n + len(clip)]
        n += len(clip)
    lib = np.vstack(parts)

    segs = {k: [npz_index(s, fps), npz_index(e, fps)] for k, (s, e) in seg_csv.items()}
    segs[name][1] = min(segs[name][1], npz_frames(n, fps))  # 마지막 end 는 npz 길이로 자름
    meta = {
        "fps_csv": fps,
        "fps_npz": FPS_NPZ,
        "blend_frames_csv": nb,
        "segments": segs,
        "segments_csv": seg_csv,
        "turn_bins": {k: turn_bin_of(k) for k in clips if turn_bin_of(k) is not None},
    }
    return lib, meta


def seam_report(lib, meta) -> dict:
    """이음새(세그먼트 경계 ± 블렌드 구간) 안의 프레임 간 최대 점프. 클립 본체의 빠른 동작은 세지 않는다."""
    nb = meta["blend_frames_csv"]
    rep = {"joint": 0.0, "root": 0.0, "yaw": 0.0, "worst_seam": None}
    for name, (b, _) in list(meta["segments_csv"].items())[1:]:
        w = lib[max(b - nb - 1, 0):min(b + nb + 2, len(lib))]
        j = float(np.max(np.abs(np.diff(w[:, 7:], axis=0))))
        r = float(np.max(np.linalg.norm(np.diff(w[:, :3], axis=0), axis=1)))
        y = float(np.max(np.abs(wrap(np.diff(quat_yaw(w[:, 3:7]))))))
        if any(v > SEAM_MAX[k] for k, v in (("joint", j), ("root", r), ("yaw", y))):
            rep["worst_seam"] = rep["worst_seam"] or name
        rep["joint"], rep["root"], rep["yaw"] = max(rep["joint"], j), max(rep["root"], r), max(rep["yaw"], y)
    return rep


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--seam_joint", type=float, default=SEAM_MAX["joint"],
                    help="이음새 관절 임계 rad/프레임. 보행 회전 클립은 걸음 자체가 ~0.18 이라 블렌드 창에 잡힌다 — "
                         "그때만 올리고, 원 클립 내부 최대 변화와 같은지 먼저 확인할 것")
    a = ap.parse_args(argv)

    with open(a.segments) as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(a.segments))
    clips = {}
    for s in cfg["segments"]:
        if s["name"] in clips:
            raise ValueError(f"세그먼트 중복: {s['name']}")
        clips[s["name"]] = load_csv(os.path.join(base, s["csv"]))  # 없는 파일은 그대로 예외 (조용히 건너뛰지 않음)
    lib, meta = build_library(clips, fps=int(cfg.get("fps", 30)), blend_s=float(cfg.get("blend_s", 0.5)))
    rep = seam_report(lib, meta)
    print(json.dumps({"frames_csv": len(lib), "segments": meta["segments"], "seam": rep}, indent=1))

    lim = dict(SEAM_MAX, joint=a.seam_joint)
    bad = [k for k in lim if rep[k] > lim[k]]
    if bad:
        print(f"이음새 임계 초과: {bad} (세그먼트 {rep['worst_seam']}) — 출력 안 함", file=sys.stderr)
        return 1
    np.savetxt(a.out_csv, lib, delimiter=",", fmt="%.6f")
    with open(a.out_meta, "w") as f:
        json.dump(meta, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
