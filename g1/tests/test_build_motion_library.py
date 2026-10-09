import os
import subprocess
import sys

import numpy as np

MOTION_DIR = os.path.join(os.path.dirname(__file__), "..", "motion")
sys.path.insert(0, MOTION_DIR)

from build_motion_library import build_library, load_csv, quat_yaw, seam_report  # noqa: E402

FPS = 30
STAND = np.linspace(-0.3, 0.3, 29)  # 합성 대기 관절각


def yaw_quat(yaw):
    """z 축 회전 쿼터니언 xyzw."""
    yaw = np.asarray(yaw, dtype=float)
    q = np.zeros(yaw.shape + (4,))
    q[..., 2] = np.sin(yaw / 2)
    q[..., 3] = np.cos(yaw / 2)
    return q


def make_clip(T, yaw0, yaw1, xy0, end_off):
    """관절 사인파 + 끝으로 갈수록 대기 자세에서 end_off rad 어긋남. root yaw 는 yaw0 -> yaw1."""
    t = np.arange(T) / FPS
    s = np.linspace(0.0, 1.0, T)
    joints = STAND + 0.1 * np.sin(2 * np.pi * 0.5 * t)[:, None] * np.sin(np.arange(29))
    joints = joints + end_off * s[:, None]
    yaw = yaw0 + (yaw1 - yaw0) * s
    pos = np.zeros((T, 3))
    pos[:, :2] = xy0 + 0.1 * s[:, None]
    pos[:, 2] = 0.79
    return np.hstack([pos, yaw_quat(yaw), joints])


def clips():
    return {
        "idle": make_clip(120, 0.3, 0.3, np.array([1.0, 2.0]), 0.0),
        "turn_l90": make_clip(90, -1.0, -1.0 + np.pi / 2, np.array([-3.0, 0.5]), 0.2),
        "bow": make_clip(90, 2.0, 2.0, np.array([0.0, -4.0]), 0.2),
    }


def test_seams_continuous():
    lib, meta = build_library(clips(), fps=FPS, blend_s=0.5)
    r = seam_report(lib, meta)
    assert r["joint"] <= 0.05, r
    assert r["root"] <= 0.02, r
    assert r["yaw"] <= 0.05, r


def test_meta_indices():
    lib, meta = build_library(clips(), fps=FPS, blend_s=0.5)
    assert meta["fps_csv"] == 30 and meta["fps_npz"] == 50
    segs = list(meta["segments"].values())
    assert list(meta["segments"]) == ["idle", "turn_l90", "bow"]
    assert segs[0][0] == 0
    for (s0, e0), (s1, _) in zip(segs, segs[1:]):
        assert e0 == s1
    for s, e in segs:
        assert e > s
    assert segs[-1][1] <= round(len(lib) * 50 / 30)
    assert meta["turn_bins"] == {"turn_l90": 90}


def test_turn_yaw_accumulates():
    lib, meta = build_library(clips(), fps=FPS, blend_s=0.5)
    _, end = meta["segments_csv"]["turn_l90"]
    start, _ = meta["segments_csv"]["bow"]
    assert start == end
    y = quat_yaw(lib[:, 3:7])
    assert abs(y[start] - y[end - 1]) < 1e-6
    # 회전 클립은 실제로 90° 를 돈다
    s0, _ = meta["segments_csv"]["turn_l90"]
    turned = (y[end - 1] - y[s0] + np.pi) % (2 * np.pi) - np.pi
    assert abs(turned - np.pi / 2) < 1e-6


def test_quat_unit():
    lib, _ = build_library(clips(), fps=FPS, blend_s=0.5)
    assert np.allclose(np.linalg.norm(lib[:, 3:7], axis=1), 1.0, atol=1e-6)


def test_load_csv_roundtrip(tmp_path):
    c = clips()["bow"]
    p = tmp_path / "bow.csv"
    np.savetxt(p, c, delimiter=",", fmt="%.6f")
    assert load_csv(str(p)).shape == (90, 36)


def test_cli_exit1_on_bad_seam(tmp_path):
    c = clips()
    lines = []
    for name, arr in c.items():
        np.savetxt(tmp_path / f"{name}.csv", arr, delimiter=",", fmt="%.6f")
        lines.append(f"  - {{name: {name}, csv: {name}.csv}}")

    def run(blend_s):
        yml = tmp_path / f"seg_{blend_s}.yaml"
        yml.write_text(f"fps: 30\nblend_s: {blend_s}\nsegments:\n" + "\n".join(lines) + "\n")
        return subprocess.run(
            [sys.executable, os.path.join(MOTION_DIR, "build_motion_library.py"),
             "--segments", str(yml), "--out_csv", str(tmp_path / "lib.csv"),
             "--out_meta", str(tmp_path / "lib_meta.json")],
            capture_output=True, text=True)

    assert run(0.5).returncode == 0
    bad = run(0)
    assert bad.returncode == 1, bad.stdout + bad.stderr



def test_per_segment_blend():
    """세그먼트별 blend_s: 지정한 세그먼트만 블렌드가 길어지고 meta 에 기록되며, 이음새는 여전히 연속."""
    c = clips()
    lib, meta = build_library(c, fps=30, blend_s=0.5, blend_by_seg={"bow": 1.0})
    by = meta["blend_frames_csv_by_seg"]
    assert by["bow"] == 30 and by["idle"] == 15 and by["turn_l90"] == 15
    assert seam_report(lib, meta)["joint"] <= 0.05
