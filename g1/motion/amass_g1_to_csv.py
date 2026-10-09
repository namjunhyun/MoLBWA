"""AMASS_Retargeted_for_G1 npz → 라이브러리용 36열 csv (30 fps).

    python3 amass_g1_to_csv.py SRC.npz OUT.csv [--t0 S --t1 S] [--waist_yaw DEG --ramp S]

- 관절은 **이름으로** csv 순서(make_idle_csv.CSV_JOINTS)에 옮긴다. 루트 = body 0, 회전 wxyz → csv 의 xyzw.
- --waist_yaw: 허리 yaw 에 더할 각도(+ = 왼쪽). 시작·끝 --ramp 초 동안 smoothstep 으로 들어가고 빠진다
  (라이브러리 빌더가 앞뒤를 대기 자세로 0.5 s 블렌드하므로 끝은 결국 대기로 돌아간다).
- 데이터셋 fps 는 30 이다(라이브러리 입력과 같음). 다르면 거부한다.
원본 AMASS 는 비상업·연구용이다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_idle_csv import CSV_JOINTS  # noqa: E402

WAIST_LIM = np.radians(150.0)   # G1 waist_yaw 관절 한계 ±2.618 rad


def smooth_env(n: int, fps: float, ramp_s: float) -> np.ndarray:
    """0 → 1 (ramp_s) → 1 → 0 (ramp_s) 의 smoothstep 포락선, 길이 n."""
    t = np.arange(n) / fps
    T = (n - 1) / fps
    r = max(ramp_s, 1e-6)
    a = np.clip(t / r, 0, 1)
    b = np.clip((T - t) / r, 0, 1)
    s = lambda x: x * x * (3 - 2 * x)  # noqa: E731
    return np.minimum(s(a), s(b))


def convert(z, t0: float = 0.0, t1: float | None = None, waist_yaw_deg: float = 0.0, ramp_s: float = 1.0) -> np.ndarray:
    fps = float(np.atleast_1d(z["fps"])[0])
    if abs(fps - 30.0) > 1e-6:
        raise ValueError(f"fps {fps} — 30 fps 만 받는다")
    names = [str(n) for n in z["dof_names"]]
    missing = [n for n in CSV_JOINTS if n not in names]
    if missing:
        raise ValueError(f"관절 이름 없음: {missing}")
    idx = [names.index(n) for n in CSV_JOINTS]
    i0 = int(round(t0 * fps))
    i1 = z["dof_positions"].shape[0] if t1 is None else int(round(t1 * fps)) + 1
    q = z["dof_positions"][i0:i1][:, idx].astype(float)
    pos = z["body_positions"][i0:i1, 0].astype(float)
    wxyz = z["body_rotations"][i0:i1, 0].astype(float)
    wxyz /= np.linalg.norm(wxyz, axis=1, keepdims=True)
    if waist_yaw_deg:
        j = CSV_JOINTS.index("waist_yaw_joint")
        q[:, j] = np.clip(q[:, j] + np.radians(waist_yaw_deg) * smooth_env(len(q), fps, ramp_s), -WAIST_LIM, WAIST_LIM)
    out = np.zeros((len(q), 36))
    out[:, :3] = pos
    out[:, 3:7] = wxyz[:, [1, 2, 3, 0]]
    out[:, 7:] = q
    return out


def retime(rows: np.ndarray, speed: float) -> np.ndarray:
    """재생 속도 배율(0.5 = 두 배 느리게). 관절·위치는 선형 보간, 쿼터니언은 보간 후 정규화(인접 프레임이라 충분)."""
    if speed == 1.0:
        return rows
    n = len(rows)
    m = int(round((n - 1) / speed)) + 1
    src = np.linspace(0, n - 1, m)
    q = rows[:, 3:7].copy()
    for k in range(1, n):                      # 부호 연속(q 와 -q 는 같은 회전)
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]
    out = np.stack([np.interp(src, np.arange(n), c) for c in np.concatenate([rows[:, :3], q, rows[:, 7:]], 1).T], 1)
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def pad(rows: np.ndarray, sec: float, fps: float = 30.0) -> np.ndarray:
    """앞뒤에 첫/끝 프레임 정지를 sec 초씩 붙인다 — 모캡 클립에는 촬영 가이드의 '앞뒤 1초 정지'가 없다."""
    k = int(round(sec * fps))
    return np.concatenate([np.repeat(rows[:1], k, 0), rows, np.repeat(rows[-1:], k, 0)])


def upper_only(rows: np.ndarray, standby: dict, root_z: float = 0.76) -> np.ndarray:
    """상체 동작 전용: 다리 12관절 = 대기자세, 골반 = 첫 프레임 xy·대기 높이·직립(첫 yaw 유지). 허리·팔만 클립을 따른다.
    손 흔들기처럼 상체 동작인데 모캡 다리 자세가 대기와 달라(엉덩이 ~0.33 rad) 흔드는 동안 키가 출렁이던 것을 없앤다."""
    out = rows.copy()
    for j, n in enumerate(CSV_JOINTS[:12]):
        out[:, 7 + j] = standby[n]
    out[:, 0:2] = rows[0, 0:2]
    out[:, 2] = root_z
    x, y, z, w = rows[0, 3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out[:, 3:7] = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
    return out


def pad_ends(rows: np.ndarray, start_s: float, end_s: float, fps: float = 30.0) -> np.ndarray:
    a, b = int(round(start_s * fps)), int(round(end_s * fps))
    return np.concatenate([np.repeat(rows[:1], a, 0), rows, np.repeat(rows[-1:], b, 0)])


def rate_limit(rows: np.ndarray, vmax: float) -> np.ndarray:
    """관절 각 프레임 변화를 ±vmax(rad/프레임)로 자른다 — 리타게팅 튐(관절 한계에 붙었다 떨어짐)을 경사로 바꾼다."""
    out = rows.copy()
    for k in range(1, len(out)):
        out[k, 7:] = out[k - 1, 7:] + np.clip(rows[k, 7:] - out[k - 1, 7:], -vmax, vmax)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--waist_yaw", type=float, default=0.0); ap.add_argument("--ramp", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율 (0.5 = 두 배 느리게)")
    ap.add_argument("--vmax", type=float, default=0.0, help="관절 변화 상한 rad/프레임 (0 = 끔)")
    ap.add_argument("--pad", type=float, default=0.0, help="앞뒤 정지 초")
    ap.add_argument("--pad_start", type=float, default=None); ap.add_argument("--pad_end", type=float, default=None)
    ap.add_argument("--upper_only", action="store_true", help="다리·골반은 대기자세로 고정하고 허리·팔만 클립을 따른다")
    a = ap.parse_args()
    rows = retime(convert(np.load(a.src, allow_pickle=False), a.t0, a.t1, a.waist_yaw, a.ramp), a.speed)
    if a.vmax > 0:
        rows = rate_limit(rows, a.vmax)
    if a.upper_only:
        import json
        meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy", "g1_tracking_policy_meta.json")))
        rows = upper_only(rows, dict(zip(meta["joint_names"], meta["default_joint_pos"])))
    ps = a.pad if a.pad_start is None else a.pad_start
    pe = a.pad if a.pad_end is None else a.pad_end
    if ps > 0 or pe > 0:
        rows = pad_ends(rows, ps, pe)
    np.savetxt(a.out, rows, delimiter=",", fmt="%.9f")
    print(f"{a.out}: {len(rows)} 프레임 {len(rows)/30:.2f}s, 허리 yaw +{a.waist_yaw:.0f}°")
