#!/usr/bin/env python
"""하드웨어 없이 도는 검증 스위트.

    python test_pipeline.py

필요한 것: numpy, scipy, opencv-python, pyyaml.
ultralytics / pupil_apriltags / rclpy / lerobot 는 필요 없다.

2026-08-19 에 고친 결함들이 다시 들어오지 못하게 막는 게 목적이다:
  1. rectified K 가 ORB-SLAM3 설정과 일치하는가        (아니면 3D점이 조용히 틀린다)
  2. FK/IK 왕복
  3. anchor 좌표변환 왕복
  4. 태그 코너 축 규약 (거울상이면 PnP 가 안 풀린다)
  5. depth 없이 테이블 평면 교차로 컵 위치 복원
  6. 검출이 한 프레임 빠져도 dwell 이 같은 컵을 유지하는가
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from anchor import AnchorTracker, build_bundle_obj_pts          # noqa: E402
from kinematics import ArmModel, IKError                        # noqa: E402
from perception import Cup, GazeDwell, cup_position_on_table    # noqa: E402
from sim_source import SimSource, _euler, _rt                   # noqa: E402

CFG = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------- 1. intrinsics
def test_intrinsics():
    from ocams_calib import RECTIFIED_K, BASELINE_M, _orbslam3_intrinsics
    K_slam = _orbslam3_intrinsics()
    if K_slam is None:
        return check("intrinsics: ORB-SLAM3 설정 대조", False, "설정 yaml 을 못 읽음")
    diff = float(np.abs(RECTIFIED_K - K_slam).max())
    ok = diff < 0.5
    check("intrinsics: ORB-SLAM3 설정과 일치", ok,
          f"fx={RECTIFIED_K[0,0]:.2f} cx={RECTIFIED_K[0,2]:.2f} 최대차 {diff:.3f}px")
    # cx 가 화면 밖으로 나가면 거의 확실히 잘못된 캘리브레이션 파일이다
    check("intrinsics: cx 가 화면 중앙 부근", abs(RECTIFIED_K[0, 2] - 320) < 80,
          f"cx={RECTIFIED_K[0,2]:.2f}")
    check("intrinsics: baseline 이 물리적으로 그럴듯", 0.05 < BASELINE_M < 0.20,
          f"{BASELINE_M*100:.2f}cm")


# ---------------------------------------------------------------- 2. FK/IK
def test_kinematics():
    m = ArmModel(**CFG["arm"]["links"])
    rng = np.random.default_rng(0)
    errs, fails = [], 0
    for _ in range(500):
        q = np.array([rng.uniform(-1, 1), rng.uniform(-1.5, 0.5),
                      rng.uniform(-1.5, 0.5), rng.uniform(-1.5, 1.5), 0.0])
        p, pitch = m.fk(q)
        try:
            p2, _ = m.fk(m.ik(p, pitch))
            errs.append(float(np.linalg.norm(p - p2)))
        except IKError:
            fails += 1
    check("kinematics: FK/IK 왕복", errs and max(errs) < 1e-9,
          f"{len(errs)}/500 성공, 최대오차 {max(errs):.2e} m")


# ---------------------------------------------------------------- 3. anchor
def test_anchor():
    a = AnchorTracker()
    T_w_hc = _rt(_euler(10, -20, 5), [0.3, -0.1, 1.2])
    T_hc_ab = _rt(_euler(0, 30, 0), [0.05, 0.1, 0.6])
    a.update_slam(T_w_hc, 0, True)
    a.update_tag(T_hc_ab)
    p_true = np.array([0.2, 0.05, 0.1])
    p_hc = (T_hc_ab @ np.r_[p_true, 1.0])[:3]
    err = float(np.linalg.norm(a.point_headcam_to_armbase(p_hc) - p_true))
    check("anchor: 좌표변환 왕복", err < 1e-9, f"오차 {err:.2e} m")
    check("anchor: latch 후 ANCHORED", a.state.value == "ANCHORED", a.state.value)

    # Atlas 맵 전환 -> latch 무효화
    a.update_slam(T_w_hc, 1, True)
    check("anchor: map_id 바뀌면 latch 무효", a.state.value == "UNANCHORED", a.state.value)


# ---------------------------------------------------------------- 4. 태그 축 규약
def _project(obj_pts_ab, T_hc_ab, K):
    """armbase 코너 -> 헤드캠 -> 픽셀."""
    P = np.asarray(obj_pts_ab, float)
    P_hc = (T_hc_ab[:3, :3] @ P.T).T + T_hc_ab[:3, 3]
    uv = (K @ P_hc.T).T
    return uv[:, :2] / uv[:, 2:3]


def test_tag_convention():
    from ocams_calib import RECTIFIED_K as K
    a = dict(CFG["anchor"])
    obj_new, right, up, normal = build_bundle_obj_pts(a)

    check("태그: 법선이 팔 정면(+x)", float(normal @ np.array([1.0, 0, 0])) > 0.99,
          f"normal={np.round(normal,3).tolist()}")

    # 태그판이 카메라 앞 0.6m 에 보이는 상황
    base = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])   # det=+1
    assert abs(np.linalg.det(base) - 1.0) < 1e-9
    T_hc_ab = _rt(_euler(-15, 0, 0) @ base, [0.0, 0.05, 0.62])
    ids = sorted(obj_new)
    obj = np.concatenate([obj_new[i] for i in ids])
    img = _project(obj, T_hc_ab, K)

    def solve(obj_model):
        ok, rvec, tvec = cv2.solvePnP(obj_model, img, K, np.zeros(5),
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, float("inf")
        rvec, tvec = cv2.solvePnPRefineLM(obj_model, img, K, np.zeros(5), rvec, tvec)
        proj, _ = cv2.projectPoints(obj_model, rvec, tvec, K, np.zeros(5))
        err = float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(axis=1).mean()))
        R, _ = cv2.Rodrigues(rvec)
        return _rt(R, tvec), err

    T_est, err_new = solve(obj)
    pos_err = float(np.linalg.norm(T_est[:3, 3] - T_hc_ab[:3, 3])) if T_est is not None else 9e9
    check("태그: 새 코너 규약으로 PnP 복원", err_new < 0.5 and pos_err < 1e-3,
          f"재투영 {err_new:.3f}px, 위치오차 {pos_err*1000:.3f}mm")

    # 예전(거울상) 규약 — y 축이 뒤집힌 코너 정의
    h = a["tag_size_m"] / 2.0
    local_old = np.array([[0, +h, -h], [0, -h, -h], [0, -h, +h], [0, +h, +h]], float)
    obj_old = np.concatenate([np.asarray(t["pos"], float) + local_old
                              for t in a["bundle"] if t["id"] in ids])
    _, err_old = solve(obj_old)
    check("태그: 예전 거울상 규약은 재투영 오차로 걸러짐", err_old > a["max_reproj_px"],
          f"재투영 {err_old:.2f}px > 임계 {a['max_reproj_px']}px")


# ---------------------------------------------------------------- 5. 테이블 평면 교차
def test_table_plane():
    from ocams_calib import RECTIFIED_K as K
    sim = SimSource(CFG, K)
    errs = []
    for i, truth in enumerate(sim.cups_true_ab):
        cup = sim.cups()[i]
        p_hc = cup_position_on_table(cup, sim.T_hc_ab, CFG["task"]["table_z"], K)
        p_ab = (np.linalg.inv(sim.T_hc_ab) @ np.r_[p_hc, 1.0])[:3]
        errs.append(float(np.linalg.norm(p_ab - truth)))
    check("테이블 평면 교차: depth 없이 컵 3D 복원", max(errs) < 1e-6,
          f"최대오차 {max(errs)*1000:.4f} mm (컵 3개)")


# ---------------------------------------------------------------- 6. dwell 견고성
def test_dwell_robustness():
    def make(n, t):
        """n개 컵. 화면 x = 200/320/440. 중간 컵이 빠지면 인덱스가 밀린다."""
        xs = [200.0, 320.0, 440.0] if n == 3 else [200.0, 440.0]
        return [Cup(idx=i, center_uv=(x, 300.0), bbox=(x - 25, 265, x + 25, 335),
                    conf=0.9) for i, x in enumerate(xs)]

    d = GazeDwell(dwell_s=1.0, release_s=0.4, track_radius_px=60.0)
    gaze = (440.0, 300.0)          # 오른쪽 컵을 계속 응시
    t = 0.0
    d.update(make(3, t), gaze, t)                 # 3개 다 보임, 오른쪽 = idx2
    t += 0.5
    d.update(make(2, t), gaze, t)                 # 가운데가 빠짐 -> 오른쪽이 idx1 로 밀림
    t += 0.6
    sel = d.update(make(3, t), gaze, t)           # 다시 3개
    ok = sel is not None and abs(sel.center_uv[0] - 440.0) < 1.0
    check("dwell: 검출 누락으로 인덱스가 밀려도 같은 컵 유지", ok,
          f"선택 uv={None if sel is None else tuple(round(x,1) for x in sel.center_uv)}")

    # 다른 컵으로 시선을 옮기면 dwell 이 다시 시작돼야 한다
    d2 = GazeDwell(dwell_s=1.0, release_s=0.4, track_radius_px=60.0)
    t = 0.0
    d2.update(make(3, t), (200.0, 300.0), t)
    t += 0.9
    d2.update(make(3, t), (200.0, 300.0), t)
    t += 0.2
    sel2 = d2.update(make(3, t), (440.0, 300.0), t)   # 컵 바꿈 -> 아직 선택되면 안 됨
    check("dwell: 컵을 바꾸면 처음부터 다시 센다", sel2 is None,
          f"sel={None if sel2 is None else sel2.center_uv}")


if __name__ == "__main__":
    print("=" * 70)
    test_intrinsics()
    test_kinematics()
    test_anchor()
    test_tag_convention()
    test_table_plane()
    test_dwell_robustness()
    print("=" * 70)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
    sys.exit(1 if n_fail else 0)
