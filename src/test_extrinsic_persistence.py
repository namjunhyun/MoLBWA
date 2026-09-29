#!/usr/bin/env python3
"""docs/12 (R,p_eye) 실전 연결용 저장/불러오기 검증 — 카메라 없이.

2026-09-27: "실전 연결 진행시키자" — (R,p_eye) 캘리브를 gaze_on_scene.py 재시작 사이에도
쓰려면 파일로 남겨야 한다(옛 affine 저장/불러오기와 같은 필요성). test_calib_persistence.py
와 짝을 이룬다: 그쪽은 affine, 이쪽은 R,p_eye.

실행: python3 test_extrinsic_persistence.py
"""
import os
import sys
import tempfile

import numpy as np

import eye_scene_extrinsic
import gaze_on_scene as g

tracker = g.tracker


def check(name, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def synthetic_calib(n=8, seed=0):
    """알려진 (R, p_eye)로 만든 (시선방향, 3D점) 쌍 — eye_scene_extrinsic.py 자체 테스트와 같은 방식."""
    rng = np.random.default_rng(seed)
    R_true = eye_scene_extrinsic.wahba_rotation(
        [np.array([0.0, 0.0, 1.0])], [np.array([0.1, 0.0, 1.0]) / np.linalg.norm([0.1, 0, 1])])
    p_eye_true = np.array([0.01, -0.02, 0.005])
    dirs, points = [], []
    for i in range(n):
        d = rng.normal(0, 0.2, 3) + [0, 0, 1.0]
        d /= np.linalg.norm(d)
        dist = 0.4 + 0.1 * i          # 여러 거리 (docs/12 취지)
        X = p_eye_true + dist * (R_true @ d)
        dirs.append(d)
        points.append(X)
    R_fit, p_eye_fit, residuals = eye_scene_extrinsic.calibrate_r_p_eye(dirs, points)
    return R_fit, p_eye_fit, dirs, points, residuals


def main():
    ok = True
    tmp = tempfile.mkdtemp()
    R, p_eye, dirs, points, residuals = synthetic_calib()
    ok &= check("합성 데이터로 (R,p_eye) 잔차 거의 0",
                float(residuals.max()) < 1e-6, f"최대잔차 {residuals.max()*1e6:.2f}um")

    # 1) 저장 -> 불러오기 왕복
    tracker.eye_sphere_adjustment_enabled = True
    p = os.path.join(tmp, "ext.json")
    g.save_extrinsic_calibration(p, R, p_eye, dirs, points, residuals)
    loaded = g.load_extrinsic_calibration(p)
    ok &= check("저장 파일이 존재하면 None 아님", loaded is not None)
    R2, p_eye2, dirs2, points2, meta = loaded
    ok &= check("왕복 후 같은 R", np.allclose(R, R2, atol=1e-9))
    ok &= check("왕복 후 같은 p_eye", np.allclose(p_eye, p_eye2, atol=1e-9))
    ok &= check("왕복 후 같은 점 개수", len(dirs2) == len(dirs) == len(points2) == len(points))
    ok &= check("자동 모드면 eye_model=None", meta.get("eye_model") is None)
    ok &= check("샘플 수/잔차 메타 저장됨",
                meta.get("sample_count") == len(dirs)
                and abs(meta.get("mean_residual_m", -1) - float(residuals.mean())) < 1e-9)

    # 2) 없는 파일 -> None (예외 아님)
    ok &= check("없는 파일 -> None", g.load_extrinsic_calibration(
        os.path.join(tmp, "nope.json")) is None)

    # 3) 깨진 파일(행렬 크기 이상) -> None, 예외 안 남
    import json
    bad = {"version": 1, "R_3x3": [[1, 0], [0, 1]], "p_eye": [0, 0, 0]}
    p_bad = os.path.join(tmp, "bad.json")
    json.dump(bad, open(p_bad, "w"))
    ok &= check("깨진 R 크기 -> None", g.load_extrinsic_calibration(p_bad) is None)

    # 4) 고정된 안구 모델도 같이 저장/복원 (affine과 동일 이유 — p_eye/R 은 이 안구 모델
    #    기준 시선벡터에서 구해졌으므로, 재시작 후 다른 모델로는 안 맞는다).
    tracker.prev_model_center_avg = (300, 320)
    tracker.max_observed_distance = 400.0
    tracker.eye_sphere_adjustment_enabled = False
    p3 = os.path.join(tmp, "locked.json")
    g.save_extrinsic_calibration(p3, R, p_eye, dirs, points, residuals)
    tracker.reset_tracking_state()
    _, _, _, _, meta3 = g.load_extrinsic_calibration(p3)
    ok &= check("고정 모델이 파일에 남음",
                meta3["eye_model"] == {"center": [300, 320], "radius": 400.0},
                str(meta3["eye_model"]))
    g.restore_eye_model(meta3["eye_model"])
    ok &= check("복원 후 중심·고정 상태",
                tracker.prev_model_center_avg == (300, 320)
                and not tracker.eye_sphere_adjustment_enabled)
    tracker.reset_tracking_state()

    print("\n전체:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
