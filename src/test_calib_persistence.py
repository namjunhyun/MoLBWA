#!/usr/bin/env python3
"""시선 affine 캘리브 저장/불러오기 검증 — 카메라 없이.

2026-09-23 실측에서 겪은 것들을 막는다:

  1. 입력 모델을 (x/z, y/z) -> (x, y) 로 바꿨다. 옛(v1) 파일은 원시 쌍으로 다시
     맞춰야 하고, 원시 쌍이 없으면 거부해야 한다 (옛 행렬을 새 입력에 그대로 쓰면
     엉뚱한 픽셀이 나온다).
  2. 깜빡임 점 하나가 17.7px 결과를 55.8px 로 자동 덮어썼다 -> 불량점 판정.
  3. affine 은 그 세션의 안구 중심 기준이다 -> 고정된 안구 모델을 같이 저장/복원.

실행: python3 test_calib_persistence.py
"""
import json
import os
import sys
import tempfile

import numpy as np

import gaze_on_scene as g

tracker = g.tracker


def check(name, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def synthetic_points(n=9, noise_px=0.0, seed=0):
    """알려진 affine 으로 만든 (시선, 픽셀) 쌍."""
    rng = np.random.default_rng(seed)
    A = np.array([[400.0, 0.0, 330.0], [0.0, -600.0, 560.0]])
    dirs = []
    for x in np.linspace(-0.3, 0.3, 3):
        for y in np.linspace(0.2, 0.6, 3):
            d = np.array([x, y, 1.0])
            dirs.append(d / np.linalg.norm(d))
    dirs = dirs[:n]
    pix = [tuple(int(round(v)) for v in A @ g.gaze_features(d) + rng.normal(0, noise_px, 2))
           for d in dirs]
    return dirs, pix


def main():
    ok = True
    tmp = tempfile.mkdtemp()
    dirs, pix = synthetic_points(noise_px=1.0)

    # 1) 저장 -> 불러오기 왕복
    tracker.eye_sphere_adjustment_enabled = True
    A = g.calibrate_affine(dirs, pix)
    p = os.path.join(tmp, "c.json")
    g.save_affine_calibration(p, A, dirs, pix, 640, 480)
    B, meta = g.load_affine_calibration(p, 640, 480)
    ok &= check("왕복 후 같은 행렬", np.allclose(A, B, atol=1e-3))
    ok &= check("자동 모드면 eye_model=None", meta["eye_model"] is None)

    # 2) 옛(v1) 파일: 원시 쌍으로 재계산
    old = dict(meta, version=1, affine_2x3=[[1, 0, 0], [0, 1, 0]])
    old.pop("features", None)
    json.dump(old, open(p, "w"))
    B, meta = g.load_affine_calibration(p, 640, 480)
    ok &= check("v1 파일은 원시 쌍으로 재계산", np.allclose(A, B, atol=1e-3))

    # 3) 옛 파일인데 원시 쌍 없음 -> 거부
    old.pop("gaze_dirs")
    json.dump(old, open(p, "w"))
    ok &= check("v1 + 원시 쌍 없음 -> 거부", g.load_affine_calibration(p, 640, 480) is None)

    # 4) 불량점 판정: 정상 점은 통과, 튀는 점은 걸러진다
    A = g.calibrate_affine(dirs, pix)
    bad, e = g.last_point_is_outlier(A, dirs, pix)
    ok &= check("정상 마지막 점은 통과", not bad, f"{e:.1f}px")
    d_bad = np.array([0.1, 0.05, 1.0]); d_bad /= np.linalg.norm(d_bad)
    dirs2, pix2 = dirs + [d_bad], pix + [(478, 230)]      # 실제 12번째 점처럼 y 가 틀린 점
    A2 = g.calibrate_affine(dirs2, pix2)
    bad, e = g.last_point_is_outlier(A2, dirs2, pix2)
    ok &= check("튀는 마지막 점은 걸림", bad, f"{e:.1f}px")

    # 5) 고정된 안구 모델 저장 -> 복원
    tracker.prev_model_center_avg = (311, 337)
    tracker.max_observed_distance = 412.5
    tracker.eye_sphere_adjustment_enabled = False
    p2 = os.path.join(tmp, "locked.json")
    g.save_affine_calibration(p2, A, dirs, pix, 640, 480)
    tracker.reset_tracking_state()                      # 재시작 흉내
    _, meta = g.load_affine_calibration(p2, 640, 480)
    ok &= check("고정 모델이 파일에 남음", meta["eye_model"] == {"center": [311, 337], "radius": 412.5},
                str(meta["eye_model"]))
    g.restore_eye_model(meta["eye_model"])
    ok &= check("복원 후 중심·고정 상태",
                tracker.prev_model_center_avg == (311, 337)
                and not tracker.eye_sphere_adjustment_enabled)
    tracker.reset_tracking_state()

    print("\n전체:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
