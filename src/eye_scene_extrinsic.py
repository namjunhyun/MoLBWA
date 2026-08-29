#!/usr/bin/env python3
"""눈 카메라 ↔ 씬 카메라 R+t 캘리브레이션 — (R, p_eye) 비선형 최소제곱.

설계 근거: docs/12_eye_scene_extrinsic_calibration.md. 기존 `gaze_on_scene.py`의 `m` 모드는
회전(R)+초점거리 스케일만 푸는 구조(Wahba)라 눈-씬 카메라 사이의 실제 평행이동(t, 수 cm)을
설명 못 해서, 캘리브한 거리에서만 맞고 다른 거리를 보면 시차 오차가 생긴다. 여기서는 회전 R과
눈의 유효 원점 p_eye(씬 카메라 좌표계 기준)를 동시에 6자유도로 푼다.

이 모듈은 최적화 함수만 담는다 — gaze_on_scene.py에 데이터 수집 UI(새 모드/키)를 붙이는 건
별도 작업(docs/12 "데이터 수집 절차" 참고, 아직 미구현).
"""
import cv2
import numpy as np
from scipy.optimize import least_squares


def wahba_rotation(d_list, u_list):
    """여러 (눈 카메라 방향 d_i, 씬 카메라 방향 u_i) 단위벡터쌍으로 R·d_i ≈ u_i인 회전
    R을 SVD로 구한다(Wahba's problem, p_eye=0 가정) — 6DoF 비선형 최적화의 초기값용."""
    D = np.asarray(d_list, dtype=np.float64)
    U = np.asarray(u_list, dtype=np.float64)
    B = U.T @ D  # sum_i u_i @ d_i^T
    Uu, _, Vt = np.linalg.svd(B)
    # det가 -1이면 반사가 섞여 나와서 진짜 회전이 아니게 됨 — 부호 보정
    M = np.diag([1.0, 1.0, np.linalg.det(Uu @ Vt)])
    return Uu @ M @ Vt


def _residuals(x, d_list, X_list):
    """docs/12의 잔차 정의: 광선(원점=p_eye, 방향=R·d_i)에서 X_i까지의 점-직선 거리 벡터.
    scipy.optimize.least_squares에 넘기려고 각 점의 3성분을 그대로 이어붙인다 — 성분별
    제곱합이 문서의 노름(residual_i) 제곱합과 동일해서 목적함수는 같다."""
    rvec = x[:3]
    p_eye = x[3:6]
    R, _ = cv2.Rodrigues(rvec)
    res = []
    for d_i, X_i in zip(d_list, X_list):
        ray_dir = R @ d_i  # R이 진짜 회전이면 이미 단위벡터
        diff = X_i - p_eye
        proj = np.dot(diff, ray_dir)
        res.append(diff - proj * ray_dir)
    return np.concatenate(res)


def calibrate_r_p_eye(d_list, X_list, p_eye0=None):
    """(R, p_eye)를 비선형 최소제곱으로 동시에 푼다.

    d_list: 눈 카메라 좌표계 시선 방향 단위벡터들, (N,3) — gaze_on_scene.py의 smooth_dir.
    X_list: 씬 카메라 좌표계 실제 응시 3D 위치, 미터 단위, (N,3) —
            X_i = D_i · K⁻¹ [u_i, v_i, 1]ᵀ (fusion.py와 동일한 공식).
    p_eye0: 초기 추정 원점(기본 [0,0,0] — CAD 실측치 없으면 이대로 둔다).

    반환: (R (3,3) 회전행렬, p_eye (3,) 씬 카메라 좌표계 기준 위치,
           residual_norms (N,) 점별 점-직선 거리(미터) — 잔차 확인/이상치 판단용).
    """
    d_arr = np.asarray(d_list, dtype=np.float64)
    X_arr = np.asarray(X_list, dtype=np.float64)
    if len(d_arr) < 6:
        raise ValueError(f"점이 너무 적음({len(d_arr)}개) — 최소 6개 이상 권장(docs/12)")
    if len(d_arr) != len(X_arr):
        raise ValueError(f"d_list({len(d_arr)})와 X_list({len(X_arr)}) 개수가 다름")

    if p_eye0 is None:
        p_eye0 = np.zeros(3)
    else:
        p_eye0 = np.asarray(p_eye0, dtype=np.float64)

    diffs0 = X_arr - p_eye0
    norms0 = np.linalg.norm(diffs0, axis=1)
    if np.any(norms0 < 1e-9):
        raise ValueError("p_eye0와 좌표가 같은 캘리브 점이 있음 — 초기 방향을 못 구함")
    u_list0 = diffs0 / norms0[:, None]

    R0 = wahba_rotation(d_arr, u_list0)
    rvec0, _ = cv2.Rodrigues(R0)
    x0 = np.concatenate([rvec0.flatten(), p_eye0])

    result = least_squares(_residuals, x0, args=(d_arr, X_arr), method='lm')

    rvec = result.x[:3]
    p_eye = result.x[3:6]
    R, _ = cv2.Rodrigues(rvec)

    residual_norms = np.linalg.norm(result.fun.reshape(-1, 3), axis=1)

    return R, p_eye, residual_norms


if __name__ == "__main__":
    # 자기 검증: 알려진 (R, p_eye)로 합성 데이터를 만들고 복원되는지 확인.
    # 실측 데이터가 아직 없어서(docs/12 "데이터 수집 절차" 미구현) 수식/구현 자체의
    # 정확성만 이걸로 먼저 검증한다.
    rng = np.random.default_rng(0)
    R_true = cv2.Rodrigues(np.array([0.12, -0.22, 0.06]))[0]
    p_eye_true = np.array([0.021, -0.013, 0.034])  # 눈-씬 카메라 사이 실제 t 스케일(수 cm) 흉내

    N = 12
    d_list, X_list = [], []
    for _ in range(N):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        depth = rng.uniform(0.4, 2.0)  # docs/12 목표 범위(0.5~3m)와 비슷하게
        X_list.append(p_eye_true + depth * (R_true @ d))
        d_list.append(d)

    R_est, p_eye_est, residuals = calibrate_r_p_eye(d_list, X_list)

    angle_err_deg = np.degrees(
        np.arccos(np.clip((np.trace(R_true.T @ R_est) - 1) / 2, -1, 1)))
    p_eye_err_m = np.linalg.norm(p_eye_true - p_eye_est)

    print(f"[self-test] 회전 오차: {angle_err_deg:.6f}도")
    print(f"[self-test] p_eye 오차: {p_eye_err_m*1000:.6f}mm")
    print(f"[self-test] 잔차(m): min={residuals.min():.2e} max={residuals.max():.2e}")
    assert angle_err_deg < 1e-3, "노이즈 없는 합성 데이터인데 회전 복원 오차가 너무 큼"
    assert p_eye_err_m < 1e-3, "노이즈 없는 합성 데이터인데 p_eye 복원 오차가 너무 큼"
    print("[self-test] OK — 노이즈 없는 데이터에서 정확히 복원됨")
