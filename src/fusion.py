#!/usr/bin/env python3
"""
docs/03_fusion.md 체크리스트 1번 — gaze_point_world() 구현 + 합성 데이터 검증.

시선 픽셀(스테레오 좌영상, rectified) + 그 픽셀의 스테레오 깊이 + 그 순간의 SLAM pose를
합쳐서, 시선이 향한 세계 좌표 3D점을 복원한다. 실카메라/실SLAM 연결 전에 순수 기하로만
먼저 검증한다 (책상 위 오프라인 검증 원칙, docs/03 참고).

다음 단계(아직 안 함):
  - oCamS rectified 좌/우 영상으로 StereoSGBM 돌려서 실제 D 얻기
  - /orbslam3/pose(PoseStamped, frame=map) 구독해서 T_WS 채우기

좌표계(완료): gaze_on_scene.py는 이제 ocams_calib.build_rectify_maps()로 rectify한 이미지
위에서 캘리브레이션한다 (raw 이미지 기준이면 여기 K와 안 맞았음). K/baseline은 ocams_calib이
유일한 출처 — 여기서 중복 정의하지 않고 그대로 가져다 쓴다.
"""
import numpy as np

from ocams_calib import RECTIFIED_K as OCAMS_RECTIFIED_K
from ocams_calib import BASELINE_M as OCAMS_BASELINE_M


def gaze_point_world(u, v, D, K, T_WS):
    """
    u, v : 캘리브레이션 매핑으로 얻은 스테레오(rectified) 이미지상의 시선 픽셀
    D    : 그 픽셀의 스테레오 깊이 (m)
    K    : rectified 카메라 내부 파라미터 3x3
    T_WS : 해당 타임스탬프의 SLAM pose (4x4, world <- sensor)
    반환 : (p_W 3D점, ray_origin, ray_dir)
    """
    Kinv = np.linalg.inv(K)
    p_S = D * (Kinv @ np.array([u, v, 1.0]))  # 카메라 좌표계 3D점

    p_S_h = np.array([p_S[0], p_S[1], p_S[2], 1.0])
    p_W = (T_WS @ p_S_h)[:3]

    origin = T_WS[:3, 3]
    ray_dir = p_W - origin
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    return p_W, origin, ray_dir


def _project(p_cam, K):
    """검증용: 카메라 좌표 3D점 -> (u, v). gaze_point_world의 역변환."""
    x, y, z = p_cam
    uvw = K @ np.array([x / z, y / z, 1.0])
    return uvw[0], uvw[1]


def _pose_matrix(translation, euler_xyz_deg):
    """검증용: XYZ 오일러(도) + 평행이동 -> 4x4 T_WS."""
    rx, ry, rz = np.deg2rad(euler_xyz_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


def _run_synthetic_check(name, p_W_true, T_WS, K, atol=1e-4):
    """알려진 p_W_true를 T_WS/K로 (u,v,D)까지 순방향 계산 -> gaze_point_world로 역복원 -> 비교."""
    T_SW = np.linalg.inv(T_WS)
    p_W_true_h = np.array([*p_W_true, 1.0])
    p_S = (T_SW @ p_W_true_h)[:3]  # world -> sensor 좌표

    D = p_S[2]
    u, v = _project(p_S, K)

    p_W_est, origin, ray_dir = gaze_point_world(u, v, D, K, T_WS)

    err = np.linalg.norm(p_W_est - p_W_true)
    ok = err < atol

    # ray_dir이 origin에서 p_W_true를 실제로 향하는지도 확인
    expected_dir = (p_W_true - origin) / np.linalg.norm(p_W_true - origin)
    dir_err = np.linalg.norm(ray_dir - expected_dir)

    status = "OK" if ok and dir_err < atol else "FAIL"
    print(f"[{status}] {name}: p_W_true={np.round(p_W_true, 4)} p_W_est={np.round(p_W_est, 4)} "
          f"pos_err={err:.2e} dir_err={dir_err:.2e}")
    return ok and dir_err < atol


if __name__ == "__main__":
    K = OCAMS_RECTIFIED_K
    all_ok = True

    # 1) 항등 pose (world == sensor), 정면 3m
    T_identity = np.eye(4)
    all_ok &= _run_synthetic_check("identity pose, 정면 3m",
                                    p_W_true=np.array([0.0, 0.0, 3.0]),
                                    T_WS=T_identity, K=K)

    # 2) 카메라가 평행이동만 된 pose (SLAM이 원점에서 벗어난 경우)
    T_translated = _pose_matrix([1.0, 0.5, -0.3], [0, 0, 0])
    all_ok &= _run_synthetic_check("평행이동 pose (1,0.5,-0.3), 좌상단 대상",
                                    p_W_true=np.array([1.5, 0.8, 2.0]),
                                    T_WS=T_translated, K=K)

    # 3) 회전 + 평행이동 pose (docs/03 체크리스트: "pose 이동 시 p_W 월드 고정성" 사전 검증)
    T_rotated = _pose_matrix([0.2, 0.1, -0.15], [10, -15, 5])
    all_ok &= _run_synthetic_check("회전+평행이동 pose, 임의 대상",
                                    p_W_true=np.array([-0.4, 0.9, 2.5]),
                                    T_WS=T_rotated, K=K)

    # 4) 실제 oCamS baseline 스케일 감(느낌 확인용): 12.48cm 베이스라인 기준 시차로 얻는 D
    #    disparity(px) = fx * baseline / D  ->  D = fx*baseline/disparity
    disparity_px = 20.0
    D_from_disparity = K[0, 0] * OCAMS_BASELINE_M / disparity_px
    print(f"[info] baseline={OCAMS_BASELINE_M*100:.2f}cm, disparity={disparity_px}px -> D={D_from_disparity:.3f}m")

    print("\n전체:", "PASS" if all_ok else "FAIL")
