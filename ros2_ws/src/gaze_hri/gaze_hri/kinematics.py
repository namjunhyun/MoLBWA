"""
SO-ARM101 기구학 + 좌표계 정합 유틸.

SO-ARM101은 5자유도(+그리퍼)입니다:
    j1 shoulder_pan   : 베이스 yaw 회전 (Z축)
    j2 shoulder_lift  : pitch
    j3 elbow_flex     : pitch
    j4 wrist_flex     : pitch
    j5 wrist_roll     : 그리퍼 축 roll
    j6 gripper        : 개폐

j2/j3/j4가 모두 같은 방향의 pitch 관절이므로, j1이 정한 수직 평면 안에서
움직이는 "3링크 평면 팔"이 됩니다. 덕분에 해석적 IK가 깔끔하게 풀립니다.

5자유도라 6D 자세(위치+방향)를 자유롭게 지정할 수 없습니다.
그래서 이 구현은 **위치 3자유도 + 접근 피치각 1자유도**만 지정합니다.
(컵 집기에는 이걸로 충분합니다.)

!! 중요 !!
아래 링크 길이(L1, L2, L3, BASE_HEIGHT, SHOULDER_OFFSET)는 반드시
본인 SO-ARM101의 URDF에서 실제 값으로 바꿔야 합니다.
    ros2 launch so_arm101_description display.launch.py
    또는 URDF의 <joint><origin xyz="..."> 값을 직접 읽으세요.
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArmGeometry:
    """URDF에서 읽어와야 하는 값들 (단위: m, rad)."""
    base_height: float = 0.0563      # base_link -> shoulder_lift 축까지의 z
    shoulder_offset: float = 0.0304  # 회전축에서 shoulder_lift 축까지의 수평 오프셋
    l1: float = 0.1160               # shoulder_lift -> elbow_flex
    l2: float = 0.1350               # elbow_flex   -> wrist_flex
    l3: float = 0.1100               # wrist_flex   -> 그리퍼 파지 중심(TCP)

    # 관절 한계 (URDF와 일치시킬 것)
    limits: dict = field(default_factory=lambda: {
        "shoulder_pan": (-1.92, 1.92),
        "shoulder_lift": (-1.75, 1.75),
        "elbow_flex": (-1.69, 1.69),
        "wrist_flex": (-1.66, 1.66),
        "wrist_roll": (-2.79, 2.79),
        "gripper": (-0.17, 1.75),
    })


JOINT_ORDER = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


# 탑다운 호모그래피 캘리브레이션 지점 — 테이블면 위 (x, y) [m], 로봇 베이스 기준.
#
# ★ 관절각이 아니라 (x, y) 목록인 이유
#   호모그래피는 "평면 -> 평면" 대응이라 캘리브 점들이 전부 **같은 높이**에 있어야
#   한다. 관절각을 손으로 적어두면 그리퍼 끝 높이가 제각각이 되는데(예전 버전은
#   3.6~13.1cm로 흩어져 있었다), 그러면 재투영 오차는 4mm로 '합격'이 나오면서
#   실제 테이블면 오차는 15mm, 카메라를 15도만 기울여도 79mm까지 벌어진다.
#   재투영 오차는 캘리브 점 자신에 대해서만 맞춘 값이라 실사용 정확도를 보증하지 않는다.
#
#   그래서 테이블면 위 지점만 적어두고 관절각은 IK로 푼다.
#   작업 영역에 넓게 퍼뜨려 화면 구석까지 외삽하지 않도록 했다.
#   (여기 있는 값은 기본 링크 길이 기준으로 전부 도달 가능함을 테스트로 확인한다)
TOPDOWN_CALIB_XY = [
    (0.19, -0.10), (0.19, 0.10),
    (0.25, -0.17), (0.25, 0.17),
    (0.31, -0.10), (0.31, 0.10),
    (0.22, 0.00), (0.30, 0.00),
]


def grasp_detected(commanded, actual, margin):
    """그리퍼가 물체를 물었는지 판정한다.

    물체가 손가락 사이에 있으면 명령한 만큼 닫히지 못하고 물체 두께에서 멈춘다.
    그 차이가 곧 증거다. 빈손이면 끝까지 닫혀서 차이가 거의 0이 된다.

    commanded : 닫으라고 명령한 그리퍼 각 [rad]
    actual    : 실제로 읽은 그리퍼 각 [rad]
    margin    : 이보다 더 열려 있어야 '물었다'로 본다 [rad]
    """
    return float(actual) - float(commanded) > float(margin)


class IKError(Exception):
    """목표점이 작업 공간(workspace) 밖이거나 관절 한계를 넘을 때."""


def forward_kinematics(q, geo: ArmGeometry):
    """관절각 -> TCP 위치 (base_link 기준). q는 최소 4개(j1~j4).

    캘리브레이션할 때 "로봇이 지금 어디를 가리키고 있는지"를 알기 위해 필요합니다.
    """
    j1, j2, j3, j4 = q[0], q[1], q[2], q[3]

    # 평면(r-z) 안에서의 누적 각도
    a1 = j2
    a2 = j2 + j3
    a3 = j2 + j3 + j4

    r = geo.shoulder_offset + geo.l1 * math.cos(a1) + geo.l2 * math.cos(a2) + geo.l3 * math.cos(a3)
    z = geo.base_height + geo.l1 * math.sin(a1) + geo.l2 * math.sin(a2) + geo.l3 * math.sin(a3)

    return np.array([r * math.cos(j1), r * math.sin(j1), z])


def inverse_kinematics(target_xyz, geo: ArmGeometry, approach_pitch=-math.pi / 2,
                       elbow_up=True):
    """base_link 기준 목표점 -> 관절각 [j1, j2, j3, j4].

    approach_pitch : TCP가 향하는 각도. 수평면 기준이며
                     -pi/2 = 바로 아래를 향함(top-down 파지),
                      0    = 수평으로 찌름(side 파지),
                     -pi/4 = 45도 비스듬히 접근 (SO-ARM101에 가장 안정적)
    elbow_up       : 두 IK 해 중 팔꿈치를 위로 드는 해를 고를지
    """
    x, y, z = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])

    # --- 1) 베이스 yaw ---
    j1 = math.atan2(y, x)

    # --- 2) 평면 문제로 축소 ---
    r = math.hypot(x, y) - geo.shoulder_offset
    zp = z - geo.base_height

    # --- 3) 접근각을 빼서 손목 중심(wrist center) 구하기 ---
    wr = r - geo.l3 * math.cos(approach_pitch)
    wz = zp - geo.l3 * math.sin(approach_pitch)

    # --- 4) 2링크 역기구학 ---
    d2 = wr * wr + wz * wz
    d = math.sqrt(d2)

    reach_max = geo.l1 + geo.l2
    reach_min = abs(geo.l1 - geo.l2)
    if d > reach_max - 1e-4 or d < reach_min + 1e-4:
        raise IKError(
            f"작업 공간 밖입니다: 손목중심까지 거리 {d:.3f} m "
            f"(가능 범위 {reach_min:.3f}~{reach_max:.3f} m). "
            f"목표점 {target_xyz} 또는 approach_pitch를 조정하세요."
        )

    cos_elbow = (d2 - geo.l1 ** 2 - geo.l2 ** 2) / (2 * geo.l1 * geo.l2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow = math.acos(cos_elbow)
    j3 = elbow if elbow_up else -elbow

    j2 = math.atan2(wz, wr) - math.atan2(
        geo.l2 * math.sin(j3), geo.l1 + geo.l2 * math.cos(j3)
    )

    # --- 5) 손목으로 접근각 맞추기 ---
    j4 = approach_pitch - j2 - j3

    q = [j1, j2, j3, j4]
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
    for name, val in zip(names, q):
        lo, hi = geo.limits[name]
        if not (lo - 1e-3 <= val <= hi + 1e-3):
            raise IKError(
                f"{name} 관절 한계 초과: {math.degrees(val):.1f}deg "
                f"(허용 {math.degrees(lo):.0f}~{math.degrees(hi):.0f}deg)"
            )
    return q


def solve_with_fallback(target_xyz, geo: ArmGeometry, pitch_candidates=None):
    """접근각 여러 개를 순서대로 시도해서 되는 걸 반환.

    시선으로 찍은 점은 오차가 있어서 한 접근각으로는 실패하기 쉽습니다.
    45도 -> 60도 -> 수직 -> 30도 순으로 시도하면 성공률이 크게 올라갑니다.
    """
    if pitch_candidates is None:
        pitch_candidates = [-math.pi / 4, -math.pi / 3, -math.pi / 2,
                            -math.pi / 6, -1.2, 0.0]
    last = None
    for pitch in pitch_candidates:
        for elbow_up in (True, False):
            try:
                return inverse_kinematics(target_xyz, geo, pitch, elbow_up), pitch
            except IKError as exc:
                last = exc
    raise IKError(f"모든 접근각 실패. 마지막 사유: {last}")


# --------------------------------------------------------------------------
# 좌표계 정합 (SLAM world W  ->  robot base B)
# --------------------------------------------------------------------------

def umeyama_rigid(src_points, dst_points):
    """대응점 쌍으로 강체 변환 T (4x4)를 구합니다. dst ~= R @ src + t

    src_points : (N,3) SLAM world 좌표에서 응시로 얻은 점들 p_W
    dst_points : (N,3) 같은 물리적 지점의 로봇 베이스 좌표 p_B (FK로 알고 있음)

    스케일은 1로 고정합니다. ORB-SLAM3 stereo-inertial은 IMU 덕분에
    미터 스케일이 나오므로 스케일을 풀면 오히려 오차가 커집니다.
    """
    src = np.asarray(src_points, dtype=float)
    dst = np.asarray(dst_points, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("대응점은 (N,3) 형태로 최소 3쌍(권장 5쌍 이상) 필요합니다.")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = (dst - mu_d).T @ (src - mu_s) / src.shape[0]

    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:      # 반사(reflection) 방지
        D[2, 2] = -1.0
    R = U @ D @ Vt
    t = mu_d - R @ mu_s

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def alignment_rmse(T, src_points, dst_points):
    """정합 품질 확인용. 2cm 이하면 데모하기 충분합니다."""
    src = np.asarray(src_points, dtype=float)
    dst = np.asarray(dst_points, dtype=float)
    pred = (T[:3, :3] @ src.T).T + T[:3, 3]
    return float(np.sqrt(((pred - dst) ** 2).sum(axis=1).mean()))


def quat_to_matrix(x, y, z, w):
    """쿼터니언 -> 3x3 회전행렬. TF 변환을 직접 적용할 때 쓴다."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_xyz_rpy(T):
    """static_transform_publisher에 넣을 수 있는 (x,y,z,roll,pitch,yaw)로 변환."""
    R = T[:3, :3]
    x, y, z = T[:3, 3]
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return x, y, z, roll, pitch, yaw


def fit_plane_ransac(points, iterations=300, threshold=0.01, rng=None):
    """테이블 평면 (a,b,c,d) 추정. ax+by+cz+d=0, (a,b,c)는 단위 법선.

    "놓을 자리" 응시점을 테이블 높이로 눌러주는 데 씁니다.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("평면 피팅에 최소 3점 필요")
    rng = rng or np.random.default_rng(0)

    best_inliers, best_plane = -1, None
    for _ in range(iterations):
        idx = rng.choice(pts.shape[0], 3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ p0
        dist = np.abs(pts @ n + d)
        inliers = int((dist < threshold).sum())
        if inliers > best_inliers:
            best_inliers, best_plane = inliers, (n, d)

    # 인라이어로 최소제곱 재피팅 (정밀도 향상)
    n, d = best_plane
    mask = np.abs(pts @ n + d) < threshold
    inlier_pts = pts[mask]
    centroid = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid)
    n = Vt[-1]
    if n[2] < 0:                      # 법선이 위를 향하도록
        n = -n
    d = -n @ centroid
    return np.array([n[0], n[1], n[2], d])


def project_onto_plane(point, plane):
    """점을 평면 위로 수직 투영."""
    p = np.asarray(point, dtype=float)
    n = np.asarray(plane[:3], dtype=float)
    d = float(plane[3])
    return p - (n @ p + d) * n


def height_above_plane(point, plane):
    p = np.asarray(point, dtype=float)
    return float(np.asarray(plane[:3]) @ p + plane[3])
