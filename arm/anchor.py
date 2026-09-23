"""SLAM anchor latch — AprilTag 번들로 팔 좌표계를 world 에 등록하고 SLAM 으로 유지한다.

    T_hc_ab = inv(T_w_hc) @ T_w_ab

T_a_b 표기 = "b 의 자세를 a 좌표계로 나타낸 것" (즉 p_a = T_a_b @ p_b).

태그는 앵커 등록용, SLAM 은 앵커 유지용. 고개를 숙여 태그가 시야에서 사라져도
T_w_ab 가 latch 되어 있으면 팔 좌표계를 계속 알 수 있다.

★ ORB-SLAM3 는 트래킹을 놓치면 Atlas 에 새 맵을 만든다. 그 순간 world 원점이
재정의되므로 latch 는 "부정확"해지는 게 아니라 "무효"가 된다. map_id 를 반드시
같이 받아서 감시해야 한다. loop closure 만 끄는 걸로는 못 막는다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

log = logging.getLogger(__name__)


class AnchorState(Enum):
    UNANCHORED = "UNANCHORED"   # latch 없음/무효 -> 팔 이동 절대 금지
    STALE = "STALE"             # latch 있으나 오래됨/품질 저하 -> 재확인 요구
    ANCHORED = "ANCHORED"       # 동작 허용


def _rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def _inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return _rt(R.T, -R.T @ t)


def _slerp_pose(T_old: np.ndarray, T_new: np.ndarray, alpha: float) -> np.ndarray:
    """자세를 EMA 로 섞는다. 회전은 쿼터니언 slerp, 위치는 선형."""
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix(np.stack([T_old[:3, :3], T_new[:3, :3]]))
    R = Slerp([0.0, 1.0], rots)([alpha])[0].as_matrix()
    t = (1 - alpha) * T_old[:3, 3] + alpha * T_new[:3, 3]
    return _rt(R, t)


@dataclass
class AnchorTracker:
    """SLAM pose 와 태그 관측을 받아 T_hc_ab 를 제공한다.

    ROS 비의존. update_slam / update_tag 만 먹여주면 된다.
    """

    stale_after_s: float = 30.0
    drift_warn_m: float = 0.02
    drift_max_m: float = 0.05
    latch_ema_alpha: float = 0.25

    T_w_hc: np.ndarray | None = None      # SLAM 이 주는 헤드캠 자세
    map_id: int | None = None
    slam_ok: bool = False
    _slam_t: float = 0.0

    T_w_ab: np.ndarray | None = None      # latch 된 팔 베이스 자세 (world)
    latched_map_id: int | None = None
    last_latch_t: float = 0.0
    last_drift_m: float = float("nan")
    drift_history: list = field(default_factory=list)

    # ---------- 입력 ----------
    def update_slam(self, T_w_hc: np.ndarray, map_id: int, tracking_ok: bool, t: float | None = None):
        t = time.time() if t is None else t
        if self.latched_map_id is not None and map_id != self.latched_map_id:
            log.error("Atlas map 전환 감지 (%s -> %s): latch 무효화", self.latched_map_id, map_id)
            self.T_w_ab = None
            self.latched_map_id = None
        self.T_w_hc, self.map_id, self.slam_ok, self._slam_t = T_w_hc, map_id, tracking_ok, t

    def update_tag(self, T_hc_ab_meas: np.ndarray, t: float | None = None):
        """태그가 보일 때마다 호출. latch 를 생성하거나 갱신한다."""
        t = time.time() if t is None else t
        if self.T_w_hc is None or not self.slam_ok:
            log.debug("SLAM pose 없음 -> latch 보류")
            return
        T_w_ab_meas = self.T_w_hc @ T_hc_ab_meas

        if self.T_w_ab is None or self.latched_map_id != self.map_id:
            self.T_w_ab = T_w_ab_meas
            self.latched_map_id = self.map_id
            self.last_drift_m = 0.0
            log.info("ANCHOR latched (map %s)", self.map_id)
        else:
            # 기존 latch 와 새 관측의 차이 = SLAM anchor drift. 이게 평가 지표다.
            drift = float(np.linalg.norm(T_w_ab_meas[:3, 3] - self.T_w_ab[:3, 3]))
            self.last_drift_m = drift
            self.drift_history.append((t, drift))
            if drift > self.drift_max_m:
                log.error("anchor drift %.1fcm > 한계 -> 강제 재latch", drift * 100)
                self.T_w_ab = T_w_ab_meas
            else:
                if drift > self.drift_warn_m:
                    log.warning("anchor drift %.1fcm", drift * 100)
                self.T_w_ab = _slerp_pose(self.T_w_ab, T_w_ab_meas, self.latch_ema_alpha)
        self.last_latch_t = t

    # ---------- 출력 ----------
    @property
    def state(self) -> AnchorState:
        if self.T_w_ab is None or self.T_w_hc is None:
            return AnchorState.UNANCHORED
        if self.latched_map_id != self.map_id:
            return AnchorState.UNANCHORED
        if not self.slam_ok:
            return AnchorState.STALE
        if time.time() - self.last_latch_t > self.stale_after_s:
            return AnchorState.STALE
        if self.last_drift_m == self.last_drift_m and self.last_drift_m > self.drift_max_m:
            return AnchorState.STALE
        return AnchorState.ANCHORED

    def T_headcam_to_armbase(self) -> np.ndarray | None:
        """p_hc = T @ p_ab. ANCHORED 가 아니면 None 을 준다."""
        if self.state is not AnchorState.ANCHORED:
            return None
        return _inv(self.T_w_hc) @ self.T_w_ab

    def point_headcam_to_armbase(self, p_hc) -> np.ndarray | None:
        """헤드캠 좌표의 3D 점을 팔 베이스 좌표로 변환. 시선 히트포인트 -> 컵 위치."""
        T = self.T_headcam_to_armbase()
        if T is None:
            return None
        p = np.ones(4)
        p[:3] = np.asarray(p_hc, float).ravel()
        return (_inv(T) @ p)[:3]

    def can_execute_grasp(self, max_latch_age_s: float = 30.0) -> tuple[bool, str]:
        """★ 팔을 움직이기 직전 게이트. 액체를 얼굴로 옮기는 동작이라 협상 대상 아님."""
        st = self.state
        if st is not AnchorState.ANCHORED:
            return False, f"anchor 상태 {st.value} — 팔 쪽 태그를 한 번 봐 주세요"
        age = time.time() - self.last_latch_t
        if age > max_latch_age_s:
            return False, f"latch 가 {age:.0f}초 지났습니다 — 팔 쪽 태그를 다시 봐 주세요"
        return True, "ok"


def build_bundle_obj_pts(anchor_cfg: dict):
    """config 의 anchor 섹션 -> {tag_id: (4,3) 코너 좌표(armbase)}, right, up, normal.

    pupil_apriltags 코너 순서는 (좌하, 우하, 우상, 좌상). 그 "오른쪽/위"가 armbase 의
    어느 축인지는 태그를 어떻게 붙였느냐의 문제라 config(anchor.tag_axes)로 명시한다.
    검출기 없이도 부를 수 있게 모듈 함수로 뺐다 (테스트용).
    """
    def _axes(src, default_right, default_up):
        r = np.asarray(src.get("right", default_right), float)
        u = np.asarray(src.get("up", default_up), float)
        r = r / np.linalg.norm(r)
        u = u / np.linalg.norm(u)
        if abs(float(r @ u)) > 1e-6:
            raise ValueError("tag_axes: right 와 up 은 직교해야 한다")
        return r, u

    axes = anchor_cfg.get("tag_axes", {})
    right, up = _axes(axes, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    normal = np.cross(right, up)

    h = anchor_cfg["tag_size_m"] / 2.0
    obj = {}
    for t in anchor_cfg["bundle"]:
        # 태그마다 다른 면에 붙일 수 있다 (예: 수직판 2장 + 테이블에 눕힌 2장).
        # 그럴 땐 bundle 항목에 right/up 을 각각 적는다. 없으면 전역 tag_axes 를 쓴다.
        r, u = _axes(t, right, up)
        local = np.stack([-h * r - h * u, +h * r - h * u, +h * r + h * u, -h * r + h * u])
        obj[t["id"]] = np.asarray(t["pos"], float) + local
    return obj, right, up, normal


def _warn_if_coplanar(positions, tol_m: float = 0.01):
    """번들이 한 평면에 몰려 있으면 PnP 회전이 구조적으로 불안정해진다."""
    P = np.asarray(positions, float)
    if len(P) < 3:
        return
    spread = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    if spread[-1] < tol_m:
        log.warning("AprilTag 번들이 거의 동일평면이다 (두께 %.1fmm). 평면 PnP 는 광축 "
                    "방향 회전이 불안정하다 — 태그 하나를 깊이 방향으로 4~6cm 어긋나게 "
                    "붙이면 크게 좋아진다.", spread[-1] * 1000)


class TagBundleDetector:
    """AprilTag 번들 -> T_hc_ab. 단일 태그의 rotation flip 을 피하려고 여러 태그를 한 번에 PnP 한다.

    ★ 코너 순서 / 축 방향 (2026-08-19 수정)
    pupil_apriltags 는 코너를 태그 자체 기준으로 (좌하, 우하, 우상, 좌상) 순서로 준다.
    그 "오른쪽/위"가 armbase 의 어느 축인지는 **태그를 어떻게 붙였느냐**의 문제라서
    코드가 마음대로 정할 수 없다 -> config 의 anchor.tag_axes 로 명시한다.

    예전 구현은 로컬 코너를 [(+y,-z), (-y,-z), (-y,+z), (+y,+z)] 로 박아놨는데, 이건
    right = -y, up = +z 이므로 법선(right x up) = -x 다. 그런데 config 주석은
    "태그 평면 법선은 +x(팔 정면)" 이라고 되어 있었다. 즉 코드와 문서가 서로 반대였고,
    실제로는 y 축이 뒤집힌 **거울상** 배치를 PnP 에 넣고 있었다 (det = -1, 회전으로 표현
    불가능 -> 재투영 오차가 크게 남거나 엉뚱한 자세가 나온다).

    이제 법선 +x / right +y / up +z 가 기본값이고, 재투영 오차로 자동 검증한다.
    """

    def __init__(self, cfg: dict, K: np.ndarray, dist=None):
        import cv2
        from pupil_apriltags import Detector

        self.cv2 = cv2
        a = cfg["anchor"]
        self.det = Detector(families=a["tag_family"], nthreads=4,
                            quad_decimate=1.0, refine_edges=True)
        self.K = np.asarray(K, float)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, float)
        self.size = a["tag_size_m"]
        self.min_tags = a["min_tags_for_latch"]
        self.max_reproj_px = a.get("max_reproj_px", 3.0)
        self.ambiguity_max_m = a.get("ambiguity_max_m", 0.02)

        self.obj_pts, self.right, self.up, self.normal = build_bundle_obj_pts(a)
        _warn_if_coplanar([t["pos"] for t in a["bundle"]])

    def detect(self, gray: np.ndarray) -> np.ndarray | None:
        dets = [d for d in self.det.detect(gray) if d.tag_id in self.obj_pts]
        if len(dets) < self.min_tags:
            return None
        obj = np.concatenate([self.obj_pts[d.tag_id] for d in dets])
        img = np.concatenate([np.asarray(d.corners, float) for d in dets])
        # ★ 평면 모호성 검사 (2026-08-19).
        # 태그를 전부 한 평면에 붙이면, 태그판을 기준으로 **거울 반사된** 자세가 픽셀상
        # 똑같이 투영된다. 재투영 오차가 0 이어도 팔 베이스가 10cm 넘게 엉뚱한 데 찍힌다
        # (합성 검증에서 정확히 120mm = 2 x 태그판 깊이만큼 틀렸다).
        # solvePnPGeneric 으로 후보를 전부 받아, 서로 크게 다르면 관측을 버린다.
        n_sol, rvecs, tvecs, _ = self.cv2.solvePnPGeneric(
            obj, img, self.K, self.dist, flags=self.cv2.SOLVEPNP_ITERATIVE
        )
        if n_sol < 1:
            return None
        cands = [(np.asarray(r), np.asarray(t)) for r, t in zip(rvecs, tvecs)]
        # cheirality: 태그가 카메라 **뒤**에 있는 해는 물리적으로 불가능하다.
        # 투영은 (x,y,z) -> (-x,-y,-z) 에 대해 불변이라 이런 해가 실제로 나온다
        # (합성 검증에서 t 가 정확히 부호만 뒤집힌 해가 재투영 0px 로 나왔다).
        def _in_front(rt):
            R, _ = self.cv2.Rodrigues(rt[0])
            return (((R @ obj.T).T + np.asarray(rt[1]).ravel())[:, 2] > 0).all()
        cands = [c for c in cands if _in_front(c)]
        if not cands:
            log.warning("태그 PnP 해가 전부 카메라 뒤 — 관측 버림")
            return None
        if len(cands) > 1:
            spread = max(float(np.linalg.norm(a[1] - b[1]))
                         for i, a in enumerate(cands) for b in cands[i + 1:])
            if spread > self.ambiguity_max_m:
                log.error("태그 PnP 해가 %d개이고 서로 %.1fcm 떨어져 있다 — 동일평면 모호성. "
                          "관측 버림. 태그 하나를 깊이 방향으로 4~6cm 어긋나게 붙일 것.",
                          len(cands), spread * 100)
                return None
        rvec, tvec = cands[0]
        rvec, tvec = self.cv2.solvePnPRefineLM(obj, img, self.K, self.dist, rvec, tvec)

        # ★ 재투영 오차 검사 — 코너 순서/축 방향이 틀리면 여기서 걸린다.
        proj, _ = self.cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        err = float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(axis=1).mean()))
        self.last_reproj_px = err
        if err > self.max_reproj_px:
            log.warning("태그 PnP 재투영 오차 %.1fpx > %.1fpx — 관측 버림. "
                        "번들 좌표(anchor.bundle) 또는 코너 축(anchor.tag_axes) 확인.",
                        err, self.max_reproj_px)
            return None

        R, _ = self.cv2.Rodrigues(rvec)
        return _rt(R, tvec)   # T_hc_ab
