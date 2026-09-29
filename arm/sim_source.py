"""합성 장면 소스 — 하드웨어 0개로 전 구간(시선->앵커->파지 시퀀스)을 돌려본다.

목적은 "예쁜 시뮬레이션"이 아니라 **좌표계 사슬 검증**이다. 팔/카메라/태그가 없어도
아래 사슬이 수치적으로 맞는지 오늘 당장 확인할 수 있어야 한다:

    컵 픽셀 + 테이블 평면 -> p_hc -> (anchor) -> p_ab -> IK -> 관절각

그래서 컵의 armbase 진값(ground truth)을 알고 시작한 뒤, 카메라 픽셀로 투영했다가
파이프라인이 되짚어 복원한 값과 비교한다. 오차가 mm 단위로 안 나오면 어딘가 틀린 것이다.

ultralytics / pupil_apriltags / rclpy / lerobot 전부 필요 없다.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from perception import Cup

log = logging.getLogger(__name__)


def _rt(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).ravel()
    return T


def _euler(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class SimSource:
    """헤드캠이 팔 앞 0.7m 위에서 테이블을 내려다보는 상황을 만든다.

    armbase 좌표계: x=팔 정면, y=왼쪽, z=위.
    헤드캠 좌표계 : x=오른쪽, y=아래, z=전방 (OpenCV 관례).
    """

    def __init__(self, cfg: dict, K: np.ndarray, gaze_target: int = 1):
        self.cfg = cfg
        self.K = np.asarray(K, float)
        self.table_z = cfg["task"]["table_z"]
        self.width, self.height = 640, 480
        self.gaze_target = gaze_target
        self.t0 = time.time()

        # --- 헤드캠 자세 (armbase 기준) ---
        # 카메라는 팔 정면 0.55m, 테이블 위 0.35m 에서 25도 내려다본다.
        # 카메라 z축(전방)이 armbase -x 쪽 아래를 향하도록 만든다.
        # armbase -> 카메라. 행이 곧 카메라 축을 armbase 로 표현한 것이다.
        # 카메라는 팔 앞(+x)에서 팔을 마주본다 -> 전방 = -x_ab.
        # 팔을 마주보고 있으므로 팔의 왼쪽(+y_ab)이 화면 오른쪽에 찍힌다.
        # ★ det = +1 (진짜 회전) 이어야 한다. 예전엔 x_cam = -y_ab 라 det = -1 인 반사였고,
        #   cv2.Rodrigues 가 조용히 엉뚱한 값을 뱉었다 (test_pipeline.py 가 잡아냄).
        base = np.array([[0.0, 1.0, 0.0],      # x_cam(오른쪽) = +y_ab
                         [0.0, 0.0, -1.0],     # y_cam(아래)   = -z_ab
                         [-1.0, 0.0, 0.0]])    # z_cam(전방)   = -x_ab
        assert abs(np.linalg.det(base) - 1.0) < 1e-9, "base 가 회전행렬이 아니다"
        tilt = _euler(25.0, 0, 0)              # 카메라 x축(오른쪽) 기준으로 아래로 25도
        R_cam_ab = tilt @ base                 # world(armbase) -> cam 회전
        t_cam = np.array([0.55, 0.0, self.table_z + 0.35])   # armbase 기준 카메라 위치
        self.T_ab_hc = _rt(R_cam_ab.T, t_cam)                # p_ab = T_ab_hc @ p_hc
        self.T_hc_ab = np.linalg.inv(self.T_ab_hc)           # p_hc = T_hc_ab @ p_ab

        # --- SLAM: world 는 armbase 와 다른 임의의 원점 (실제 상황 재현) ---
        self.T_w_ab = _rt(_euler(3, -7, 40), [1.3, -0.6, 0.25])
        self.T_w_hc = self.T_w_ab @ self.T_ab_hc

        # --- 컵 3개 진값 (armbase 좌표, 테이블 위) ---
        cz = self.table_z + 0.04          # 컵 중심 높이
        self.cups_true_ab = [np.array([0.17, 0.09, cz]),
                             np.array([0.19, 0.00, cz]),
                             np.array([0.17, -0.09, cz])]
        self.frame_bgr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._log_setup()

    # ---------- 내부 ----------
    def _project_ab(self, p_ab):
        p_hc = (self.T_hc_ab @ np.r_[np.asarray(p_ab, float), 1.0])[:3]
        if p_hc[2] <= 0:
            return None
        uvw = self.K @ p_hc
        return np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]]), p_hc

    def _log_setup(self):
        log.info("[sim] 컵 진값(armbase): %s",
                 [np.round(c, 3).tolist() for c in self.cups_true_ab])
        for i, c in enumerate(self.cups_true_ab):
            uv, p_hc = self._project_ab(c)
            log.info("[sim]   컵%d -> 픽셀 %s, 헤드캠거리 %.3fm",
                     i, np.round(uv, 1).tolist(), np.linalg.norm(p_hc))

    # ---------- run_demo 가 쓰는 인터페이스 ----------
    def slam(self):
        return self.T_w_hc, 0, True

    def tag_pose(self):
        """태그 검출 대신 진값 T_hc_ab 를 그대로 준다 (검출기 없이 앵커 사슬만 검증)."""
        return self.T_hc_ab

    def cups(self):
        """컵 3개를 픽셀로 투영해 Cup 객체로 만든다. p_cam 은 비워둔다 —
        테이블 평면 교차가 이걸 복원해내야 정답이다."""
        out = []
        for i, c_ab in enumerate(self.cups_true_ab):
            uv, _ = self._project_ab(c_ab)
            base_ab = np.array([c_ab[0], c_ab[1], self.table_z])
            base_uv, _ = self._project_ab(base_ab)
            u, v = float(uv[0]), float(uv[1])
            out.append(Cup(idx=i, center_uv=(u, v),
                           bbox=(u - 25, v - 35, u + 25, v + 35),
                           conf=0.9, mask=None, p_cam=None,
                           base_uv=(float(base_uv[0]), float(base_uv[1]))))
        return out

    def frame(self):
        """시선은 gaze_target 컵 중심에 고정. dwell 이 찰 때까지 같은 값을 준다."""
        uv, _ = self._project_ab(self.cups_true_ab[self.gaze_target])
        return self.frame_bgr, (float(uv[0]), float(uv[1])), None

    def truth_ab(self):
        return self.cups_true_ab[self.gaze_target]
