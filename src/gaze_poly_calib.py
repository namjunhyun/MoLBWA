"""시선 씬 픽셀 캘리브 — 동공 중심 -> 씬 카메라 픽셀 2차 회귀 + 시차항 (2026-10-30).

왜 (R, p_eye) 대신 이걸 하나
---------------------------
10-26 캘리브(14점, 잔차 6.5cm)를 다시 보니 p_eye 가 세 축 모두 ±10cm 경계에 붙었고,
Orlosky 고정 구 시선 벡터가 실제 각도의 0.46(가로)/0.35(세로)배만 움직이다 -38° 근처에서
포화됐다. 같은 자세 반복은 0.4° 안으로 맞았으니 노이즈가 아니라 입력 자체의 휨이다.
-> 안구 모델을 거치지 않고 동공 중심 픽셀을 씬 픽셀로 바로 회귀한다(휨까지 흡수).
   정답 픽셀은 클릭 대신 표적 시트(tools/make_gaze_target_sheet.py)의 과녁을 태그로 계산한다.
   팔·클립보드 태그·머리 자세가 전혀 안 끼므로 FK 오차와 태그 가림이 캘리브에서 빠진다.

모델 (축마다 7계수)
    u = a0 + a1 x + a2 y + a3 x² + a4 xy + a5 y² + c / d
x, y = 정규화한 동공 중심, d = 과녁까지 깊이(씬 카메라 z). c/d 는 눈과 씬 카메라가 떨어져
생기는 시차 — 1/d 에 비례하므로 두 거리(가까이·멀리)에서 찍어야 c 가 정해진다.
이 꼴이면 같은 동공에 대한 3D 점 X(d) 가 d 에 대해 직선이라, 시선 광선이 정확히 나온다:
    X(d) = A d + B,  A = ((a_u-cx)/fx, (a_v-cy)/fy, 1),  B = (c_u/fx, c_v/fy, 0)
-> gaze_tag_bridge 에 (origin=B, direction=A) 로 보낸다. 깊이를 몰라도 된다.

절차: p = 가까이(0.35~0.8m) 9칸 -> 멀리(0.85~1.6m) 9칸 -> 계산·저장 / v = 5x5 검증(각도 오차) /
      j = 1점 보정(상수항만, 세션 한정) / n = 이번 칸 건너뛰기.
칸 맞추기: 과녁을 계속 보면서 **고개를 돌려** 과녁이 노란 칸에 들어오게 -> 1초 유지되면 자동 기록.
"""
import json
import os
import time

import cv2
import numpy as np
import yaml

try:
    from pupil_apriltags import Detector
except ImportError:          # 표적 검출 없이도 저장된 모델로 시선 계산은 된다
    Detector = None

NEAR = (0.35, 0.80)
FAR = (0.85, 1.60)


def grid_cells(n, w, h, margin=0.15):
    """n x n 칸 (cx, cy, 반폭, 반높이). 가장자리에서 margin 안쪽에 중심을 둔다."""
    xs = np.linspace(margin, 1 - margin, n) * w
    ys = np.linspace(margin, 1 - margin, n) * h
    hw, hh = w / (2.0 * (n + 2)), h / (2.0 * (n + 2))
    return [(float(x), float(y), hw, hh) for y in ys for x in xs]


def pixel_angle_deg(K, uv_a, uv_b):
    """같은 카메라의 두 픽셀이 이루는 각도."""
    Ki = np.linalg.inv(K)
    a = Ki @ np.array([uv_a[0], uv_a[1], 1.0])
    b = Ki @ np.array([uv_b[0], uv_b[1], 1.0])
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class TargetDetector:
    """표적 시트의 과녁(원점) 픽셀과 깊이. 태그 한 장만 보여도 된다."""

    def __init__(self, layout_path, K):
        with open(layout_path) as f:
            lay = yaml.safe_load(f)
        self.K = np.asarray(K, dtype=np.float64)
        s = float(lay["tag_size_m"]) / 2.0
        # 시트 좌표: x 오른쪽, y 아래, z 시트 안쪽 (카메라가 보는 방향).
        # 태그 자체 좌표의 모서리 순서는 (-s,s),(s,s),(s,-s),(-s,-s) (IPPE_SQUARE 규약)인데,
        # cv2.aruco 로 찍은 시트는 태그 좌표가 시트에 대해 180° 돌아 있다 — 실측(2026-10-30):
        # corners[0] 이 똑바로 놓인 시트의 우상단. 그래서 시트 좌표 = c - (태그 좌표).
        self.canon = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        self.centers = {int(t): np.array([cx, cy, 0.0]) for t, (cx, cy) in lay["tags"].items()}
        self.obj = {t: c - self.canon for t, c in self.centers.items()}
        self.det = Detector(families=lay.get("tag_family", "tag36h11")) if Detector else None

    def detect(self, gray):
        """-> (uv(2,), depth_m, 태그 수) 또는 None. gray 는 뒤집지 않은 rectified 좌영상."""
        if self.det is None:
            return None
        found = [d for d in self.det.detect(gray) if d.tag_id in self.obj and d.decision_margin > 20]
        if not found:
            return None
        img = np.concatenate([np.asarray(d.corners, dtype=np.float64) for d in found])
        if len(found) > 1:
            obj = np.concatenate([self.obj[d.tag_id] for d in found])
            ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, None, flags=cv2.SOLVEPNP_IPPE)
            origin = np.zeros(3)
        else:
            # 한 장: 태그 좌표로 풀고, 과녁(시트 원점)을 태그 좌표로 옮긴다 (= c, 위 180° 참고)
            ok, rvec, tvec = cv2.solvePnP(self.canon, img, self.K, None,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            origin = self.centers[found[0].tag_id]
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        X = R @ origin + tvec.ravel()
        if X[2] <= 0.05:
            return None
        uv = (self.K @ (X / X[2]))[:2]
        return uv, float(X[2]), len(found)


class PolyModel:
    """동공 (px, py) + 깊이 -> 씬 픽셀. 2차 다항 + 1/d."""

    def __init__(self, coef_u, coef_v, norm, K):
        self.coef_u = np.asarray(coef_u, dtype=np.float64)
        self.coef_v = np.asarray(coef_v, dtype=np.float64)
        self.norm = np.asarray(norm, dtype=np.float64)       # [mx, my, sx, sy]
        self.K = np.asarray(K, dtype=np.float64)

    @staticmethod
    def _phi(xy, norm):
        x = (xy[..., 0] - norm[0]) / norm[2]
        y = (xy[..., 1] - norm[1]) / norm[3]
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)

    @classmethod
    def fit(cls, pupils, uvs, depths, K, norm=None):
        P = np.asarray(pupils, dtype=np.float64)
        U = np.asarray(uvs, dtype=np.float64)
        D = np.asarray(depths, dtype=np.float64)
        if norm is None:
            norm = np.array([P[:, 0].mean(), P[:, 1].mean(),
                             max(P[:, 0].std(), 1.0), max(P[:, 1].std(), 1.0)])
        A = np.column_stack([cls._phi(P, norm), 1.0 / D])
        if len(P) < A.shape[1] + 2:
            raise ValueError(f"점 {len(P)}개 — {A.shape[1] + 2}개 이상 필요")
        if np.ptp(1.0 / D) < 0.3:
            raise ValueError("깊이 폭이 좁아 시차항을 못 정한다 — 가까이·멀리 둘 다 찍을 것")
        reg = 1e-6 * np.eye(A.shape[1])
        cu = np.linalg.solve(A.T @ A + reg, A.T @ U[:, 0])
        cv = np.linalg.solve(A.T @ A + reg, A.T @ U[:, 1])
        return cls(cu, cv, norm, K)

    def predict(self, pupil, depth):
        phi = self._phi(np.asarray(pupil, dtype=np.float64), self.norm)
        return np.array([phi @ self.coef_u[:6] + self.coef_u[6] / depth,
                         phi @ self.coef_v[:6] + self.coef_v[6] / depth])

    def ray(self, pupil):
        """씬 카메라 좌표의 시선 광선 (origin, unit direction). 위 X(d) = A d + B."""
        phi = self._phi(np.asarray(pupil, dtype=np.float64), self.norm)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        A = np.array([(phi @ self.coef_u[:6] - cx) / fx, (phi @ self.coef_v[:6] - cy) / fy, 1.0])
        B = np.array([self.coef_u[6] / fx, self.coef_v[6] / fy, 0.0])
        return B, A / np.linalg.norm(A)

    def shift(self, du, dv):
        """1점 보정 — 상수항만 옮긴다 (안경이 조금 흘러내린 만큼)."""
        self.coef_u[0] += du
        self.coef_v[0] += dv

    def to_dict(self):
        return {"coef_u": self.coef_u.tolist(), "coef_v": self.coef_v.tolist(),
                "norm": self.norm.tolist(), "K": self.K.tolist()}


def angle_errors(model, pupils, uvs, depths):
    return np.array([pixel_angle_deg(model.K, model.predict(p, d), uv)
                     for p, uv, d in zip(pupils, uvs, depths)])


def loo_errors(pupils, uvs, depths, K):
    """한 점씩 빼고 맞춘 뒤 그 점의 각도 오차 — 처음 보는 곳 정확도."""
    out = []
    for i in range(len(pupils)):
        keep = [j for j in range(len(pupils)) if j != i]
        try:
            m = PolyModel.fit([pupils[j] for j in keep], [uvs[j] for j in keep],
                              [depths[j] for j in keep], K)
        except (ValueError, np.linalg.LinAlgError):
            out.append(np.nan)
            continue
        out.append(pixel_angle_deg(K, m.predict(pupils[i], depths[i]), uvs[i]))
    return np.array(out)


def save_model(path, model, pupils, uvs, depths, meta):
    data = {"version": 1, "model": "pupil_poly2_parallax", "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **model.to_dict(), "pupils": np.asarray(pupils).tolist(), "uvs": np.asarray(uvs).tolist(),
            "depths": list(map(float, depths)), **meta}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def load_model(path):
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    return PolyModel(d["coef_u"], d["coef_v"], d["norm"], d["K"]), d


class PolyCalibSession:
    """칸 안내 + 자동 기록. update() 를 매 프레임 부르고, 끝난 단계는 반환값으로 알린다.

    kind: "calib" (가까이 9 -> 멀리 9) / "val" (5x5, 깊이 무관) / "drift" (아무 칸, 1점)
    """

    HOLD_S, SKIP_S = 1.0, 0.3            # 1초 유지, 앞 0.3초는 버림 (과녁에 눈이 도착하는 시간)
    PUPIL_TOL_PX, TARGET_TOL_PX = 3.0, 8.0

    def __init__(self, kind, w, h):
        self.kind = kind
        if kind == "calib":
            self.phases = [("near", grid_cells(3, w, h), NEAR), ("far", grid_cells(3, w, h), FAR)]
        elif kind == "val":
            self.phases = [("val", grid_cells(5, w, h), (0.3, 2.0))]
        else:
            self.phases = [("drift", [(w / 2.0, h / 2.0, w / 2.0, h / 2.0)], (0.3, 2.0))]
        self.phase, self.cell = 0, 0
        self.samples = []                  # (phase, cell, pupil, uv, depth)
        self.buf = []
        self.msg, self._msg_t = "", 0.0
        self.last_target = None

    @property
    def done(self):
        return self.phase >= len(self.phases)

    def current(self):
        name, cells, drange = self.phases[self.phase]
        return name, cells, cells[self.cell], drange

    def _say(self, text, now, every=1.5):
        self.msg = text
        if now - self._msg_t > every:
            print(f"[poly] {text}", flush=True)
            self._msg_t = now

    def skip(self):
        if self.done:
            return
        print(f"[poly] 칸 {self.cell + 1} 건너뜀")
        self._advance()

    def _advance(self):
        self.buf = []
        _, cells, _ = self.phases[self.phase]
        self.cell += 1
        if self.cell >= len(cells):
            self.phase, self.cell = self.phase + 1, 0
            if not self.done:
                name = self.phases[self.phase][0]
                print(f"[poly] 다음 단계: {name} — "
                      + ("의자를 뒤로 빼서 0.85~1.6m 에서" if name == "far" else ""), flush=True)

    def update(self, now, pupil, target):
        """pupil = (px, py) 또는 None, target = TargetDetector.detect 결과 또는 None.
        -> 기록했으면 True."""
        if self.done:
            return False
        self.last_target = target
        name, _, (cx, cy, hw, hh), (dmin, dmax) = self.current()
        ok = True
        if target is None:
            ok = False
            self._say("표적이 안 보임", now)
        elif pupil is None:
            ok = False
            self._say("동공이 안 잡힘", now)
        else:
            uv, depth, _n = target
            if not (dmin <= depth <= dmax):
                ok = False
                self._say(f"거리 {depth:.2f}m — {dmin:.2f}~{dmax:.2f}m 로", now)
            elif abs(uv[0] - cx) > hw or abs(uv[1] - cy) > hh:
                ok = False
                self.msg = "과녁을 보면서 고개를 돌려 노란 칸에"
        if not ok:
            self.buf = []
            return False
        self.buf.append((now, np.asarray(pupil, dtype=np.float64), target[0].copy(), target[1]))
        if now - self.buf[0][0] < self.HOLD_S:
            self.msg = f"유지 {now - self.buf[0][0]:.1f}s"
            return False
        win = [b for b in self.buf if b[0] >= self.buf[0][0] + self.SKIP_S]
        P = np.array([b[1] for b in win])
        U = np.array([b[2] for b in win])
        pm, um = np.median(P, axis=0), np.median(U, axis=0)
        p_spread = float(np.linalg.norm(P - pm, axis=1).max())
        u_spread = float(np.linalg.norm(U - um, axis=1).max())
        if len(win) < 5 or p_spread > self.PUPIL_TOL_PX or u_spread > self.TARGET_TOL_PX:
            self._say(f"흔들림 — 동공 {p_spread:.1f}px / 표적 {u_spread:.1f}px (고개·눈 멈추기)", now, 0.5)
            self.buf = []
            return False
        depth = float(np.median([b[3] for b in win]))
        self.samples.append((name, self.cell, pm, um, depth))
        print(f"[poly] {name} 칸 {self.cell + 1} 기록: 동공 {np.round(pm, 1)} -> 픽셀 {np.round(um, 1)} "
              f"깊이 {depth:.2f}m (동공 흔들림 {p_spread:.1f}px)", flush=True)
        self._advance()
        return True

    def draw(self, img, flip=False):
        """칸·표적·진행 상황. flip = 씬 화면을 180° 돌려 보여줄 때."""
        h, w = img.shape[:2]
        f = (lambda x, y: (w - 1 - x, h - 1 - y)) if flip else (lambda x, y: (x, y))
        if not self.done:
            name, cells, _, (dmin, dmax) = self.current()
            done_cells = {s[1] for s in self.samples if s[0] == name}
            for i, (cx, cy, hw, hh) in enumerate(cells):
                col = (0, 220, 255) if i == self.cell else ((0, 200, 0) if i in done_cells else (90, 90, 90))
                p1, p2 = f(cx - hw, cy - hh), f(cx + hw, cy + hh)
                cv2.rectangle(img, tuple(map(int, p1)), tuple(map(int, p2)), col, 3 if i == self.cell else 1)
            n_cells = len(cells)
            cv2.putText(img, f"POLY {name} {self.cell + 1}/{n_cells}  {dmin:.2f}-{dmax:.2f}m  {self.msg}",
                        (20, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        if self.last_target is not None:
            uv, depth, n = self.last_target
            p = tuple(map(int, f(uv[0], uv[1])))
            cv2.drawMarker(img, p, (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(img, f"{depth:.2f}m tags={n}", (p[0] + 14, p[1] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
