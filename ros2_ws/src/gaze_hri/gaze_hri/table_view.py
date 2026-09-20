#!/usr/bin/env python3
"""
탑다운 카메라 ↔ 테이블 평면 공통 로직.

`topdown_click`(캘리브레이션)과 `control_panel`(통합 GUI)이 같이 쓴다.
카메라 열기, 호모그래피 로드/적용, 색 기반 물체 검출이 전부 여기 있다.

호모그래피는 "화면 픽셀 ↔ 로봇 베이스 (x, y)" 대응이다. 책상이 평면이고
카메라가 고정이라 3×3 행렬 하나로 끝난다. 깊이도 SLAM도 필요 없다.
"""

import os

import cv2
import numpy as np
import yaml


class TableHomography:
    """화면 픽셀 <-> 로봇 베이스 좌표 변환."""

    def __init__(self):
        self.H = None
        self.H_inv = None
        self.reproj_error_mm = None
        self.num_points = None

    @property
    def ready(self):
        return self.H is not None

    def load(self, path):
        """저장된 캘리브레이션을 읽는다. 실패 시 (False, 사유)."""
        if not os.path.exists(path):
            return False, f"파일이 없습니다: {path}"
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            H = np.array(data["H"], dtype=float)
            if H.shape != (3, 3):
                return False, "H 가 3x3 이 아닙니다"
            self.set(H)
            self.reproj_error_mm = data.get("reproj_error_mm")
            self.num_points = data.get("num_points")
        except Exception as exc:                        # noqa: BLE001
            return False, f"읽기 실패: {exc}"
        return True, None

    def set(self, H):
        self.H = np.asarray(H, dtype=float)
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H_inv = None

    def to_robot(self, u, v):
        """화면 픽셀 -> 로봇 (x, y). 한 번의 행렬 곱."""
        if self.H is None:
            return None
        p = self.H @ np.array([float(u), float(v), 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_pixel(self, x, y):
        """로봇 (x, y) -> 화면 픽셀. 선택 결과를 영상 위에 되돌려 그릴 때 쓴다."""
        if self.H_inv is None:
            return None
        p = self.H_inv @ np.array([float(x), float(y), 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])


class ObjectDetector:
    """색(HSV) 기반 물체 검출.

    컵 색이 일정하다는 전제다. YOLO 를 붙일 거면 이 클래스만 갈아끼우면 된다
    (반환 형식만 맞추면 나머지는 그대로 돈다).

    빨강은 HSV 색상환의 양 끝(0 근처와 180 근처)에 걸쳐 있어서, 하한이 상한보다
    크면 두 구간을 OR 로 합친다. 붉은 컵의 절반을 놓치는 흔한 함정을 막는다.
    """

    def __init__(self, hsv_lower, hsv_upper, min_area=400):
        self.lower = np.array(hsv_lower, dtype=np.uint8)
        self.upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = int(min_area)

    def mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo, hi = self.lower, self.upper
        if int(lo[0]) > int(hi[0]):      # 색상환을 넘어가는 범위(예: 170~10)
            a = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8),
                            np.array([hi[0], hi[1], hi[2]], np.uint8))
            b = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]], np.uint8),
                            np.array([179, hi[1], hi[2]], np.uint8))
            m = cv2.bitwise_or(a, b)
        else:
            m = cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def detect(self, frame, homography):
        """검출 결과를 [(픽셀(u,v), 로봇(x,y), contour), ...] 로 돌려준다.

        호모그래피가 없으면 로봇 좌표를 낼 수 없으므로 빈 목록.
        """
        if not homography.ready:
            return []
        contours, _ = cv2.findContours(self.mask(frame), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
            xy = homography.to_robot(u, v)
            if xy is not None:
                found.append(((u, v), xy, c))
        # 로봇 기준 왼쪽 -> 오른쪽 순으로 번호가 안정되게 정렬
        found.sort(key=lambda it: it[1][1])
        return found


def open_camera(index, width, height):
    """카메라를 열고 해상도를 맞춘다. 실패하면 RuntimeError."""
    cap = cv2.VideoCapture(int(index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if not cap.isOpened():
        raise RuntimeError(
            f"카메라 {index} 를 열 수 없습니다. `ls /dev/video*` 로 인덱스를 "
            "확인하세요. 다른 프로그램이 쓰고 있어도 안 열립니다."
        )
    return cap
