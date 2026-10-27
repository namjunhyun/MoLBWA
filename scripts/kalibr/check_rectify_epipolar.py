#!/usr/bin/env python3
"""정렬(rectify) 검증: 같은 태그 모서리가 정렬 뒤 좌우 영상에서 같은 줄(y)에 오는가.
    python3 scripts/kalibr/check_rectify_epipolar.py calibration/bags/<이름> ros2_ws/src/uvc_stereo/config
세로 어긋남 |Δy| 중앙이 1.5 px 안쪽이면 된다. 이 측정은 OpenCV 모서리 검출(사진당 1~2 px 잡음)에 묶여 있어
1 px 아래로는 안 내려간다 — 2026-10-27 같은 bag 에서 우리 결과 1.13 px, Kalibr 스테레오 결과 1.11 px 로 같았다.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stereo_extrinsic_opencv as S  # noqa: E402


def load(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    out = [fs.getNode(k).mat() for k in ("camera_matrix", "distortion_coefficients", "rectification_matrix", "projection_matrix")]
    fs.release()
    return out


bag, cfg = sys.argv[1], sys.argv[2]
L, R = load(f"{cfg}/left_opencv.yaml"), load(f"{cfg}/right_opencv.yaml")
dy, n = [], 0
for dl, dr in S.detections(bag, 4.0, 4):
    for i in set(dl) & set(dr):
        pl = cv2.undistortPoints(dl[i].reshape(-1, 1, 2).astype(np.float64), L[0], L[1], R=L[2], P=L[3]).reshape(-1, 2)
        pr = cv2.undistortPoints(dr[i].reshape(-1, 1, 2).astype(np.float64), R[0], R[1], R=R[2], P=R[3]).reshape(-1, 2)
        dy.extend(np.abs(pl[:, 1] - pr[:, 1])); n += 1
dy = np.array(dy)
print(f"태그 {n}개·모서리 {len(dy)}개: 세로 어긋남 |Δy| 중앙 {np.median(dy):.2f} px · 90% {np.percentile(dy, 90):.2f} px · 최대 {dy.max():.2f} px")
print("판정:", "좋음 (≤1.5 px, 측정 한계 ~1.1 px)" if np.median(dy) <= 1.5 else "나쁨 — 캘리브레이션 다시 확인")
