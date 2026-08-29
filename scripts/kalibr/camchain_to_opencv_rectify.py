#!/usr/bin/env python3
"""Kalibr camchain.yaml(K,D,extrinsic) -> OpenCV rectification 포맷(K,D,R,P)
left_opencv.yaml/right_opencv.yaml로 변환.

2026-08-11 재캘리브레이션 때 쓴 것과 같은 절차(docs/11_camera_imu_calibration.md
"재작업 절차" 4단계)인데, 그때 스크립트는 커밋되지 않고 1회성으로 실행돼서 이번에
다시 만들었다. cv2.stereoRectify()가 하는 일: Kalibr가 raw(왜곡보정 전) 영상 기준으로
푼 K0/D0/K1/D1/R/T를 받아서, remap 한 번으로 왜곡보정+정렬(rectify)까지 되는
R1/R2(회전)와 P1/P2(정렬된 좌표계의 새 카메라 행렬 + baseline)를 계산한다.

Kalibr의 T_cn_cnm1(cam1 == cam_n, cam0 == cam_{n-1})은 "p_cam1 = T_cn_cnm1 @ p_cam0"
정의라, R=T_cn_cnm1[:3,:3], T=T_cn_cnm1[:3,3]가 cv2.stereoRectify()가 기대하는
"p_right = R @ p_left + T" 규약과 그대로 맞아떨어진다 — 별도 변환 불필요.
"""
import argparse
import sys

import cv2
import numpy as np
import yaml

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def load_cam(camchain, key):
    c = camchain[key]
    fx, fy, cx, cy = c["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    d = list(c["distortion_coeffs"])
    if len(d) == 4:
        d = d + [0.0]  # radtan(k1,k2,p1,p2) -> plumb_bob(k1,k2,p1,p2,k3=0)
    D = np.array(d, dtype=np.float64)
    return K, D


def write_opencv_yaml(path, K, D, R, P):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", IMAGE_WIDTH)
    fs.write("image_height", IMAGE_HEIGHT)
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", D.reshape(1, -1))
    fs.write("rectification_matrix", R)
    fs.write("projection_matrix", P)
    fs.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camchain", help="kalibr_calibrate_cameras가 만든 camchain.yaml")
    ap.add_argument("left_out")
    ap.add_argument("right_out")
    args = ap.parse_args()

    with open(args.camchain) as f:
        camchain = yaml.safe_load(f)

    K0, D0 = load_cam(camchain, "cam0")
    K1, D1 = load_cam(camchain, "cam1")
    T_cn_cnm1 = np.array(camchain["cam1"]["T_cn_cnm1"], dtype=np.float64)
    R = T_cn_cnm1[:3, :3]
    T = T_cn_cnm1[:3, 3]
    baseline_m = float(np.linalg.norm(T))
    print(f"[convert] Kalibr baseline |T| = {baseline_m*100:.3f} cm")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K0, D0, K1, D1, (IMAGE_WIDTH, IMAGE_HEIGHT), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    write_opencv_yaml(args.left_out, K0, D0, R1, P1)
    write_opencv_yaml(args.right_out, K1, D1, R2, P2)

    rectified_fx = P1[0, 0]
    rectified_baseline = -P2[0, 3] / P2[0, 0]
    print(f"[convert] rectified fx = {rectified_fx:.3f}px, "
          f"rectified baseline = {rectified_baseline*100:.3f} cm")
    print(f"[convert] 저장 완료: {args.left_out}, {args.right_out}")


if __name__ == "__main__":
    sys.exit(main())
