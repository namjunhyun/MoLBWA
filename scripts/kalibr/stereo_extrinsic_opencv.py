#!/usr/bin/env python3
"""Kalibr 스테레오가 발산할 때: 카메라별 Kalibr **단독** 내부값을 고정하고 두 카메라 사이 위치만 OpenCV 로 구한다.

    python3 scripts/kalibr/stereo_extrinsic_opencv.py calibration/bags/<이름> \
        calibration/ov9281/mono_left_camchain.yaml calibration/ov9281/mono_right_camchain.yaml \
        calibration/target.yaml calibration/ov9281/camchain.yaml [--baseline_cm 16.4]

2026-10-26 OV9281: 단독 캘리브레이션은 둘 다 재투영 0.25 px 로 수렴했는데 스테레오는 초기 baseline 을
23 cm(실측 16.4)로 잡고 발산했다. 같은 bag 에서 두 카메라가 동시에 본 AprilGrid 모서리로 stereoCalibrate
(CALIB_FIX_INTRINSIC). 판의 태그 번호 배치(아래→위 / 위→아래)는 둘 다 풀어 보고 오차가 작은 쪽을 쓴다.
출력은 Kalibr camchain 형식(cam1.T_cn_cnm1) → camchain_to_opencv_rectify.py 로 그대로 이어진다.
"""
import argparse

import cv2
import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

P = cv2.aruco.DetectorParameters()
P.perspectiveRemovePixelPerCell, P.perspectiveRemoveIgnoredMarginPerCell, P.errorCorrectionRate = 8, 0.25, 1.0
P.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), P)


def cam(path):
    c = yaml.safe_load(open(path))["cam0"]
    fx, fy, cx, cy = c["intrinsics"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), np.array(list(c["distortion_coeffs"]) + [0.0]), c


def obj_points(tg, flip):
    """태그 id → 네 모서리 3D(m). OpenCV 모서리 순서: 좌상·우상·우하·좌하.
    배치는 2026-10-26 실측으로 정했다: 사진마다 PnP 로 후보 64가지(열·행 뒤집기·전치·모서리 회전·거울)를 풀어
    오차가 가장 작은 쪽(중앙 1.6 px, 다른 후보는 수십~수백 px) = **열을 뒤집은** 행 우선 배치. 처음엔 열을 안 뒤집어
    판이 거울상이 됐고 stereoCalibrate 가 24 px·baseline 22.5 cm 로 틀렸다."""
    s, pitch, R, C = tg["tagSize"], tg["tagSize"] * (1 + tg["tagSpacing"]), tg["tagRows"], tg["tagCols"]
    out = {}
    for i in range(R * C):
        r, c = divmod(i, C)
        x0, y0 = (C - 1 - c) * pitch, ((R - 1 - r) if flip else r) * pitch
        out[i] = np.array([[x0, y0, 0], [x0 + s, y0, 0], [x0 + s, y0 + s, 0], [x0, y0 + s, 0]], np.float32)
    return out


def detections(bag, hz, min_tags):
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    fr, last = {}, {}
    while rd.has_next():
        topic, data, _ = rd.read_next()
        side = topic.rsplit("/", 1)[-1]
        m = deserialize_message(data, Image)
        st = round(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, 6)
        if st - last.get(side, -1e9) < 1.0 / hz - 1e-3 and st not in fr:
            continue
        last[side] = st
        g = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
        corners, ids, _ = DET.detectMarkers(g)
        fr.setdefault(st, {})[side] = {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    return [(v["left"], v["right"]) for v in fr.values()
            if len(v) == 2 and len(set(v["left"]) & set(v["right"])) >= min_tags]


def solve(pairs, objs, K0, D0, K1, D1):
    O, I0, I1 = [], [], []
    for L, R in pairs:
        ids = sorted(set(L) & set(R) & set(objs))
        O.append(np.concatenate([objs[i] for i in ids]))
        I0.append(np.concatenate([L[i] for i in ids]).astype(np.float32))
        I1.append(np.concatenate([R[i] for i in ids]).astype(np.float32))
    rms, _, _, _, _, Rm, T, _, _ = cv2.stereoCalibrate(
        O, I0, I1, K0, D0, K1, D1, (640, 480), flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9))
    return rms, Rm, T.reshape(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag"); ap.add_argument("left"); ap.add_argument("right"); ap.add_argument("target"); ap.add_argument("out")
    ap.add_argument("--hz", type=float, default=4.0)
    ap.add_argument("--min_tags", type=int, default=8)
    ap.add_argument("--baseline_cm", type=float, default=None, help="자로 잰 렌즈 간격 — 결과와 비교만 한다")
    a = ap.parse_args()
    K0, D0, c0 = cam(a.left)
    K1, D1, c1 = cam(a.right)
    tg = yaml.safe_load(open(a.target))
    pairs = detections(a.bag, a.hz, a.min_tags)
    print(f"동시에 태그 ≥{a.min_tags}개 본 쌍: {len(pairs)}")
    if len(pairs) < 15:
        raise SystemExit("쌍이 너무 적다 — 두 카메라에 함께 보이게 다시 녹화")
    res = {flip: solve(pairs, obj_points(tg, flip), K0, D0, K1, D1) for flip in (False, True)}
    for flip, (rms, Rm, T) in res.items():
        print(f"  태그 배치 {'아래→위' if flip else '위→아래'}: 재투영 RMS {rms:.3f} px · baseline {np.linalg.norm(T) * 100:.2f} cm")
    flip = min(res, key=lambda f: res[f][0])
    rms, Rm, T = res[flip]
    ang = np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
    print(f"채택: RMS {rms:.3f} px · T {np.round(T * 100, 2)} cm · |T| {np.linalg.norm(T) * 100:.2f} cm · 회전 {ang:.2f}°")
    if a.baseline_cm:
        print(f"자로 잰 값 {a.baseline_cm:.1f} cm 와 차이 {np.linalg.norm(T) * 100 - a.baseline_cm:+.2f} cm")
    Tm = np.eye(4); Tm[:3, :3], Tm[:3, 3] = Rm, T
    c1 = dict(c1); c1["T_cn_cnm1"] = Tm.tolist(); c1["cam_overlaps"] = [0]
    c0 = dict(c0); c0["cam_overlaps"] = [1]
    yaml.safe_dump({"cam0": c0, "cam1": c1}, open(a.out, "w"), default_flow_style=None, sort_keys=False)
    print("저장", a.out)


if __name__ == "__main__":
    main()
