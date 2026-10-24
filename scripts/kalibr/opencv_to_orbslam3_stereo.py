#!/usr/bin/env python3
"""left/right_opencv.yaml(정렬 결과) → ORB-SLAM3 **스테레오 전용(IMU 없음)** 설정.

    python3 scripts/kalibr/opencv_to_orbslam3_stereo.py ros2_ws/src/uvc_stereo/config ros2_ws/src/uvc_stereo/config/orbslam3_stereo_ov9281.yaml

OV9281 두 대에는 IMU 가 없어 oCamS 의 stereo-inertial 설정을 못 쓴다. 입력 영상이 이미 정렬돼 오므로
(uvc_stereo_bridge rectify:=true) 왜곡 계수는 0, 두 카메라 내부값은 정렬 행렬 P 의 fx·cx·cy 로 같고,
Stereo.T_c1_c2 는 x 방향 baseline 만 있는 순수 평행이동이다(oCamS 설정과 같은 방식).
"""
import sys

import cv2


def load(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        sys.exit(f"못 엶: {path}")
    P = fs.getNode("projection_matrix").mat()
    w, h = int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real())
    fs.release()
    return P, w, h


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    d, out = sys.argv[1], sys.argv[2]
    P1, w, h = load(f"{d}/left_opencv.yaml")
    P2, _, _ = load(f"{d}/right_opencv.yaml")
    fx, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
    b = -P2[0, 3] / P2[0, 0]
    if abs(P2[0, 0] - fx) > 1e-6 or abs(P2[1, 2] - cy) > 1e-6:
        sys.exit("좌우 정렬 fx/cy 가 다르다 — stereoRectify 결과가 아니다")
    cam = "".join(f"Camera{i}.fx: {fx:.8f}\nCamera{i}.fy: {fx:.8f}\nCamera{i}.cx: {cx:.8f}\nCamera{i}.cy: {cy:.8f}\n"
                  f"Camera{i}.k1: 0.0\nCamera{i}.k2: 0.0\nCamera{i}.p1: 0.0\nCamera{i}.p2: 0.0\n" for i in (1, 2))
    txt = f"""%YAML:1.0
# OV9281 USB 두 대(UNO Q 중계) 스테레오 — IMU 없음. {d} 에서 자동 생성 (opencv_to_orbslam3_stereo.py)
# 입력은 uvc_stereo_bridge 가 이미 정렬해 보낸다 → 왜곡 0, do_rectify=false 로 실행.
File.version: "1.0"
Camera.type: "PinHole"
{cam}Camera.width: {w}
Camera.height: {h}
Camera.fps: 30
Camera.RGB: 0
Stereo.ThDepth: 40.0
Stereo.T_c1_c2: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [1.0, 0.0, 0.0, {b:.8f},
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
ORBextractor.nFeatures: 1200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""
    open(out, "w").write(txt)
    print(f"저장 {out}: fx {fx:.2f} cx {cx:.2f} cy {cy:.2f} baseline {b * 100:.2f} cm ({w}x{h})")


if __name__ == "__main__":
    main()
