#!/usr/bin/env bash
# 가짜 Kalibr camchain(fx 300, baseline 6 cm) → camchain_to_opencv_rectify → opencv_to_orbslam3_stereo
# → uvc_stereo_bridge 정렬 경로까지. 저장소 config 는 건드리지 않고 /tmp/uvc_calibtest 에서만.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$HERE/../../../.."
T=/tmp/uvc_calibtest; rm -rf $T; mkdir -p $T
cat > $T/camchain.yaml <<'EOF'
cam0:
  camera_model: pinhole
  distortion_model: radtan
  distortion_coeffs: [-0.28, 0.08, 0.0005, -0.0003]
  intrinsics: [302.1, 301.8, 322.4, 238.9]
  resolution: [640, 480]
  rostopic: /camera/left
cam1:
  T_cn_cnm1:
  - [0.99998, 0.0021, -0.0057, -0.0601]
  - [-0.0020, 0.99999, 0.0012, 0.0004]
  - [0.0057, -0.0012, 0.99998, 0.0011]
  - [0.0, 0.0, 0.0, 1.0]
  camera_model: pinhole
  distortion_model: radtan
  distortion_coeffs: [-0.27, 0.075, -0.0002, 0.0006]
  intrinsics: [299.7, 299.9, 318.2, 241.3]
  resolution: [640, 480]
  rostopic: /camera/right
EOF
/usr/bin/python3 "$REPO/scripts/kalibr/camchain_to_opencv_rectify.py" $T/camchain.yaml $T/left_opencv.yaml $T/right_opencv.yaml
/usr/bin/python3 "$REPO/scripts/kalibr/opencv_to_orbslam3_stereo.py" $T $T/orbslam3_stereo_ov9281.yaml
source /opt/ros/jazzy/setup.bash; source "$REPO/ros2_ws/install/setup.bash"
/usr/bin/python3 "$HERE/fake_relay.py" --port 55198 > $T/relay.log 2>&1 &
RP=$!; sleep 1
ros2 run uvc_stereo uvc_stereo_bridge --ros-args -p host:=127.0.0.1 -p port:=55198 -p prefix:=/uvctest2/camera \
  -p rectify:=true -p calib_dir:=$T -p publish_raw:=true > $T/bridge.log 2>&1 &
BP=$!; sleep 2
set +e
/usr/bin/python3 "$HERE/check_topics.py" /uvctest2/camera 6; RC=$?
kill $BP $RP 2>/dev/null; wait 2>/dev/null
grep -E "정렬|캘리브레이션|Error|Traceback" $T/bridge.log | head -5
exit $RC
