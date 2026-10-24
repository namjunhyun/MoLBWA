#!/bin/bash
# ORB-SLAM3 **스테레오 전용**(OV9281 두 대엔 IMU 가 없다). 입력은 run_uvc_stereo.sh 가 낸 정렬된 /pc/camera/*/compressed.
# 설정은 캘리브레이션 뒤 scripts/kalibr/opencv_to_orbslam3_stereo.py 로 만든 ros2_ws/src/uvc_stereo/config/orbslam3_stereo_ov9281.yaml.
# 출력: /orbslam3/pose (스테레오 전용이라 축척은 baseline 에서 온다 — IMU 초기화 없음)
cd "$(dirname "$0")"
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
export LD_LIBRARY_PATH=$HOME/ORB_SLAM3/lib:$HOME/ORB_SLAM3/Thirdparty/DBoW2/lib:$HOME/ORB_SLAM3/Thirdparty/g2o/lib:/usr/local/lib:$LD_LIBRARY_PATH
CFG=ros2_ws/src/uvc_stereo/config/orbslam3_stereo_ov9281.yaml
[ -f "$CFG" ] || { echo "$CFG 없음 — 캘리브레이션 먼저 (docs/14_ov9281_stereo_uno_q.md 5단계)"; exit 1; }
for side in left right; do
  ros2 run image_transport republish --ros-args -p in_transport:=compressed -p out_transport:=raw \
    -r in/compressed:=${CAM_PREFIX-/pc}/camera/$side/compressed -r out:=/slam/$side &
done
trap 'kill -- -$$ 2>/dev/null' EXIT
ros2 run orbslam3 stereo "$HOME/ORB_SLAM3/Vocabulary/ORBvoc.txt" "$CFG" false \
  --ros-args -r camera/left:=/slam/left -r camera/right:=/slam/right
