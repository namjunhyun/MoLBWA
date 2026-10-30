#!/bin/bash
# ORB-SLAM3 스테레오-이너셜 (하이브리드 시험용). Pi 무압축 영상은 WiFi 로 못 버텨서(~146Mbps)
# PC 에서 중계(/pc/camera/*/compressed, ./run_cam_relay.sh)를 풀어 /slam/left,right 로 넣는다. IMU(/imu)는 그대로.
# 출력: /orbslam3/pose  -> gaze_tag_bridge.py --slam 이 받는다.
cd ~/MoLBWA-gaze-hri
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
export LD_LIBRARY_PATH=$HOME/ORB_SLAM3/lib:$HOME/ORB_SLAM3/Thirdparty/DBoW2/lib:$HOME/ORB_SLAM3/Thirdparty/g2o/lib:/usr/local/lib:$LD_LIBRARY_PATH
PIDS=()
for side in left right; do
  ros2 run image_transport republish --ros-args -p in_transport:=compressed -p out_transport:=raw \
    -r in/compressed:=${CAM_PREFIX-/pc}/camera/$side/compressed -r out:=/slam/$side &
  PIDS+=($!)
done
trap 'kill -- -$$ 2>/dev/null' EXIT   # ros2 run 이 띄운 republish 자식까지 같이 (setsid 로 띄울 것)
ros2 run orbslam3 stereo-inertial "$HOME/ORB_SLAM3/Vocabulary/ORBvoc.txt" \
  ros2_ws/src/orbslam3_ros2/config/stereo-inertial/oCamS.yaml false \
  --ros-args -r camera/left:=/slam/left -r camera/right:=/slam/right -r imu:=/imu
