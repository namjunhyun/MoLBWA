#!/bin/bash
# arm_server(데모/캘리브 launch)가 떠 있을 때 팔을 home 관절각으로 보낸다. ★ 팔이 움직임.
# home = gaze_hri.yaml home_xyz [0.25,0,0.12] pitch -1.2 의 IK 결과 (arm_server 시작 로그와 같음).
cd ~/MoLBWA-gaze-hri
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
# arm_server 가 없으면 -w 1 이 구독자를 무한정 기다리다가, 나중에 데모를 띄우는 순간 팔을 움직인다.
# 그래서 5초 안에 못 보내면 포기한다.
if ! timeout 5 ros2 topic pub --once -w 1 /arm/goto_joints std_msgs/msg/Float64MultiArray \
     "{data: [0.0, 1.257, -1.216, -1.241, 0.0, 1.2]}"; then
  echo "실패: arm_server 가 없다 — 데모(run_b_demo.sh --real)를 먼저 띄울 것"; exit 1
fi
