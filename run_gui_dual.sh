#!/bin/bash
# 두 팔(a, b) 마우스 GUI — 집을 컵에 가까운 팔이 움직인다 (2026-10-14). 사용: ./run_gui_dual.sh
cd ~/MoLBWA-gaze-hri
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
exec ros2 launch gaze_hri dual.launch.py backend:=feetech camera_index:=0 input:=${INPUT:-mouse} ${RECORD_DIR:+record_dir:=$RECORD_DIR}   # INPUT=gaze 면 안경 시선, RECORD_DIR=폴더 면 화면 녹화 (빈 값을 넘기면 launch 가 거부)
