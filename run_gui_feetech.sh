#!/bin/bash
# 마우스(탑다운 화면 클릭)로 집기/놓기 — 실제 팔. 사용: ./run_gui_feetech.sh a  (또는 b)
# 2026-10-12: 팔로워 2대. a = …5667 (기존 팔), b = …5906.
ARM=${1:-a}
cd ~/MoLBWA-gaze-hri
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
exec ros2 launch gaze_hri gui.launch.py input:=mouse backend:=feetech camera_index:=0 arm:=$ARM
