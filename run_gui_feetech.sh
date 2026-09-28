#!/bin/bash
cd ~/MoLBWA-gaze-hri
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
exec ros2 launch gaze_hri gui.launch.py input:=mouse backend:=feetech camera_index:=0
