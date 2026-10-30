#!/bin/bash
cd ~/MoLBWA-gaze-hri
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch gaze_hri gui.launch.py input:=mouse backend:=feetech camera_index:=0 2>&1 | tee /tmp/gui_session.log
