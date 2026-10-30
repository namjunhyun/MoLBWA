#!/bin/bash
cd ~/MoLBWA-gaze-hri
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch gaze_hri calibrate_world_to_base.launch.py source:=udp backend:=feetech 2>&1 | tee /tmp/calib_launch.log
