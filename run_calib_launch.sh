#!/bin/bash
cd ~/MoLBWA-gaze-hri
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
exec ros2 launch gaze_hri calibrate_world_to_base.launch.py source:=udp backend:=feetech
