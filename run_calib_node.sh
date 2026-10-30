#!/bin/bash
cd ~/MoLBWA-gaze-hri
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
exec ros2 run gaze_hri calibrate_world_to_base --ros-args --params-file ros2_ws/src/gaze_hri/config/gaze_hri.yaml
