#!/usr/bin/env bash
# ROS 2 bag 의 토픽별 메시지 수: scripts/check_bag_counts.sh calibration/bags/<이름>
source /opt/ros/jazzy/setup.bash
ros2 bag info "$1" | grep -E "Duration|Topic:"
