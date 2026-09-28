#!/bin/bash
# 팔을 저장된 휴식 자세로 (conda 환경 자동 회피). ★ 팔이 움직임.
cd ~/MoLBWA-gaze-hri
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
exec python3 -u tools/so101_goto.py rest
