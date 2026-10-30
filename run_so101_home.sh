#!/bin/bash
# 팔 서버 없이(포트 직접) 팔을 home 자세로. ★ 팔이 움직임. 데모/캘리브 서버가 켜져 있으면 쓰지 말 것.
cd ~/MoLBWA-gaze-hri
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
exec python3 -u tools/so101_goto.py home
