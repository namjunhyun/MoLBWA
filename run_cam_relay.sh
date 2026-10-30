#!/bin/bash
# Pi 씬 영상 중계(WiFi 1회 수신 -> /pc/camera/{left,right}/compressed). 다른 실행 스크립트보다 먼저 띄운다.
cd ~/MoLBWA-gaze-hri
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
exec python3 -u tools/cam_relay.py "$@"
