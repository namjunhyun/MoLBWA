#!/bin/bash
cd ~/MoLBWA-gaze-hri
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
exec python3 -u arm/gaze_tag_bridge.py
