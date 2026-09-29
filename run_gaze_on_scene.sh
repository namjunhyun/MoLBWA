#!/bin/bash
cd ~/MoLBWA-gaze-hri/src
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
exec python3 -u gaze_on_scene.py --no-rerun --source ros --flip --send-gaze-px
