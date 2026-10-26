#!/bin/bash
cd ~/MoLBWA-gaze-hri/src
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
# 2026-10-26: y 반전은 확정값이라 기본으로 (y 키는 캘리브 점을 지운다). 새 캘리브는 f 한 번 -> goto_sequence --arm a --gui
exec python3 -u gaze_on_scene.py --no-rerun --source ros --flip --send-gaze-px --mirror-y "$@" \
  --left-topic "${CAM_PREFIX-/pc}/camera/left/compressed" --right-topic "${CAM_PREFIX-/pc}/camera/right/compressed"
