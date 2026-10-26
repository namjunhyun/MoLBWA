#!/bin/bash
cd ~/MoLBWA-gaze-hri
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
# 2026-10-26: 중계(/pc) 영상 + 인자 전달. 광선이 책상과 안 만나도 방향은 보냄(--ray-fallback-m 1.5 기본)
exec python3 -u arm/gaze_tag_bridge.py --scene-topic "${CAM_PREFIX-/pc}/camera/left/compressed" "$@"
