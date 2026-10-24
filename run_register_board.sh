#!/bin/bash
# 클립보드 태그 등록 (2026-10-22). 안경 쓰고 책상 태그 + 클립보드가 같이 보이게 머리를 여러 각도로. q = 계산
cd ~/MoLBWA-gaze-hri
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
exec python3 -u arm/register_board_tags.py "$@"
