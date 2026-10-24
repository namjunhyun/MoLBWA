#!/bin/bash
# PC: UNO Q 중계(edge/uno_q/stereo_relay.py) → 정렬된 스테레오 /pc/camera/{left,right}/compressed.
# 기존 oCamS 경로의 'Pi ~/ocams.sh + run_cam_relay.sh' 를 이것 하나로 바꾼다. 소비 코드는 그대로.
#   UNO_Q=192.168.50.xx ./run_uvc_stereo.sh            (평소: 정렬 켬)
#   UNO_Q=... RAW=1 ./run_uvc_stereo.sh                 (캘리브레이션 녹화용: 원본을 /camera/left·right 로)
cd "$(dirname "$0")"
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
: "${UNO_Q:?UNO_Q=<UNO Q IP> 를 주세요}"
if [ -n "$RAW" ]; then
  exec ros2 run uvc_stereo uvc_stereo_bridge --ros-args -p host:=$UNO_Q -p rectify:=false -p prefix:=/camera -p publish_raw:=true
fi
exec ros2 run uvc_stereo uvc_stereo_bridge --ros-args -p host:=$UNO_Q -p prefix:=${CAM_PREFIX-/pc}/camera
