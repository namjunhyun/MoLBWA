#!/usr/bin/env bash
# 카메라 없이 전체 경로 시험: 가짜 중계(포트 55199) → uvc_stereo_bridge → 토픽 8초 수신 검사.
#   bash ros2_ws/src/uvc_stereo/test/run_fake_e2e.sh
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE/../../.."
source /opt/ros/jazzy/setup.bash
(cd "$WS" && colcon build --packages-select uvc_stereo 2>&1 | tail -2)
source "$WS/install/setup.bash"
/usr/bin/python3 "$HERE/fake_relay.py" --port 55199 > /tmp/uvc_fake_relay.log 2>&1 &
RP=$!
sleep 1
ros2 run uvc_stereo uvc_stereo_bridge --ros-args -p host:=127.0.0.1 -p port:=55199 -p prefix:=/uvctest/camera \
  -p rectify:=false > /tmp/uvc_bridge.log 2>&1 &
BP=$!
sleep 2
set +e
/usr/bin/python3 "$HERE/check_topics.py" /uvctest/camera 8; RC=$?
kill $BP $RP 2>/dev/null; wait 2>/dev/null
echo "--- 중계 로그"; tail -3 /tmp/uvc_fake_relay.log
echo "--- 브리지 로그"; tail -3 /tmp/uvc_bridge.log
exit $RC
