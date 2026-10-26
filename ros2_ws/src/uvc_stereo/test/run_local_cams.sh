#!/usr/bin/env bash
# 카메라를 PC 에 바로 꽂았을 때: 중계(로컬) → 브리지(원본, /camera/...) → 토픽 검사. UNO Q 없이 시험·캘리브레이션 녹화용.
#   LEFT=<by-path> RIGHT=<by-path> CONTROLS="auto_exposure=1,exposure_time_absolute=80" bash run_local_cams.sh [초|keep]
#   keep 이면 검사 없이 계속 띄워 둔다(캘리브레이션 녹화하는 동안). Ctrl-C 로 끝.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$HERE/../../../.."
: "${LEFT:?LEFT=/dev/v4l/by-path/...-video-index0}"; : "${RIGHT:?RIGHT=...}"
source /opt/ros/jazzy/setup.bash; source "$REPO/ros2_ws/install/setup.bash"
/usr/bin/python3 "$REPO/edge/uno_q/stereo_relay.py" --left "$LEFT" --right "$RIGHT" --fps "${FPS:-60}" \
  --controls "${CONTROLS:-}" --port 55101 > /tmp/uvc_local_relay.log 2>&1 &
RP=$!; sleep 2
ros2 run uvc_stereo uvc_stereo_bridge --ros-args -p host:=127.0.0.1 -p port:=55101 -p prefix:=/camera \
  -p rectify:=false -p publish_raw:=true > /tmp/uvc_local_bridge.log 2>&1 &
BP=$!
trap 'pkill -P $BP 2>/dev/null; kill $BP $RP 2>/dev/null' EXIT   # ros2 run 이 띄운 실제 노드(자식)까지 — 안 그러면 남는다
if [ "${1:-8}" = keep ]; then
  echo "중계·브리지 실행 중 (/camera/left·right 원본). 로그: /tmp/uvc_local_relay.log  Ctrl-C 로 끝"; wait $BP; exit
fi
sleep 3
/usr/bin/python3 "$HERE/check_topics.py" /camera "${1:-8}" || true
echo "--- 중계"; grep -vE "^\s*$" /tmp/uvc_local_relay.log | tail -4
