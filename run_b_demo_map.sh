#!/bin/bash
# run_b_demo.sh 와 같은 조합이지만 gaze_frame:=map 으로 띄워서
# T_BW 캘리브(~/.ros/gaze_hri_calib.yaml) 보정을 적용한다.
set -u
CAM=0; BACKEND=dry_run
while [ $# -gt 0 ]; do
  case "$1" in
    --cam) CAM="$2"; shift 2 ;;
    --real) BACKEND=feetech; shift ;;
    *) echo "모르는 인자: $1"; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="${TMPDIR:-/tmp}/molbwa_b_demo_map_$(date +%H%M%S)"
mkdir -p "$LOG"

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
set +u
source /opt/ros/jazzy/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
set -u

if [ "$BACKEND" = feetech ]; then
  echo "★ 실제 팔로워를 움직입니다."
  read -r -p "  계속하려면 '움직여' 를 입력: " ans
  [ "$ans" = "움직여" ] || { echo "취소"; exit 1; }
fi

PIDS=()
cleanup() {
  echo; echo "정리 중..."
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null || kill -INT "$p" 2>/dev/null; done
  sleep 3
  for p in "${PIDS[@]}"; do kill -9 -- "-$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done
  echo "로그: $LOG"
  exit 0
}
trap cleanup INT TERM

setsid python3 -u "$ROOT/arm/gaze_tag_bridge.py" > "$LOG/bridge.log" 2>&1 &
PIDS+=($!)
setsid ros2 launch gaze_hri gaze_hri.launch.py source:=udp \
  use_topdown:=true camera_index:="$CAM" backend:="$BACKEND" > "$LOG/gaze_hri.log" 2>&1 &
PIDS+=($!)

echo "띄움: 브리지(태그 전용) + gaze_hri(gaze_frame=map, backend=$BACKEND). 로그 $LOG"
echo "Ctrl+C 로 종료. 2초마다 요약:"
while true; do
  sleep 2
  b=$(grep -a "유효 송신" "$LOG/bridge.log" | tail -1 | sed 's/.*INFO //')
  g=$(grep -aE "목표 확정|완료:|실패:|거부|안전 정지|모호" "$LOG/gaze_hri.log" | tail -1 | sed 's/.*\]: //')
  echo "  [브리지] ${b:-대기} | [gaze_hri] ${g:--}"
done
