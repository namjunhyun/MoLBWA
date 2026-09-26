#!/bin/bash
# 배치 B 시연 스택 한 번에 띄우기: 태그 브리지 + gaze_hri(탑다운 컵 검출 + 스냅 + 팔).
#
#   scripts/run_b_demo.sh --cam 0              # 팔 안 움직임 (dry_run, 기본)
#   scripts/run_b_demo.sh --cam 0 --slam       # 태그 안 보일 때 SLAM 으로 이어 가기
#   scripts/run_b_demo.sh --cam 0 --real       # ★ 실제 팔 (확인 문구 입력 필요)
#
# 시선 뷰어는 캘리브(f -> m -> 9~12점)를 사람이 해야 해서 따로 띄운다:
#   cd src && python3 -u gaze_on_scene.py --no-rerun --source ros --flip --send-gaze-px
#
# Ctrl+C 로 끝내면 띄운 프로세스를 전부 정리한다 (pkill 패턴 대신 PID/프로세스 그룹으로 —
# 패턴은 이 스크립트를 부른 셸 자신과 매칭돼 죽는 일이 있었다).
set -u
CAM=0; SLAM=""; BACKEND=dry_run
while [ $# -gt 0 ]; do
  case "$1" in
    --cam) CAM="$2"; shift 2 ;;
    --slam) SLAM="--slam"; shift ;;
    --real) BACKEND=feetech; shift ;;
    *) echo "모르는 인자: $1"; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${TMPDIR:-/tmp}/molbwa_b_demo_$(date +%H%M%S)"
mkdir -p "$LOG"

# conda 가 ROS2 를 깨뜨린다 (HANDOFF_2026-09-23 §7)
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
set +u                                   # ROS setup.bash 는 미정의 변수를 쓴다
source /opt/ros/jazzy/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
set -u

if [ "$BACKEND" = feetech ]; then
  echo "★ 실제 팔로워를 움직입니다. 작업 영역을 비우고, 12V 전원을 바로 뽑을 수 있게 하세요."
  read -r -p "  계속하려면 '움직여' 를 입력: " ans
  [ "$ans" = "움직여" ] || { echo "취소"; exit 1; }
fi

if ! timeout 4 ros2 topic list 2>/dev/null | grep -q '^/camera/left/compressed$'; then
  echo "[경고] 씬 카메라 토픽이 없다 — Pi 에서 ~/ocams.sh 를 띄웠는지 확인 (태그 브리지가 대기만 한다)"
fi
HF="$HOME/.ros/topdown_homography.yaml"
if [ ! -f "$HF" ]; then
  echo "[경고] 탑다운 호모그래피 없음 — tools/topdown_calib.py 먼저"
elif ! python3 -c "import yaml,sys; d=yaml.safe_load(open('$HF')); sys.exit(0 if (d.get('num_points') or 0)>=4 and d.get('reproj_error_mm') is not None else 1)"; then
  echo "[경고] 탑다운 호모그래피가 실측값이 아니다 (촬영용 임시값?) — topdown_click 이 거부한다. tools/topdown_calib.py 로 캘리브할 것"
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

setsid python3 -u "$ROOT/arm/gaze_tag_bridge.py" $SLAM > "$LOG/bridge.log" 2>&1 &
PIDS+=($!)
setsid ros2 launch gaze_hri gaze_hri.launch.py source:=udp gaze_frame:=base_link \
  use_topdown:=true camera_index:="$CAM" backend:="$BACKEND" > "$LOG/gaze_hri.log" 2>&1 &
PIDS+=($!)

echo "띄움: 브리지(${SLAM:-태그 전용}) + gaze_hri(backend=$BACKEND, 탑다운 카메라 $CAM). 로그 $LOG"
echo "Ctrl+C 로 종료. 2초마다 요약:"
while true; do
  sleep 2
  b=$(grep -a "유효 송신" "$LOG/bridge.log" | tail -1 | sed 's/.*INFO //')
  g=$(grep -aE "목표 확정|완료:|실패:|거부|안전 정지|모호" "$LOG/gaze_hri.log" | tail -1 | sed 's/.*\]: //')
  echo "  [브리지] ${b:-대기} | [gaze_hri] ${g:--}"
done
