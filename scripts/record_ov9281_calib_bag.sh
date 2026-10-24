#!/usr/bin/env bash
# OV9281 스테레오 캘리브레이션용 녹화 (IMU 없음 → 카메라 두 대만).
# 먼저 다른 터미널에서 원본 모드로 브리지를 띄운다:   UNO_Q=<IP> RAW=1 ./run_uvc_stereo.sh
#   사용: scripts/record_ov9281_calib_bag.sh [이름]     → calibration/bags/<이름>/  (Ctrl-C 로 끝)
# 그다음: scripts/convert_bag_to_ros1.sh calibration/bags/<이름>   →   FOCAL_GUESS=300 scripts/kalibr/calibrate_cameras.sh calibration/bags/<이름>.bag
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
TOPICS=(/camera/left /camera/right)
OUT="calibration/bags/${1:-ov9281_stereo_$(date +%Y%m%d_%H%M%S)}"
mkdir -p calibration/bags
AV="$(ros2 topic list)"
for t in "${TOPICS[@]}"; do
  grep -qx "$t" <<<"$AV" || { echo "토픽 없음: $t — 먼저 'UNO_Q=<IP> RAW=1 ./run_uvc_stereo.sh'"; exit 1; }
done
echo "== 녹화 → $OUT  (Ctrl-C 로 끝)"
echo "   타깃(에이프릴그리드)을 두 카메라 모두에 보이게, 천천히, 1~2분: 가까이/멀리, 화면 네 귀퉁이, 위아래·좌우로 30~45° 기울여서"
echo "   빛 반사(광택) 없게, 흐림 없게 — 빠르게 움직이면 두 카메라가 동기 안 돼서 캘리브레이션이 틀어진다"
exec ros2 bag record -s sqlite3 -o "$OUT" "${TOPICS[@]}"
