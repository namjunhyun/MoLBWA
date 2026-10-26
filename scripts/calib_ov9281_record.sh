#!/usr/bin/env bash
# 카메라를 PC 에 직결했을 때 캘리브레이션 녹화를 한 번에: 중계 + 브리지(원본) + 미리보기 창 + 녹화.
#   scripts/calib_ov9281_record.sh [초=120] [이름]
#   왼·오 장치는 LEFT/RIGHT 로 바꿀 수 있다(기본: 포트 2.2 = 착용자 왼쪽, 2.4 = 오른쪽 — 2026-10-26 렌즈 가리기로 확인).
#   판이 화면의 1/3 이상을 채우게(태그 한 변 ≥ 20 px). 멀면 태그를 못 읽는다(1.5 m 에서 8 px → 0개).
# 끝나면 → scripts/convert_bag_to_ros1.sh calibration/bags/<이름>  →  FOCAL_GUESS=300 scripts/kalibr/calibrate_cameras.sh calibration/bags/<이름>.bag
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
SEC="${1:-120}"; NAME="${2:-ov9281_$(date +%Y%m%d_%H%M%S)}"
export LEFT="${LEFT:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.2:1.0-video-index0}"
export RIGHT="${RIGHT:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.4:1.0-video-index0}"
export CONTROLS="${CONTROLS:-auto_exposure=1,exposure_time_absolute=80}"   # 수동·짧은 노출(8 ms) — 흔들림 흐림 방지
source /opt/ros/jazzy/setup.bash; source ros2_ws/install/setup.bash
bash ros2_ws/src/uvc_stereo/test/run_local_cams.sh keep & CAM=$!
trap 'kill $PV $CAM 2>/dev/null; pkill -P $CAM 2>/dev/null; true' EXIT
for i in $(seq 1 20); do ros2 topic list | grep -qx /camera/left && break; sleep 0.5; done
/usr/bin/python3 scripts/ov9281_calib_preview.py & PV=$!
sleep 2
echo "== ${SEC}초 녹화 → calibration/bags/${NAME}  (판을 천천히: 가까이·멀리·네 귀퉁이·30~45° 기울여서)"
mkdir -p calibration/bags
timeout -s INT "$SEC" ros2 bag record -s sqlite3 -o "calibration/bags/${NAME}" /camera/left /camera/right || true
echo "== 녹화 끝: calibration/bags/${NAME}"
