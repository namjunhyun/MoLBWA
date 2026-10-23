#!/bin/bash
# 데모2 라이브러리 v7 (2026-10-14): v6(허리 고정 손 흔들기·악수) + '같이 보기'용 point(가리키기)·celebrate(박수, 2026-10-14 변경)
#   두 클립도 상체만·허리 고정(--upper_only --freeze_waist). 구간·속도는 후보 영상 보고 정한다.
#   사용: POINT_NPZ=... POINT_T0=.. POINT_T1=.. CHEER_NPZ=... CHEER_T0=.. CHEER_T1=.. [CHEER_SPEED=0.7] \
#         ART=~/molbwa_g1 TURNS=$ART/loco_turns4 bash g1/tools/build_library_v7.sh
set -e
: "${POINT_NPZ:?}" "${POINT_T0:?}" "${POINT_T1:?}" "${CHEER_NPZ:?}" "${CHEER_T0:?}" "${CHEER_T1:?}"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ART=${ART:-$HOME/molbwa_g1}
L=${OUT:-$ART/lib7}
OUT=$L bash "$REPO/g1/tools/build_library_v6.sh" > /dev/null    # 대기·회전·손 흔들기·악수는 v6 그대로
C="$REPO/g1/motion/amass_g1_to_csv.py"
python3 -I "$C" "$POINT_NPZ" "$L/clips/point.csv" --t0 "$POINT_T0" --t1 "$POINT_T1" --speed "${POINT_SPEED:-1.0}" ${POINT_EXTRA:-} \
  --upper_only --freeze_waist --pad_start 0 --pad_end 0.5
python3 -I "$C" "$CHEER_NPZ" "$L/clips/celebrate.csv" --t0 "$CHEER_T0" --t1 "$CHEER_T1" --speed "${CHEER_SPEED:-1.0}" ${CHEER_EXTRA:-} \
  --upper_only --freeze_waist --pad_start 0 --pad_end 0.5
cat >> "$L/segments.yaml" <<'EOF'
  - {name: point, csv: clips/point.csv}
  - {name: celebrate, csv: clips/celebrate.csv}
EOF
python3 -I "$REPO/g1/motion/build_motion_library.py" --segments "$L/segments.yaml" --out_csv "$L/molbwa_library7.csv" \
  --out_meta "$L/library_meta.json" --seam_joint 0.2 | grep -A5 '"seam"'
ls "$L"
