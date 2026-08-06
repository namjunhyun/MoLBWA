#!/usr/bin/env bash
# Convert a ROS2 bag (recorded by scripts/record_imu_cam_bag.sh) into a ROS1
# .bag file, since Kalibr's kalibr_calibrate_imu_camera is ROS1-only.
#
# Uses the `rosbags` pure-Python library (no ROS install needed) via its
# `rosbags-convert` CLI. Self-contained: creates/reuses a small venv at
# calibration/.convert_venv so it doesn't touch whatever Python env you use
# for gaze_on_scene.py / orbslam3_rerun.py.
#
# Usage:
#   scripts/convert_bag_to_ros1.sh <ros2_bag_dir> [output.bag]
#
# Example:
#   scripts/convert_bag_to_ros1.sh calibration/bags/imu_cam_calib_20260806_190000

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ros2_bag_dir> [output.bag]"
  exit 1
fi

SRC="$1"
if [[ ! -d "$SRC" ]]; then
  echo "Not a directory: $SRC"
  exit 1
fi

DST="${2:-${SRC%/}.bag}"

VENV_DIR="calibration/.convert_venv"
if [[ ! -x "${VENV_DIR}/bin/rosbags-convert" ]]; then
  echo "== Setting up isolated venv for rosbags at ${VENV_DIR} =="
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install --quiet --upgrade pip
  "${VENV_DIR}/bin/pip" install --quiet rosbags
fi

if [[ -e "$DST" ]]; then
  echo "Output already exists, refusing to overwrite: $DST"
  echo "Remove it or pass a different output path."
  exit 1
fi

echo "Converting ${SRC} -> ${DST}"
"${VENV_DIR}/bin/rosbags-convert" --src "${SRC}" --dst "${DST}"

echo
echo "Done: ${DST}"
echo "Next: run kalibr_calibrate_imu_camera (ROS1 Kalibr docker) against"
echo "  --bag ${DST} --target calibration/target.yaml --cam <camchain.yaml> --imu <imu.yaml>"
