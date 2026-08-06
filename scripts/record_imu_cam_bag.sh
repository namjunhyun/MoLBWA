#!/usr/bin/env bash
# Record a ROS2 bag of oCamS stereo images + IMU for Kalibr camera-IMU
# extrinsic calibration (kalibr_calibrate_imu_camera).
#
# Prereq: ocams_stereo_imu_node already running, e.g. in another terminal:
#   ros2 run ocams_ros2 ocams_stereo_imu_node
#
# Usage:
#   scripts/record_imu_cam_bag.sh [output_name]
#
# Output: a rosbag2 directory (sqlite3/mcap) under calibration/bags/<output_name>.
# Convert it to a ROS1 .bag afterwards with the `rosbags` tool before feeding
# it to the ROS1-based Kalibr docker image.

set -euo pipefail

TOPICS=(/camera/left /camera/right /imu)
OUT_NAME="${1:-imu_cam_calib_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="calibration/bags/${OUT_NAME}"

mkdir -p calibration/bags

echo "== Checking topics are alive =="
AVAILABLE="$(ros2 topic list)"
MISSING=0
for t in "${TOPICS[@]}"; do
  if ! grep -qx "$t" <<<"$AVAILABLE"; then
    echo "  MISSING: $t"
    MISSING=1
  else
    echo "  OK: $t"
  fi
done

if [[ "$MISSING" -eq 1 ]]; then
  echo
  echo "One or more topics are not being published. Start the driver first:"
  echo "  ros2 run ocams_ros2 ocams_stereo_imu_node"
  exit 1
fi

cat <<'EOF'

== Recording checklist (AprilGrid, calibration/apriltag_grid.pdf) ==
- Keep the grid fully visible to BOTH left and right cameras as much as possible.
- Move slowly and smoothly (fast motion blurs the tags and starves the IMU excitation).
- Cover all 6 excitation axes: translate along x/y/z, rotate about roll/pitch/yaw.
- Vary distance and tilt (near/far, angled views) too, not just translation.
- Aim for ~2-3 minutes total. Start and end with the rig held still for ~2s
  (helps Kalibr's initial bias estimate).

Press Enter to start recording, Ctrl+C to stop.
EOF
read -r

echo "Recording to: ${OUT_DIR}"
ros2 bag record -o "${OUT_DIR}" "${TOPICS[@]}"

echo
echo "Done. Next: convert to ROS1 bag with the 'rosbags' library, then run"
echo "kalibr_calibrate_imu_camera against calibration/target.yaml."
