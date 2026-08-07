#!/usr/bin/env bash
# Run kalibr_calibrate_imu_camera non-interactively in the kalibr docker
# image, to produce the camera-IMU extrinsic (T_cam_imu) — the actual goal
# of this whole pipeline: replacing the identity placeholder currently in
# orbslam3_ros2/config/stereo-inertial/oCamS.yaml's IMU.T_b_c1.
#
# Usage:
#   scripts/kalibr/calibrate_imu_camera.sh <ros1_bag_path> [data_dir] [camchain_file]
#
# <ros1_bag_path> : the same bag used for calibrate_cameras.sh.
# data_dir        : mounted into the container as /data. Defaults to
#                   ./calibration. Must contain target.yaml, imu.yaml, and
#                   camchain_file.
# camchain_file   : camchain filename inside data_dir. Defaults to
#                   camchain.yaml (full stereo chain from calibrate_cameras.sh).
#                   Pass camchain_mono.yaml to calibrate cam0-IMU only,
#                   sidestepping cam1-chain numerical issues since the actual
#                   project need (IMU.T_b_c1) is cam0(left)-IMU only anyway.
#
# Output: a `<bag>-camchain-imucam.yaml` (contains T_cam_imu) + report
# pdf/txt written into data_dir, next to the bag.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ros1_bag_path> [data_dir] [camchain_file]"
  exit 1
fi

cd "$(git rev-parse --show-toplevel)"

IMAGE_TAG="kalibr"
BAG_HOST_PATH="$(realpath "$1")"
DATA_DIR="$(realpath "${2:-calibration}")"
CAMCHAIN_FILE="${3:-camchain.yaml}"

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Docker image '${IMAGE_TAG}' not found. Build it first:"
  echo "  scripts/kalibr/build_kalibr_docker.sh"
  exit 1
fi

if [[ "$BAG_HOST_PATH" != "$DATA_DIR"/* ]]; then
  echo "Bag must live under data_dir (${DATA_DIR}) so it's visible at /data in the container."
  exit 1
fi

for f in target.yaml "$CAMCHAIN_FILE" imu.yaml; do
  if [[ ! -f "${DATA_DIR}/${f}" ]]; then
    echo "Missing ${DATA_DIR}/${f}."
    exit 1
  fi
done

BAG_IN_CONTAINER="/data/${BAG_HOST_PATH#${DATA_DIR}/}"

echo "== Running kalibr_calibrate_imu_camera =="
echo "   bag:      ${BAG_IN_CONTAINER}"
echo "   target:   /data/target.yaml"
echo "   cams:     /data/${CAMCHAIN_FILE}"
echo "   imu:      /data/imu.yaml"
docker run --rm \
  --entrypoint bash \
  -v "${DATA_DIR}:/data" \
  "$IMAGE_TAG" \
  -lc "source /catkin_ws/devel/setup.bash && cd /data && \
    rosrun kalibr kalibr_calibrate_imu_camera \
      --bag '${BAG_IN_CONTAINER}' \
      --target /data/target.yaml \
      --cams '/data/${CAMCHAIN_FILE}' \
      --imu /data/imu.yaml \
      --dont-show-report"

echo
echo "Done. Check ${DATA_DIR}/bags for *-camchain-imucam.yaml and the report pdf/txt."
echo "ORB-SLAM3 IMU.T_b_c1 is camera-to-body: use inverse(T_cam_imu), not T_cam_imu directly."
