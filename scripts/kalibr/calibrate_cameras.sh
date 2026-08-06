#!/usr/bin/env bash
# Run kalibr_calibrate_cameras non-interactively in the kalibr docker image,
# to produce a fresh camchain.yaml (camera intrinsics + stereo extrinsic)
# from the AprilGrid recording — used as input to kalibr_calibrate_imu_camera.
#
# Usage:
#   scripts/kalibr/calibrate_cameras.sh <ros1_bag_path> [data_dir]
#
# <ros1_bag_path> : path to the .bag produced by scripts/convert_bag_to_ros1.sh
#                   (relative to repo root, or absolute).
# data_dir        : mounted into the container as /data. Defaults to
#                   ./calibration. Must contain target.yaml.
#
# Model: pinhole-radtan for both cameras, matching the existing OpenCV
# calibration format in ocams_ros2/config/{left,right}_opencv.yaml
# (5-coeff plumb_bob with k3=0 -> radtan's k1,k2,p1,p2).
#
# Output: camchain.yaml (+ report pdf/txt) written into data_dir, next to
# the bag, by kalibr itself (it names outputs after the bag file).

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ros1_bag_path> [data_dir]"
  exit 1
fi

cd "$(git rev-parse --show-toplevel)"

IMAGE_TAG="kalibr"
BAG_HOST_PATH="$(realpath "$1")"
DATA_DIR="$(realpath "${2:-calibration}")"

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Docker image '${IMAGE_TAG}' not found. Build it first:"
  echo "  scripts/kalibr/build_kalibr_docker.sh"
  exit 1
fi

if [[ "$BAG_HOST_PATH" != "$DATA_DIR"/* ]]; then
  echo "Bag must live under data_dir (${DATA_DIR}) so it's visible at /data in the container."
  echo "  bag:      ${BAG_HOST_PATH}"
  echo "  data_dir: ${DATA_DIR}"
  exit 1
fi

if [[ ! -f "${DATA_DIR}/target.yaml" ]]; then
  echo "Missing ${DATA_DIR}/target.yaml (AprilGrid target description)."
  exit 1
fi

BAG_IN_CONTAINER="/data/${BAG_HOST_PATH#${DATA_DIR}/}"

echo "== Running kalibr_calibrate_cameras =="
echo "   bag:    ${BAG_IN_CONTAINER}"
echo "   target: /data/target.yaml"
docker run -it --rm \
  -v "${DATA_DIR}:/data" \
  "$IMAGE_TAG" \
  bash -lc "source devel/setup.bash && cd /data && \
    kalibr_calibrate_cameras \
      --bag '${BAG_IN_CONTAINER}' \
      --target /data/target.yaml \
      --models pinhole-radtan pinhole-radtan \
      --topics /camera/left /camera/right \
      --dont-show-report"

echo
echo "Done. Check ${DATA_DIR} for camchain-*.yaml and the report pdf/txt."
echo "Rename/copy the resulting camchain yaml to calibration/camchain.yaml for the next step."
