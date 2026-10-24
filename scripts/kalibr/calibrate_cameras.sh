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
# KALIBR_MANUAL_FOCAL_LENGTH_INIT is normally exported by the image's own
# ENTRYPOINT, but --entrypoint bash below bypasses that, so it's set here
# explicitly. When Kalibr's homography-based focal length auto-init fails for
# a camera, it blocks on a single `std::cin >> focal_length` per failure;
# `yes 400` keeps feeding a reasonable guess (oCamS fx is ~400-425px at
# 640x480, see ocams_ros2/config/{left,right}_opencv.yaml) so the run never
# hangs waiting for a TTY, whether or not any camera actually needs it.
# yes's write fails with SIGPIPE once docker's stdin closes at exit; that's
# expected, so check docker's own exit code (PIPESTATUS[1]) rather than the
# pipeline's overall status.
# 카메라마다 초점 추정값이 다르다: oCamS 640x480 ≈ 400px, OV9281(105°) 640x480 ≈ 300px.
# 렌즈 모델·토픽도 바꿀 수 있다 (광각이라 가장자리 잔차가 크면 MODELS="pinhole-equi pinhole-equi").
FOCAL_GUESS="${FOCAL_GUESS:-400}"
MODELS="${MODELS:-pinhole-radtan pinhole-radtan}"
TOPICS="${TOPICS:-/camera/left /camera/right}"
echo "   focal guess: ${FOCAL_GUESS} · models: ${MODELS} · topics: ${TOPICS}"
set +o pipefail
yes "${FOCAL_GUESS}" | docker run -i --rm \
  --entrypoint bash \
  -e KALIBR_MANUAL_FOCAL_LENGTH_INIT=1 \
  -v "${DATA_DIR}:/data" \
  "$IMAGE_TAG" \
  -lc "source /catkin_ws/devel/setup.bash && cd /data && \
    rosrun kalibr kalibr_calibrate_cameras \
      --bag '${BAG_IN_CONTAINER}' \
      --bag-freq 4.0 \
      --target /data/target.yaml \
      --models ${MODELS} \
      --topics ${TOPICS} \
      --dont-show-report"
DOCKER_STATUS=${PIPESTATUS[1]}
set -o pipefail
if [[ "$DOCKER_STATUS" -ne 0 ]]; then
  echo "kalibr_calibrate_cameras failed (exit ${DOCKER_STATUS})"
  exit "$DOCKER_STATUS"
fi

echo
echo "Done. Check ${DATA_DIR} for camchain-*.yaml and the report pdf/txt."
echo "Rename/copy the resulting camchain yaml to calibration/camchain.yaml for the next step."
