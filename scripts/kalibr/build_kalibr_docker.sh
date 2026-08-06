#!/usr/bin/env bash
# Clone ethz-asl/kalibr and build its Docker image (ROS1, Ubuntu 20.04 route),
# per https://github.com/ethz-asl/kalibr/wiki/installation
#
# Kalibr itself is only needed inside the container, so the clone lives under
# external/ (already gitignored, same convention as the other third-party
# clones in this repo — see patches/PATCH_NOTES.md).
#
# Usage: scripts/kalibr/build_kalibr_docker.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

REPO_DIR="external/kalibr"
IMAGE_TAG="kalibr"
DOCKERFILE="Dockerfile_ros1_20_04"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "== Cloning ethz-asl/kalibr into ${REPO_DIR} =="
  git clone https://github.com/ethz-asl/kalibr.git "$REPO_DIR"
else
  echo "== ${REPO_DIR} already exists, pulling latest =="
  git -C "$REPO_DIR" pull --ff-only
fi

if [[ ! -f "${REPO_DIR}/${DOCKERFILE}" ]]; then
  echo "Expected ${DOCKERFILE} not found in ${REPO_DIR} — kalibr's repo layout"
  echo "may have changed. Check https://github.com/ethz-asl/kalibr/wiki/installation"
  exit 1
fi

echo "== Building docker image '${IMAGE_TAG}' (this takes a while, ROS1 + catkin build) =="
docker build -t "$IMAGE_TAG" -f "${REPO_DIR}/${DOCKERFILE}" "$REPO_DIR"

echo
echo "Done. Run scripts/kalibr/run_kalibr_docker.sh <data_dir> to open a shell."
