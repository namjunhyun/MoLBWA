#!/usr/bin/env bash
# Open a shell in the kalibr docker image (built by build_kalibr_docker.sh),
# with a host data directory mounted at /data so kalibr can read the
# converted ROS1 bag + calibration/target.yaml and write its report there.
#
# Usage:
#   scripts/kalibr/run_kalibr_docker.sh [data_dir]
# data_dir defaults to ./calibration (relative to repo root).
#
# Inside the container, kalibr's workspace is pre-sourced, so you can run
# e.g.:
#   kalibr_calibrate_cameras --bag /data/bags/<name>.bag \
#       --target /data/target.yaml --models pinhole-radtan pinhole-radtan \
#       --topics /camera/left /camera/right --dont-show-report
#   kalibr_calibrate_imu_camera --bag /data/bags/<name>.bag \
#       --target /data/target.yaml --cam /data/camchain.yaml \
#       --imu /data/imu.yaml --dont-show-report
#
# --dont-show-report avoids needing a working X11 display inside the
# container; the PDF/txt report still gets written to /data on the host.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

IMAGE_TAG="kalibr"
DATA_DIR="$(realpath "${1:-calibration}")"

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Docker image '${IMAGE_TAG}' not found. Build it first:"
  echo "  scripts/kalibr/build_kalibr_docker.sh"
  exit 1
fi

mkdir -p "$DATA_DIR"

X11_ARGS=()
if [[ -n "${DISPLAY:-}" ]] && command -v xhost >/dev/null 2>&1; then
  if xhost +local:root >/dev/null 2>&1; then
    X11_ARGS=(-e DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
    echo "== X11 forwarding enabled (only needed if you skip --dont-show-report) =="
  else
    echo "== xhost failed, continuing without X11 forwarding (use --dont-show-report inside) =="
  fi
else
  echo "== No DISPLAY/xhost found, continuing without X11 forwarding (use --dont-show-report inside) =="
fi

echo "== Mounting ${DATA_DIR} -> /data =="
docker run -it --rm \
  "${X11_ARGS[@]}" \
  -v "${DATA_DIR}:/data" \
  "$IMAGE_TAG" \
  bash -lc "source devel/setup.bash && exec bash"
