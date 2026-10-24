#!/usr/bin/env bash
# kalibr 도커 이미지가 쓸 수 있는 상태인지: 패키지 경로와 캘리브레이션 명령이 있는지.
docker run --rm --entrypoint bash kalibr -lc \
  "source /catkin_ws/devel/setup.bash && rospack find kalibr && ls \$(rospack find kalibr)/python | grep -E 'kalibr_calibrate_(cameras|imu_camera)'"
