#!/bin/bash
# (R,p_eye) 실전용: 'y' 반전하고 찍은 캘리브(calibration/gaze_scene_extrinsic.json)를 그대로 쓴다.
# --mirror-y = 캘리브 때 누른 'y' 와 같은 축 반전, --restore-eye-model = 저장된 안구 모델 복원.
# 안경을 벗었다 다시 썼으면 안구 모델이 달라지므로 이걸 쓰지 말고 새로 캘리브(f→y→e)할 것.
cd ~/MoLBWA-gaze-hri/src
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
# 씬 영상은 ./run_cam_relay.sh 중계(/pc/...)를 받는다. 중계 없이 Pi 직접: CAM_PREFIX= ./run_gaze_on_scene_ray.sh
exec python3 -u gaze_on_scene.py --no-rerun --source ros --flip --send-gaze-px --mirror-y --restore-eye-model \
  --left-topic "${CAM_PREFIX-/pc}/camera/left/compressed" --right-topic "${CAM_PREFIX-/pc}/camera/right/compressed"
