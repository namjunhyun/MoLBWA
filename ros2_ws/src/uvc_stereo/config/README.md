# uvc_stereo/config

캘리브레이션 뒤 여기에 생긴다 (docs/14_ov9281_stereo_uno_q.md 5-3단계):

- `left_opencv.yaml`, `right_opencv.yaml` — OpenCV FileStorage 형식(K, D, R, P), oCamS 의 같은 이름 파일과 같은 형식.
  `uvc_stereo_bridge`(정렬)와 `src/ocams_calib.py`(`SCENE_CAM=ov9281`)가 읽는다.
- `orbslam3_stereo_ov9281.yaml` — ORB-SLAM3 스테레오 전용(IMU 없음), `run_slam_ov9281.sh` 가 읽는다.

파일을 바꾼 뒤에는 `colcon build --packages-select uvc_stereo` 로 설치 폴더에 반영할 것.
