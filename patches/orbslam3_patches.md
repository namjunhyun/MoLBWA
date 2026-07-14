# SLAM 스택 패치 노트 — ORB-SLAM3 / ORB_SLAM3-ROS2

EyeTracker와 같은 이유로 이 두 레포도 **리포에 통째로 커밋하지 않는다** (ORB-SLAM3는 GPLv3 + 용량 큼,
ORB_SLAM3-ROS2는 서드파티 포크). `external/` 클론 + 패치 적용 방식을 그대로 따른다.

## 대상 레포
- [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) → `~/ORB_SLAM3`
- [zang09/ORB_SLAM3_ROS2](https://github.com/zang09/ORB_SLAM3_ROS2) → ROS2 워크스페이스 `src/orbslam3_ros2`

## 셋업
```bash
# 1) ORB-SLAM3 본체
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ~/ORB_SLAM3
cd ~/ORB_SLAM3
git apply /path/to/MoLBWA/patches/orb_slam3.patch   # base commit: patches/orb_slam3_base_commit.txt
./build.sh

# 2) ORB_SLAM3 ROS2 래퍼
git clone https://github.com/zang09/ORB_SLAM3_ROS2.git ~/ros2_ws/src/orbslam3_ros2
cd ~/ros2_ws/src/orbslam3_ros2
git apply /path/to/MoLBWA/patches/orbslam3_ros2.patch   # base commit: patches/orbslam3_ros2_base_commit.txt

# oCamS용 ORB-SLAM3 설정 파일 배치
cp /path/to/MoLBWA/ros2_ws/src/ocams_ros2/config/orbslam3_stereo_inertial_oCamS.yaml \
   ~/ros2_ws/src/orbslam3_ros2/config/stereo-inertial/oCamS.yaml

cd ~/ros2_ws && colcon build --packages-select orbslam3
```

## 패치 내용

### `orb_slam3.patch` (ORB-SLAM3 본체)
- `CMakeLists.txt`: `-Wall -w`(경고 억제) + `CMAKE_CXX_STANDARD 17` 강제 — Ubuntu 24.04/최신 컴파일러에서
  C++11 추론이 실패하는 경우가 있어 명시.
- `src/LoopClosing.cc`: `mnFullBAIdx++` → `mnFullBAIdx = mnFullBAIdx + 1` (동작 동일, 순수 스타일 변경으로
  이 세션 시작 시점에 이미 로컬에 있던 수정 — 별도 조사 불필요).

### `orbslam3_ros2.patch` (ROS2 래퍼)
- `CMakeLists.txt`:
  - **PYTHONPATH 버그 수정**: 원본이 `/opt/ros/foxy/lib/python3.8/site-packages/`로 하드코딩되어 있어
    ROS2 Jazzy(Python 3.12)에서 `ament_package` 모듈을 못 찾고 cmake configure가 실패함
    (`ModuleNotFoundError: No module named 'ament_package'`). Jazzy 경로로 수정.
  - **OpenCV 링크 누락 수정**: `stereo`/`stereo-inertial` 타깃이 `cv::initUndistortRectifyMap`
    (opencv_calib3d)을 쓰는데 `cv_bridge` 의존성만으론 안 딸려와서 링크 에러
    (`undefined reference ... initUndistortRectifyMap`). `find_package(OpenCV REQUIRED)` +
    `target_link_libraries(... ${OpenCV_LIBS})` 추가.
- `package.xml`: 위 두 타깃용 `geometry_msgs`, `tf2_ros` 의존성 추가 (아래 pose 발행 기능용).
- `src/stereo-inertial/stereo-inertial-node.{hpp,cpp}`:
  - **pose 미발행 문제 수정(신규 기능)**: 원본은 `SLAM_->TrackStereo(...)`의 리턴값(추정 pose)을
    그냥 버림 — SLAM은 돌지만 다른 노드가 pose를 받을 방법이 없었음. **이게 이 프로젝트에서 제일
    중요한 부분**: 시선 융합(`docs/03_fusion.md`)이 SLAM pose를 구독해야 하므로.
  - `geometry_msgs/msg/PoseStamped`를 `/orbslam3/pose`에 발행 (frame_id `map`), 트래킹 상태가
    `OK`일 때만 발행 (LOST/초기화 중 쓰레기 pose 안 내보냄).
  - `tf2_ros::TransformBroadcaster`로 `map → camera_left` TF도 같이 브로드캐스트 (Foxglove/RViz
    3D 시각화 및 TF 기반 조회 둘 다 가능하게).
  - Pose는 `Tcw`(world→camera, ORB-SLAM3 raw 리턴값)를 역변환한 `Twc`(camera→world)로 발행 —
    헷갈리기 쉬운 부분이니 재작업 시 주의.

## 알려진 한계 (다음 세션에서 처리)
- `orbslam3_stereo_inertial_oCamS.yaml`의 `IMU.T_b_c1`(카메라-IMU 외부 파라미터)은 **미검증 identity
  placeholder**다. 이 유닛에 대한 실제 캘리브레이션(Kalibr 등)이 없음 — 시선 융합 정확도를 신뢰하려면
  이걸 반드시 실측 캘리브레이션으로 교체해야 한다.
- `IMU.NoiseGyro/NoiseAcc/GyroWalk/AccWalk`도 myAHRS+ 데이터시트 실측치가 아니라 일반적인 consumer
  MEMS 추정치.
- `IMU.Frequency: 100.0`은 실측 확인됨(`ros2 topic hz /imu` ≈ 100Hz), 카메라도 30fps 실측 확인됨.
