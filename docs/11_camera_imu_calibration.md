# 11. oCamS camera–IMU 캘리브레이션

## 목적과 완료 상태

oCamS-1MGN-U의 스테레오 카메라와 내장 myAHRS+ IMU 사이 외부 파라미터를 Kalibr로 실측한다.
이 값은 ORB-SLAM3 stereo-inertial 설정의 `IMU.T_b_c1`에 들어간다. 기존 identity placeholder는
VI-SLAM 반복 초기화의 주요 원인 후보였다.

2026-08-07 실측을 완료하고 다음 파일에 반영했다.

- `calibration/camchain.yaml`: 새 stereo camera chain
- `calibration/camchain-imucam.yaml`: 최종 camera–IMU 결과
- `ros2_ws/src/ocams_ros2/config/orbslam3_stereo_inertial_oCamS.yaml`: ORB-SLAM3용 extrinsic

대용량 bag과 보고서는 `calibration/bags/`에 보관하며 Git에는 커밋하지 않는다.

## 좌표계와 행렬 방향

이 방향을 틀리면 수치가 좋아도 ORB-SLAM3가 정상 동작하지 않는다.

- Kalibr `T_cam_imu`: IMU → 카메라, `p_cam = T_cam_imu · p_imu`
- ORB-SLAM3 `IMU.T_b_c1`: 왼쪽 카메라 → IMU body
- 따라서 `T_b_c1 = inverse(T_cam0_imu)`

Kalibr YAML의 `T_cam_imu`를 ORB-SLAM3에 그대로 복사하면 안 된다.

## 센서와 토픽

| 센서 | ROS 2 토픽 | 실측 주기 | 메시지 |
|---|---|---:|---|
| 왼쪽 카메라 | `/camera/left` | 약 30 Hz | `sensor_msgs/msg/Image` |
| 오른쪽 카메라 | `/camera/right` | 약 30 Hz | `sensor_msgs/msg/Image` |
| myAHRS+ IMU | `/imu` | 약 100 Hz | `sensor_msgs/msg/Imu` |

드라이버는 `ros2_ws/src/ocams_ros2`에 있다. Withrobot SDK는 라이선스 때문에 포함하지 않으므로
패키지 `README.md`에 따라 별도로 배치한다.

## AprilGrid 준비

기준 파일은 `calibration/apriltag_grid.pdf`, 설정은 `calibration/target.yaml`이다.

```yaml
target_type: 'aprilgrid'
tagCols: 6
tagRows: 6
tagSize: 0.020
tagSpacing: 0.3
```

이번 출력물은 프린터 여백 때문에 축소했고, 큰 검정 AprilTag 한 개의 바깥쪽 한 변을 실측한 값이
20 mm였다. 따라서 `tagSize`는 `0.020 m`를 사용했다. 작은 모서리 표시나 보드 전체 폭을 재는 것이
아니다. 태그 간격은 약 6 mm여서 `tagSpacing = 6 / 20 = 0.3`이다.

PDF 명목 치수는 25 mm다. 다른 프린터로 출력하면 반드시 다시 측정한다. 예를 들어 실제 태그가
24.7 mm면 `tagSize: 0.0247`로 바꾼다.

- 모든 태그가 잘리지 않게 출력한다.
- 휘지 않도록 평평하고 단단한 판에 붙인다.
- 반사, 그림자, 모션 블러를 피한다.
- 보드는 고정하고 카메라+IMU 장치를 한 덩어리로 움직인다.

## 녹화

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ocams_ros2
source install/setup.bash
ros2 run ocams_ros2 ocams_stereo_imu_node
```

다른 터미널에서 좌우 영상을 확인하고 녹화한다.

```bash
scripts/record_imu_cam_bag.sh
```

2~3분 동안 시작·끝에 약 2초 정지한다. 보드가 양쪽 카메라에 최대한 보이게 유지하면서 좌우·상하·
앞뒤 병진과 yaw·pitch·roll 회전을 천천히 준다. 화면 네 모서리와 다양한 거리·기울기를 커버한다.

이번 bag은 129.09초, 좌우 각 3,884장, IMU 12,948개였다.

## 실행 절차

### 1. ROS 1 bag으로 변환

```bash
scripts/convert_bag_to_ros1.sh \
  calibration/bags/imu_cam_calib_20260807_final \
  calibration/bags/imu_cam_calib_20260807_final.bag
```

Kalibr가 ROS 1 기반이므로 변환한다. 도구는 `calibration/.convert_venv`에 격리 설치된다.

### 2. Kalibr Docker 준비

```bash
scripts/kalibr/build_kalibr_docker.sh
```

### 3. Stereo camera calibration

```bash
scripts/kalibr/calibrate_cameras.sh \
  calibration/bags/imu_cam_calib_20260807_final.bag calibration
```

스크립트는 중복 프레임을 줄이기 위해 `--bag-freq 4.0`을 사용한다. 샘플링된 486장 중 왼쪽
430장, 오른쪽 454장에서 코너를 검출했다.

```text
calibration/bags/<bag>-camchain.yaml
calibration/bags/<bag>-results-cam.txt
calibration/bags/<bag>-report-cam.pdf
```

### 4. Camera–IMU calibration

```bash
scripts/kalibr/calibrate_imu_camera.sh \
  calibration/bags/imu_cam_calib_20260807_final.bag \
  calibration \
  bags/imu_cam_calib_20260807_final-camchain.yaml
```

```text
calibration/bags/<bag>-camchain-imucam.yaml
calibration/bags/<bag>-results-imucam.txt
calibration/bags/<bag>-report-imucam.pdf
```

## 2026-08-07 결과

| 항목 | 결과 |
|---|---:|
| cam0 재투영 오차 평균 | 0.443 px |
| cam1 재투영 오차 평균 | 0.546 px |
| 자이로 잔차 평균 | 0.0807 rad/s |
| 가속도 잔차 평균 | 0.233 m/s² |
| cam0→IMU 시간 이동 | 2.581 ms |
| cam1→IMU 시간 이동 | 7.895 ms |

Kalibr `T_cam0_imu`:

```text
[-0.3091073864 -0.9504632103 -0.0327461370  0.0174068685]
[ 0.0209662011  0.0276135270 -0.9993987750  0.0009279676]
[ 0.9507960044 -0.3096081054  0.0113920577 -0.0031530436]
[ 0.0000000000  0.0000000000  0.0000000000  1.0000000000]
```

ORB-SLAM3 `IMU.T_b_c1 = inverse(T_cam0_imu)`:

```text
[-0.3091073864  0.0209662011  0.9507960044  0.0083590369]
[-0.9504632103  0.0276135270 -0.3096081054  0.0155427558]
[-0.0327461370 -0.9993987750  0.0113920577  0.0015333370]
[ 0.0000000000  0.0000000000  0.0000000000  1.0000000000]
```

회전 블록은 `det(R) = 1.0000000000003`, 직교성 오차 약 `1.39e-12`로 검증했다.

## ORB-SLAM3 적용 원칙

위 역행렬로 `orbslam3_stereo_inertial_oCamS.yaml`의 identity placeholder를 교체했다.

드라이버 입력 영상은 이미 rectified되어 있다. 따라서 ORB-SLAM3 내부 파라미터와 zero-distortion은
기존 rectified 스트림 값을 유지하고, 이번 결과에서는 camera–IMU extrinsic만 적용했다.
`calibration/camchain.yaml`의 radtan 값을 ORB 설정에 그대로 복사하지 않는다.

## 한계와 다음 검증

- `calibration/imu.yaml`의 noise density/random walk는 Allan variance 실측값이 아니라 추정치다.
- 좌우 카메라 시간 이동 추정치에 약 5.3 ms 차이가 있다. ORB-SLAM3에는 별도 적용하지 않았다.
- 최종 합격 기준은 `New Map created`/IMU reset 반복 감소와 정적 대상의 월드 시선점 고정 여부다.
- 센서 마운트를 풀거나 카메라와 IMU 상대 위치·각도가 바뀌면 다시 캘리브레이션한다.

## 체크리스트

- [x] AprilGrid 실측 크기를 `target.yaml`에 반영
- [x] 좌/우/IMU 동시 녹화와 주기 확인
- [x] stereo 및 camera–IMU calibration 완료
- [x] 행렬 방향 역변환 후 `IMU.T_b_c1` 반영
- [x] 회전행렬 determinant/직교성 검증
- [ ] Allan variance로 IMU noise 실측
- [ ] ORB-SLAM3 장시간 안정성 비교
- [ ] 정적 대상의 `p_W` 월드 고정성 검증
