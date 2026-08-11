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

## 2026-08-11 — 스테레오 rectification 재캘리브레이션 (근본 원인 수정)

### 문제

`IMU.T_b_c1` 반영 후에도 ORB-SLAM3가 계속 `New Map created` (점 40~120개) →
`Fail to track local map!` → `IMU is not or recently initialized. Reseting active map...`
루프를 반복했다. CPU 부족(헤드리스 실행으로 배제), 조명/텍스처(밝은 텍스처 있는 장면으로
재현), ORB feature 파라미터(기본값과 동일, 실제로는 좌 1141 / 우 1156개로 검출량 자체는
충분)를 순서대로 배제하고 나서, 실제 좌우 정합을 직접 측정해 원인을 찾았다.

**측정 방법**: 좌/우 rectified 프레임에서 ORB 특징점을 뽑고 ratio test(0.75)로 매칭한 뒤,
매칭 쌍의 y좌표 차이(epipolar error)를 계산. 제대로 rectify된 스테레오 쌍이면 대부분 매칭이
y오차 1~2px 이내여야 한다.

| | 기존 캘리브레이션 | |
|---|---:|---|
| ratio-test 매칭 96개 중 y오차 2px 이내 | **0개 (0%)** | 완전히 어긋남 |
| y오차 평균/표준편차 | 18.3px / 40.2px | |

**원인**: `ocams_ros2/config/left_opencv.yaml`, `right_opencv.yaml`(2026-07-14부터 사용,
`left.yaml`/`right.yaml`을 OpenCV 포맷으로 옮긴 것)의 rectification 캘리브레이션 자체가
현재 카메라의 실제 물리 정렬과 안 맞았다. 방증: 2026-08-07 Kalibr 카메라 캘리브레이션을
`/camera/left`, `/camera/right`(**드라이버가 이미 rectify해서 내보내는 영상**)에 대해
돌렸는데, 원래 rectified 영상이면 distortion이 거의 0이어야 하는데 Kalibr가 유의미한
distortion(cam0 k1=-0.13 등)을 다시 적합시켰다 — 이미 rectify된 영상을 다시
캘리브레이션하는 순환 오류였고, 이 자체가 기존 rectification이 틀렸다는 신호였다.

### 재작업 절차

1. `ocams_stereo_imu_node`를 존재하지 않는 `calib_dir`로 실행 → rectification 파일을 못
   찾아 자동으로 **raw(왜곡 보정 전) 영상**을 발행하도록 함 (`publishing UNRECTIFIED images`
   로그로 확인).
2. 이 raw 영상으로 AprilGrid를 새로 녹화 (156.7초, 좌우 각 4701프레임).
3. `scripts/kalibr/calibrate_cameras.sh`로 Kalibr 스테레오 캘리브레이션 —
   `calibration/camchain_raw.yaml`.
4. `cv2.stereoRectify()`로 Kalibr 결과(K, D, R, t)를 OpenCV rectification 포맷(K, D, R1/R2,
   P1/P2)으로 변환해 `left_opencv.yaml`/`right_opencv.yaml` 재작성 (기존 파일은
   `*.bak_20260811`로 백업).
5. `orbslam3_stereo_inertial_oCamS.yaml`의 `Camera1/2.fx/fy/cx/cy`와 `Stereo.T_c1_c2`
   baseline도 새 rectified 값으로 갱신 (이전엔 예전 rectification 기준값이 그대로 남아있어
   또 다른 불일치 요인이었음).
6. 드라이버를 정상(rectify) 모드로 재시작해 같은 epipolar-정렬 측정을 재실행.

### 결과

| 항목 | 기존 | 재캘리브레이션 후 |
|---|---:|---:|
| 재투영 오차 | — | cam0 0.214px, cam1 0.218px |
| distortion (cam0 k1) | (rectified 영상에 residual) | -0.4506 (raw 렌즈다운 값) |
| baseline | 0.12482 m (2026-07-14 원본) | **0.10671 m** (실측) |
| y오차 2px 이내 매칭 (드라이버 실제 출력, 111개 중) | 0/96 (0%) | **89/111 (80%)** |
| y오차 평균/표준편차 | 18.3px / 40.2px | **-0.12px / 4.3px** |

ORB-SLAM3 재실행 결과, `New Map created` 포인트 수가 40~120개 → **244~676개**로 증가하고
키프레임이 리셋 없이 21~93프레임까지 생존, `start VIBA 1` / `end VIBA 1`(VI 초기화 1단계
진행)까지 처음으로 도달했다. 남은 실패 모드는 `Fail to track local map!`이 아니라
`Not enough motion for initializing`로 성격이 바뀌었다 — 이건 캘리브레이션이 아니라
리셋 이후 매번 IMU 스케일/중력 추정에 필요한 만큼의 격렬한 움직임(특히 병진 가속도 변화)이
필요하다는, 훨씬 정상적인 요구사항이다.

**주의**: 2026-08-07의 camera–IMU extrinsic(`IMU.T_b_c1`)은 그 당시의 (틀렸던) rectified
스트림 기준으로 계산됐다. Kalibr가 그 스트림에 맞는 자체 카메라 모델(비영 distortion 포함)을
따로 적합시켜 self-consistent하게 풀었기 때문에 회전/이동 방향 자체는 여전히 물리적으로
타당해 보이지만(1.8cm, 직교성 검증됨), 지금 rectification을 바꿨으니 **재검증이 필요할 수
있다** — 특히 SLAM이 완전히 안정화된 후에도 pose 정확도가 부족하면 이 부분부터 의심할 것.

## 2026-08-11 — Allan variance IMU 노이즈 실측 (부분)

`ocams_stereo_imu_node`로 `/imu`만 2시간 13분 완전 정지 녹화
(`calibration/bags/allan_imu_20260811_183141`) 후 `scripts/kalibr/analyze_allan_variance.py`로
분석했다. `allan_variance_ros`(ROS1) 빌드 없이 `rosbags` + `allantools`로 오프라인 처리.

방법: 축별 overlapping Allan deviation을 구하고, log-log 곡선의 **국소 기울기**가 -1/2에
가장 가까운 지점(ADEV 최저점 이전)에서 white noise density `N`을, +1/2에 가장 가까운 지점
(최저점 이후)에서 rate random walk `K`를 읽어 3축 평균했다. (첫 구현은 "전체 tau 개수 중
앞쪽 1/3"이라는 상대적 구간으로 직선을 피팅해서 녹화가 길어질수록 진짜 white-noise 구간을
벗어나는 버그가 있었다 — gyro 기울기가 양수로 나와서 발견, 국소 기울기 탐색 방식으로 수정.)

결과:

| 파라미터 | 값 | 비고 |
|---|---:|---|
| gyroscope_noise_density | 0.005486 rad/s/√Hz | 3축 기울기 -0.50~-0.51, 신뢰 가능 |
| accelerometer_noise_density | 0.003613 m/s²/√Hz | 3축 기울기 -0.50~-0.60, 신뢰 가능 |
| accelerometer_random_walk | 0.013300 m/s³/√Hz | 3축 기울기 정확히 0.50, 신뢰 가능 |
| gyroscope_random_walk | (미측정, 기존 2.0e-04 유지) | 2h13m 동안 자이로 ADEV 곡선이 최저점을 못 찍음(계속 하강) — 4시간 이상 재녹화 필요 |

`calibration/imu.yaml`, `orbslam3_stereo_inertial_oCamS.yaml`의 `IMU.NoiseGyro/NoiseAcc/AccWalk`에
반영했다. `IMU.GyroWalk`만 아직 실측 전 추정치다.

## 한계와 다음 검증

- `IMU.GyroWalk`는 아직 Allan variance 실측값이 아니라 추정치다 (위 표 참고, 4시간+ 재녹화 필요).
- 좌우 카메라 시간 이동 추정치에 약 5.3 ms 차이가 있다. ORB-SLAM3에는 별도 적용하지 않았다.
- 최종 합격 기준은 `New Map created`/IMU reset 반복 감소와 정적 대상의 월드 시선점 고정 여부다.
- 센서 마운트를 풀거나 카메라와 IMU 상대 위치·각도가 바뀌면 다시 캘리브레이션한다.
- (2026-08-11 추가) rectification을 다시 잡았으므로 camera–IMU extrinsic도 재검증 후보다.

## 체크리스트

- [x] AprilGrid 실측 크기를 `target.yaml`에 반영
- [x] 좌/우/IMU 동시 녹화와 주기 확인
- [x] stereo 및 camera–IMU calibration 완료
- [x] 행렬 방향 역변환 후 `IMU.T_b_c1` 반영
- [x] 회전행렬 determinant/직교성 검증
- [x] 스테레오 rectification 재캘리브레이션 (2026-08-11, raw 영상 기준, epipolar 정렬 0%→80% 개선)
- [x] Allan variance로 IMU noise 실측 (2026-08-11, NoiseGyro/NoiseAcc/AccWalk 완료, GyroWalk는 재녹화 필요)
- [ ] ORB-SLAM3 장시간 안정성 비교 (VI 초기화까지는 도달, 완전 안정화는 미검증)
- [ ] 정적 대상의 `p_W` 월드 고정성 검증
- [ ] rectification 변경에 따른 camera–IMU extrinsic 재검증 필요성 판단
- [ ] 4시간+ 재녹화로 `IMU.GyroWalk` 실측 마무리
