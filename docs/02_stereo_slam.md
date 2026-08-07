# 02. 스테레오 + SLAM

## SLAM
- **ORB-SLAM3 스테레오-이너셜(VI)** 그대로 사용.
- oCamS-1MGN-U에 **IMU 내장** → VI-SLAM 구성이 자연스러움.
- 사용자는 이미 ORB-SLAM3 스테레오-이너셜 경험 있음 → 이 파트는 강점.

## 주의: SLAM 맵으로 깊이를 얻지 마라
- ORB-SLAM3 맵은 **sparse** → 임의 픽셀의 깊이를 못 준다.
- **시선 픽셀의 깊이는 SLAM 맵이 아니라 별도의 스테레오 매칭으로 얻는다.**
  - OpenCV **StereoSGBM**, 또는 oCamS SDK 깊이.
- 스테레오 정합/rectification은 OpenCV 캘리브레이션으로 **미리** 잡아둔다.

## 실시간 팁
- 전체 프레임 dense disparity를 매 프레임 계산하면 비싸다.
- **시선 픽셀 주변 윈도우(예: 64×64)에서만 disparity 계산** → 실시간 여유 大.

## 역할 정리
| 출력 | 소스 |
|------|------|
| pose `T_WS` | ORB-SLAM3 VI |
| 깊이 `D` | StereoSGBM (시선 픽셀 로컬 윈도우) |
| 내부파라미터 `K`, rectification | OpenCV 스테레오 캘리브레이션 (사전) |

## 체크리스트
- [x] oCamS 스테레오 캘리브레이션 (K, baseline, rectification 맵) — 기존 계측치 확보,
      `ros2_ws/src/ocams_ros2/config/{left,right}.yaml`. Baseline ≈0.12482m @640x480.
- [ ] StereoSGBM 로컬 윈도우 깊이 함수 작성 — 아직. (SLAM 맵과는 별도 파이프라인, 융합 단계에서 필요)
- [x] ORB-SLAM3 스테레오-이너셜 실행 확인 — 2026-07-14, 실카메라+실IMU 라이브 확인.
      `notes/2026-07-14_ocams_ros2_slam_live.md` 참고.
- [x] pose 스트림 타임스탬프 확보 — `/orbslam3/pose` (`geometry_msgs/PoseStamped`, world/`map` 프레임)
      + TF(`map`→`camera_left`), 트래킹 OK 상태일 때만 발행.

## 남은 리스크
- **카메라-IMU 외부 파라미터(`IMU.T_b_c1`) 실측 완료 (2026-08-07)** — Kalibr 결과의
  `T_cam0_imu`를 역변환해 ORB-SLAM3 설정에 반영했다. 절차, 수치, 행렬 방향과 남은 검증은
  [11_camera_imu_calibration.md](11_camera_imu_calibration.md) 참고.
- IMU 노이즈 파라미터도 데이터시트 실측치 아님 (일반 MEMS 추정치).
