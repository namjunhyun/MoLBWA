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
- [ ] oCamS 스테레오 캘리브레이션 (K, baseline, rectification 맵)
- [ ] StereoSGBM 로컬 윈도우 깊이 함수 작성
- [ ] ORB-SLAM3 스테레오-이너셜 실행 확인 (기존 경험 활용)
- [ ] pose 스트림 타임스탬프 확보
