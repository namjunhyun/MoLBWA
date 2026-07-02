# 04. 동기화 — 조용히 프로젝트를 망치는 부분

세 스트림(눈 카메라, 스테레오, IMU)의 타임스탬프를 **하나의 시계**로 맞춰야 한다.

## 원칙
- `t` 시점의 시선은 **반드시** `t` 시점의 SLAM pose와 결합.
- 프레임레이트가 다르므로 pose를 **시선 타임스탬프로 보간(interpolate)**해서 사용.
  - translation: 선형 보간(lerp), rotation: **SLERP**.

## ROS2 백본 (이미 사용 중)
- `message_filters`의 **ApproximateTimeSynchronizer** + **TF** 가 이걸 대신 처리.
- 사용자가 이미 ROS2 사용 → 이걸 백본으로 삼는 게 자연스러움.

## 하이브리드 통신 재확인
- **영상**: GStreamer 별도 스트림 (docs/05).
- **시선벡터 / pose / 제어**: ROS2 토픽.
  - ROS2 이미지 토픽을 Wi-Fi로 그냥 쏘는 건 DDS 디스커버리·QoS 문제로 큰 데이터에 약함.

## 실전 팁
- 각 스트림에 **캡처 시각 타임스탬프**를 소스에서 찍어라 (수신 시각 아님).
- Pi ↔ 데스크톱 **시계 동기화**(chrony/NTP 또는 PTP) 먼저 맞춰라. 안 그러면 보간이 틀어짐.

## 체크리스트
- [ ] Pi ↔ 데스크톱 NTP/PTP 시계 동기화
- [ ] pose SLERP+lerp 보간 함수
- [ ] ApproximateTimeSynchronizer 파이프라인 구성
