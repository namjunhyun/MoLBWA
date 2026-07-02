# 2026-07-02 — RealSense 임시 테스트 (IR 카메라 도착 전)

## 상황
- 눈 카메라(OV9281/B0332)가 **아직 안 옴**.
- 연구실에 있는 **RealSense**로 눈 카메라 자리를 대체해 파이프라인을 먼저 시험.
- 목적: 하드웨어 최종본 없이도 **소프트웨어 파이프라인 / 융합 수식 / 캘리브레이션 절차**를 미리 굴려보기.

## RealSense로 뭘 대체할 수 있나
| 최종 하드웨어 | RealSense 임시 대체 | 비고 |
|---------------|---------------------|------|
| OV9281 IR 눈 카메라 | RealSense **IR 스트림**(적외선 imager) | RealSense는 IR 프로젝터/IR 카메라 내장 → 동공 검출 입력 흉내 가능 |
| oCamS 스테레오 깊이 | RealSense **depth 스트림** | D 값을 SDK가 직접 줌 → 융합 수식 입력 확보 |
| oCamS + IMU VI-SLAM | (D400은 IMU 없음 / D435i·D455는 IMU 있음) | 모델 확인 필요 |

## 먼저 확인할 것 (다음 세션 작업)
1. 연구실 RealSense **모델명** (D435 / D435i / D455 …) — IMU 유무·IR 스트림 확인.
2. `pyrealsense2` 설치 및 스트림 열기 (`rs-enumerate-devices`).
3. IR 스트림 캡처 → pye3d 동공 검출 시험 (사람 눈 클로즈업으로).
4. depth 스트림으로 융합 수식 `gaze_point_world` 입력 D 확보.

## 주의
- RealSense IR은 **구조광 프로젝터 패턴**이 눈에 찍힐 수 있음 → 동공 검출엔 프로젝터 끄고(emitter off) 순수 IR 조명만 쓰는 게 나을 수 있음. 시험해서 판단.
- 어디까지나 **임시**. 최종은 OV9281 + 850nm + 밴드패스 필터.

## 다음에 할 일
- [ ] `git clone <교수님/팀 GitHub 링크>` → `external/` 또는 지정 위치
- [ ] RealSense 모델 확인 + pyrealsense2 스트림 테스트
- [ ] pye3d 설치 + IR 프레임 동공검출 PoC
- [ ] 융합 수식에 RealSense depth 꽂아 책상 검증
