# 창종설 — Gaze-in-World SLAM (착용형 시선 3D 매핑)

착용형 IR 아이트래킹으로 **시선 벡터**를 얻고, 스테레오-이너셜 SLAM으로 **머리(카메라)의 6DoF pose**를 얻어,
"사용자가 지금 세계 좌표계의 어느 3D 지점을 보고 있는가"를 실시간으로 복원·시각화하는 프로젝트.

> 2026 창의적 종합설계 경진대회 · **팀 MoLBWA** · 남준현

## 데모 — RealSense IR 실시간 시선 벡터 추출
IR 카메라 도착 전, 연구실 **RealSense D455**로 눈 카메라 자리를 임시 대체해 파이프라인을 검증한 결과.
동공에 타원 락온 → 안구 구 모델 → **3D 시선 벡터**(`Direction`)까지 실시간 추출.

![RealSense 시선 추적 데모](assets/realsense_eyetracking.gif)

> 알고리즘: [JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker) (Orlosky 검출기) + numpy2 overflow 수정.
> RealSense는 임시 대체용이며, 최종 OV9281 밀착 IR 카메라면 더 안정적입니다.

## 한 줄 요약
```
IR 눈 카메라 → 동공/시선 벡터 (pye3d)
스테레오 카메라 → 깊이 (StereoSGBM) + pose (ORB-SLAM3 VI)
        └── 융합 → 시선이 향하는 세계 3D점 p_W + 시선 광선
```

## 하드웨어
| 부품 | 모델 | 역할 |
|------|------|------|
| 눈 카메라 | OV9281 (B0332), 모노 글로벌셔터 | IR 동공 촬영 (**아직 미도착**) |
| IR 조명 | 850nm LED + IR 밴드패스 필터 | 조명 안정화 (필터 유무가 검출 안정성 좌우) |
| 씬 카메라 | oCamS-1MGN-U (스테레오 + 내장 IMU) | 스테레오 깊이 + VI-SLAM |
| 엣지 | Raspberry Pi 5 | 캡처 + 동공검출 + 스트리밍 (전송 위주) |
| 데스크톱 | (본체) | SLAM + 융합 + 시각화 |

> **임시**: IR 카메라 도착 전, 연구실 **RealSense**로 눈 카메라 자리를 대체해 파이프라인을 먼저 검증한다. → [notes/2026-07-02_realsense_임시테스트.md](notes/2026-07-02_realsense_임시테스트.md)

## 문서 지도
- [docs/00_overview.md](docs/00_overview.md) — 전체 아키텍처 / 변환 사슬
- [docs/01_eye_tracking.md](docs/01_eye_tracking.md) — 동공 검출 + 캘리브레이션
- [docs/02_stereo_slam.md](docs/02_stereo_slam.md) — 스테레오 깊이 + ORB-SLAM3
- [docs/03_fusion.md](docs/03_fusion.md) — 융합 수식 (프로젝트의 심장)
- [docs/04_sync.md](docs/04_sync.md) — 타임스탬프 동기화
- [docs/05_pi_streaming.md](docs/05_pi_streaming.md) — Pi 스트리밍 / 대역폭
- [docs/06_roadmap.md](docs/06_roadmap.md) — 0~3단계 실행 계획
- [docs/07_accuracy.md](docs/07_accuracy.md) — 정확도 기대치 / 설계 지침
- [docs/08_materials_BOM.md](docs/08_materials_BOM.md) — 물품 구매 목록(BOM)

## 셋업 (아이트래킹 시험)
아이트래킹 알고리즘은 서드파티 레포를 쓰므로 별도 클론 + 패치가 필요하다 (리포엔 미포함).
```bash
# 1) 아이트래커 클론
git clone https://github.com/JEOresearch/EyeTracker.git external/EyeTracker

# 2) numpy2 uint8 overflow 수정 (필수) — patches/PATCH_NOTES.md 의 int() 캐스팅 4곳
#    (안 하면 동공 대신 눈꺼풀 전체를 잡는 오검출 발생)

# 3) 의존 패키지
pip install numpy opencv-python pyrealsense2 pillow

# 4) RealSense IR로 시선 벡터 추출 시험 (RealSense는 뒷면 USB3 포트에 연결)
python src/realsense_gaze_test.py --live
```

## 현재 단계
**0단계 (전부 유선, 데스크톱 직결)** — 각 센서가 독립적으로 잘 도는지 확인 중.
IR 카메라 미도착으로 RealSense로 대체 시험.

## 진행 기록
`notes/` 에 날짜별로 남긴다. 최신: [notes/](notes/)
