# CLAUDE.md — MoLBWA (창종설) 프로젝트 컨텍스트

Claude Code가 이 리포에서 작업할 때 참고하는 컨텍스트 문서.

## 프로젝트
- **과제명**: 인간과 로봇의 상호작용 시스템을 위한 저가 눈동자 추적 안경 개발
- **사업**: 2026 창의적 종합설계 경진대회 / **팀 MoLBWA** / 남준현
- **목표**: 착용형 IR 아이트래킹(시선 벡터) + 스테레오-이너셜 SLAM(머리 pose)을 융합해,
  착용자가 **세계 좌표계의 어느 3D 지점을 보는지** 실시간 복원·시각화.

## 아키텍처 한눈에
```
IR 눈 카메라 → 동공/시선 벡터   ─┐
스테레오 → 깊이 D + pose(ORB-SLAM3 VI) ─┼→ 융합 → 시선이 향한 세계 3D점 p_W + 시선 광선
                                 ─┘
p_W = T_WS · (D · K⁻¹ [u,v,1]ᵀ)
```
자세한 설계는 `docs/00_overview.md` ~ `docs/08_materials_BOM.md`.

## 하드웨어
- 눈 카메라: **GC0308** (Sonix UVC 브리지, `/dev/v4l/by-id`에서 "Sonix"로 탐색) + 850nm IR
  - ※ 당초 계획한 OV9281에서 **변경됨**. 롤링셔터 / 640×480 / 30fps / 컬러센서.
  - 검출은 동작 확인. 단 **사케이드는 못 잡는다** → fixation 기반 설계. 상세 `docs/09_visualization.md`.
- 씬 카메라: **oCamS-1MGN-U** 스테레오 + 내장 IMU
- 엣지: **Raspberry Pi 5** (캡처+동공검출+스트리밍, 전송 위주)
- 데스크톱: SLAM + 융합 + 시각화

## 현재 상태 (2026-07-02)
- **0단계**: 전부 유선, 데스크톱 직결. IR 카메라 미도착으로 **RealSense D455**로 임시 대체 시험.
- 아이트래킹 알고리즘: **JEOresearch/EyeTracker** (Orlosky 검출기) 사용.
  - `external/`에 클론(리포엔 미포함) → `patches/PATCH_NOTES.md`의 numpy2 overflow 수정 필수.
- **검증됨**: RealSense IR → 동공 검출 → 3D 시선 벡터 추출까지 관통. `assets/` GIF 참고.
- 진행 로그: `notes/` (날짜별).

## 코드 (`src/`)
| 파일 | 용도 |
|------|------|
| `realsense_eye_test.py` | RealSense IR → Lite 동공검출(타원) 라이브/헤드리스 |
| `realsense_gaze_test.py` | RealSense IR → 3D 시선 벡터 추출 라이브/헤드리스 |
| `realsense_record.py` | RealSense 라이브 + 알고리즘 오버레이 녹화 |
| `run_on_eyevideo.py` | 클로즈업 눈영상으로 알고리즘 검증(오버레이+통계) |
| `make_demo_video.py` | 눈영상에 시각화 입혀 데모 mp4 생성 |
| `play_demo.py` | cv2 창으로 결과 영상 재생 |

## 작업 규칙 / 메모
- 사용자와는 **한국어**로 소통.
- RealSense는 **뒷면 USB3(파란) 포트**에 꽂아야 USB3.2로 잡힘(USB2면 프레임 안 옴).
- RealSense는 **눈 카메라 임시 대체용**일 뿐 — 눈을 화면에 못 채워 검출 품질 한계. 최종은 OV9281.
- 큰 산출물(mp4/webm)은 리포에 커밋하지 않음(`.gitignore`). README 미디어는 `assets/` GIF.
- 다음: 시선 벡터 스무딩, 스테레오+융합(docs/03) 착수, IR 카메라 도착 후 실검출/캘리브레이션.
