# 10. RealSense D455 기반 실시간 컬러 TSDF 매핑 실험

`docs/09_visualization.md`에서 계획한 **ORB-SLAM3 pose + TSDF + Rerun** 구성을
Intel RealSense D455로 먼저 검증한 실험 기록이다.

이 실험은 최종 안경형 하드웨어(oCamS + 눈 카메라)를 대체하지 않는다. 목적은 다음 세 가지다.

1. stereo-inertial SLAM pose를 Rerun의 월드 좌표에 연결할 수 있는가
2. D455의 RGB-D 프레임을 pose에 맞춰 TSDF로 융합할 수 있는가
3. 월드 3D 맵과 카메라 영상을 한 화면에서 실시간으로 볼 수 있는가

## 데모

[D455 컬러 TSDF + Rerun 2분할 데모 영상 (WebM, 26 MB)](assets/d455_tsdf_rerun_demo.webm)

Rerun 화면은 두 패널로 고정했다.

| 패널 | 내용 |
|---|---|
| **World** | 누적 컬러 TSDF 맵, 현재 카메라 위치와 방향 |
| **Camera** | D455의 실시간 RGB 시점 |

## 구성

```text
RealSense D455
 ├─ IR Left/Right + IMU ──> ORB-SLAM3 (IMU_STEREO) ──> T_CW
 ├─ Depth (Z16) ──────────┐
 └─ RGB ── texture map ───┴─> aligned RGB-D
                                      │
                                      v
                         Open3D ScalableTSDFVolume
                                      │
                       ┌──────────────┴──────────────┐
                       v                             v
                 Rerun World 3D              d455_tsdf_map.ply
```

깊이 픽셀과 RGB 색의 대응은 librealsense의 texture coordinate를 사용한다.
TSDF에는 ORB-SLAM3가 출력한 `T_CW`를 extrinsic으로 전달한다.

## 주요 파라미터

| 항목 | 값 |
|---|---|
| 스테레오/깊이/RGB | 640×480, 30 Hz |
| Gyro / Accel | 200 Hz / 100 Hz |
| Stereo baseline | 약 95.08 mm |
| TSDF voxel | 2.5 cm |
| TSDF truncation | voxel의 4배 |
| 키프레임 선택 | 이동 4 cm 또는 회전 4° 이상 |
| 맵 추출 | 15 키프레임마다 |
| Rerun 메모리 상한 | 3 GiB |

## 파일

- `src/d455_tsdf_rerun.py`: RGB-D TSDF 융합, PLY 저장, Rerun 2분할
- `assets/RealSense_D455_stereo_inertial.yaml`: ORB-SLAM3 D455 설정
- `docs/assets/d455_tsdf_rerun_demo.webm`: 실행 영상

## 설치

전용 Python 환경에서 다음 패키지가 필요하다.

```bash
python3 -m venv realsense_rerun_env
source realsense_rerun_env/bin/activate
pip install numpy pillow rerun-sdk==0.35.0 open3d==0.19.0
```

ORB-SLAM3의 `stereo_inertial_realsense_D435i` 예제에는 다음 출력 연결이 추가로 필요하다.

- `/tmp/d455_slam_pose.csv`: timestamp, tracking state, 4×4 `T_CW`
- `/tmp/d455_depth.png`: Z16 depth
- `/tmp/d455_color.png`: depth 픽셀에 정렬한 RGB

## 실행

먼저 D455 stereo-inertial SLAM을 실행한다.

```bash
ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_realsense_D435i \
  ORB_SLAM3/Vocabulary/ORBvoc.txt \
  assets/RealSense_D455_stereo_inertial.yaml \
  D455_rgb_tsdf
```

다른 터미널에서 TSDF/Rerun 브리지를 실행한다.

```bash
source realsense_rerun_env/bin/activate
python src/d455_tsdf_rerun.py
```

맵은 실행 중 주기적으로 `~/d455_tsdf_map.ply`에 저장된다. `Ctrl+C`로 종료할 때도
최종 point cloud를 한 번 더 저장한다.

## 단순 점 누적과 다른 점

초기 구현은 매 깊이 프레임의 점을 월드 좌표로 변환해 그대로 추가했다. 빠르지만 같은 표면이
여러 겹으로 쌓이고, 점 개수와 Rerun 메모리가 계속 증가했다. 실제로 Rerun이 약 9.5 GB까지
사용한 뒤 Linux OOM killer에 의해 종료됐다.

현재 구현은 동일 공간의 반복 관측을 TSDF에 융합한다.

- 같은 표면의 깊이 노이즈가 평균화된다.
- 중복점 대신 하나의 표면으로 수렴한다.
- Rerun에는 일정 간격으로 추출한 정제된 컬러 point cloud만 전달한다.
- PLY 결과가 남아 Rerun 종료 후에도 확인할 수 있다.

## 현재 한계

1. **SLAM 초기화 동작이 필요하다.** 시작 직후 카메라를 회전만 하지 말고 앞뒤·좌우 이동과
   회전을 섞어야 IMU 초기화가 안정된다.
2. **ORB-SLAM3 백엔드가 한 차례 segfault로 종료됐다.** 장시간 안정성 검증과 자동 재시작이
   아직 필요하다.
3. **루프 폐쇄 후 TSDF 재융합은 아직 없다.** ORB-SLAM3가 과거 keyframe pose를 보정해도
   이미 융합한 TSDF를 자동으로 다시 만들지는 않는다.
4. 이 실험은 D455 검증 경로다. 최종 MoLBWA의 oCamS 깊이 및 gaze hit/fixation heatmap
   연결은 `docs/09_visualization.md`의 후속 작업으로 남아 있다.

## 확인된 결과

- D455 stereo-inertial pose와 RGB-D 동시 취득
- RGB와 depth의 보정 기반 대응
- Open3D 컬러 TSDF 실시간 융합
- Rerun `World | Camera` 2분할
- 컬러 point cloud PLY 주기 저장

따라서 `docs/09_visualization.md`의 C(ORB-SLAM3 pose 연결)와 F(TSDF 누적)는
**D455 실험 환경에서는 동작을 확인했다.** 최종 하드웨어 경로의 완료를 의미하지는 않는다.
