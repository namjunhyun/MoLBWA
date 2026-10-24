# 14. OV9281 USB 카메라 두 대 + Arduino UNO Q 중계 스테레오 (oCamS 대체)

> 2026-10-24 · 브랜치 `feat/ov9281-stereo` (기반 `feat/gaze-hri-pick-place`)
> 카메라: Waveshare **OV9281 1MP Mono USB Camera (A)** ×2 — 글로벌 셔터, 흑백, USB 2.0 Type-C, 105°, M12

## 한눈에

```
OV9281 왼 ─┐                                   TCP :55100                    ROS 2 (기존 소비 코드 그대로)
           ├─ USB-C 허브(PD) ─ UNO Q ──────────────────────────► PC uvc_stereo_bridge ──► /pc/camera/{left,right}/compressed
OV9281 오 ─┘   stereo_relay.py: MJPG 그대로,                       디코드 → 정렬(rectify)        (정렬된 640x480 mono)
               커널 시각으로 좌우 짝 맞춤                           → 다시 JPEG
```

- **기존 계약을 그대로 지킨다**: 소비 코드(`gaze_on_scene`, `gaze_tag_bridge`, `run_demo`, `tag_check`…)는 모두
  "이미 정렬된 640×480 흑백 + `{left,right}_opencv.yaml` 의 P" 를 기대한다. 드라이버와 캘리브레이션 파일만 바뀐다.
- `ocams_calib.py` 는 **`SCENE_CAM=ov9281`** 이면 `ros2_ws/src/uvc_stereo/config/` 를 읽는다(기본은 여전히 oCamS).
- oCamS 경로의 "Pi `~/ocams.sh` + `run_cam_relay.sh`" 는 `edge/uno_q/stereo_relay.py` + `run_uvc_stereo.sh` 로 바뀐다.

## 알고 시작할 것 (oCamS 와 다른 점)

| 항목 | oCamS | OV9281 두 대 | 영향 |
|---|---|---|---|
| 좌우 동기 | 한 프레임에 둘이 섞여 나옴(완전 동기) | **하드웨어 동기 없음** (트리거 핀 없음) | 프레임마다 좌우 어긋남이 생긴다. 중계가 커널 시각으로 가장 가까운 것끼리 짝짓고 어긋남을 5초마다 출력. 30 fps 면 최악 ±16.7 ms — 머리를 빨리 돌리면 그만큼 깊이가 틀어진다 |
| IMU | 내장 (myAHRS+) | **없음** | ORB-SLAM3 는 **스테레오 전용**(`run_slam_ov9281.sh`). 지금 데모 경로(태그 직접, SLAM 안 씀)는 영향 없음 |
| USB | 하나 | **USB 2.0 두 대** | 무압축 두 대는 대역폭 초과 → **MJPG** 로 받는다. 가능하면 허브의 서로 다른 포트에 |
| 같은 모델 두 대 | — | 장치 이름(by-id)이 같을 수 있다 | **꽂은 포트(by-path)로** 왼·오를 고른다 — 포트를 바꿔 꽂으면 좌우가 바뀐다 |
| 렌즈 | 고정 | M12 · 105° 광각 | 초점 링 확인(돌려서 맞춘 뒤 고정), 광각이라 왜곡이 큼 |

**장착:** 두 카메라를 **평행하게, 단단히** (조금이라도 휘면 캘리브레이션이 깨진다). 간격(baseline)은 안경 폭이 허락하는 만큼
(6~10 cm). 장착을 바꾸면 5단계 캘리브레이션을 다시 해야 한다.

---

## 1. UNO Q 준비 (한 번)

- **전원 공급(PD)되는 USB-C 허브** 필요 — UNO Q 의 USB-C 가 하나뿐이다. Arduino 문서는 USB 카메라엔 **4 GB 모델**을 권한다.
- UNO Q 리눅스(Debian)에서:
  ```bash
  sudo apt update && sudo apt install -y python3-opencv v4l-utils
  mkdir -p ~/stereo   # 여기에 edge/uno_q/stereo_relay.py 복사 (scp)
  ```
- PC → UNO Q 복사 예: `scp edge/uno_q/stereo_relay.py arduino@<UNO Q IP>:stereo/`

## 2. 어느 카메라가 왼쪽인지 (한 번, 허브 포트를 바꾸면 다시)

```bash
python3 ~/stereo/stereo_relay.py --list
```
- 장치마다 `/dev/v4l/by-path/...video-index0` 이름, 지원 형식(MJPG 가 있어야 함)·해상도, 사진 `/tmp/cam_0.jpg`… 가 나온다.
- **한쪽 렌즈를 손으로 가리고 다시 돌리면** 밝기 평균이 낮게 나오는 쪽이 그 카메라다. 착용자 기준 왼쪽 = `--left`.
- `v4l2-ctl -d <장치> -l` 로 노출 컨트롤 이름을 확인해 둔다(3단계 `--controls`).

## 3. 중계 시작

UNO Q:
```bash
python3 ~/stereo/stereo_relay.py \
  --left  /dev/v4l/by-path/<왼쪽>-video-index0 \
  --right /dev/v4l/by-path/<오른쪽>-video-index0 \
  --width 640 --height 480 --fps 30 \
  --controls "auto_exposure=1,exposure_time_absolute=150"     # 이름·값은 카메라에 맞게, 두 대 똑같이
```
- 5초마다 `받은 프레임 · 보낸 쌍 · 어긋남 ms 중앙/95%/최대` 가 찍힌다. **보낸 쌍이 fps 에 가깝고 어긋남 중앙이 10 ms 안쪽**이면 정상.
- 부팅 때 자동 시작: `edge/uno_q/stereo-relay.service` (장치 경로 고쳐서 설치).

PC (캘리브레이션 전에는 **원본 모드**):
```bash
cd ros2_ws && colcon build --packages-select uvc_stereo && cd ..
UNO_Q=<UNO Q IP> RAW=1 ./run_uvc_stereo.sh          # → /camera/left, /camera/right (정렬 안 된 원본 + compressed)
```

## 4. 카메라 없이 먼저 시험하고 싶으면

```bash
bash ros2_ws/src/uvc_stereo/test/run_fake_e2e.sh          # 가짜 카메라(좌우 7 ms 어긋남) → 브리지 → 토픽: 30 쌍/s 기대
bash ros2_ws/src/uvc_stereo/test/run_calib_chain_test.sh  # 가짜 Kalibr 결과 → 변환 → 정렬 켠 브리지까지
```

## 5. 캘리브레이션 (내부 파라미터 + 좌우 위치) — Kalibr, oCamS 때와 같은 절차

### 5-0. 준비 (한 번)
- **도커 저장 위치를 /mnt/data 로** (루트가 99% 참): `sudo bash scripts/move_docker_to_mnt_data.sh` 다음 `sudo bash scripts/move_containerd_to_mnt_data.sh` — 도커 29 는 이미지를 containerd 쪽(`/var/lib/containerd`)에 두므로 **둘 다** 옮겨야 한다. 확인 뒤 `sudo bash scripts/remove_docker_bak.sh`. (2026-10-24 이 PC 에서 완료: 루트 9.7 → 49 GB 여유)
- **Kalibr 도커 이미지**: `scripts/kalibr/build_kalibr_docker.sh` (`external/kalibr` 클론 + 빌드, 시간 꽤 걸림)
- **타깃**: `calibration/apriltag_grid.pdf` (AprilGrid 6×6) 를 **배율 100%** 로 인쇄해 판판한 판에 붙인다.
  **태그 한 변을 자로 재서** `calibration/target.yaml` 의 `tagSize` 를 그 값(m)으로. (oCamS 때 이 값이 틀려 baseline 이 25% 틀렸던 적 있음)

### 5-1. 녹화
```bash
UNO_Q=<IP> RAW=1 ./run_uvc_stereo.sh                 # 터미널 1
scripts/record_ov9281_calib_bag.sh ov9281_01         # 터미널 2, 1~2분 뒤 Ctrl-C
```
- 타깃이 **두 카메라 모두에** 보이게, **천천히**. 가까이·멀리, 화면 네 귀퉁이까지, 위아래·좌우로 30~45° 기울여서.
- **빠르게 움직이지 말 것** — 두 카메라가 동기가 아니라 움직이는 중의 쌍은 좌우가 다른 순간이다.

### 5-2. 변환·계산
```bash
scripts/convert_bag_to_ros1.sh calibration/bags/ov9281_01                         # → calibration/bags/ov9281_01.bag
FOCAL_GUESS=300 scripts/kalibr/calibrate_cameras.sh calibration/bags/ov9281_01.bag
```
- `FOCAL_GUESS=300`: OV9281(105°) 640×480 의 초점 추정값(oCamS 는 400). Kalibr 가 초점을 못 잡을 때만 쓰인다.
- 결과 `calibration/bags/ov9281_01-camchain.yaml` 과 보고서 pdf. **재투영 오차가 0.5 px 넘게 크거나 가장자리에서 크면**
  광각 모델로 다시: `MODELS="pinhole-equi pinhole-equi"` (단, 그 경우 6단계 변환기는 radtan 전용이라 손봐야 한다).

### 5-3. 드라이버·SLAM 이 읽는 형식으로
```bash
python3 scripts/kalibr/camchain_to_opencv_rectify.py calibration/bags/ov9281_01-camchain.yaml \
  ros2_ws/src/uvc_stereo/config/left_opencv.yaml ros2_ws/src/uvc_stereo/config/right_opencv.yaml
python3 scripts/kalibr/opencv_to_orbslam3_stereo.py ros2_ws/src/uvc_stereo/config \
  ros2_ws/src/uvc_stereo/config/orbslam3_stereo_ov9281.yaml
cd ros2_ws && colcon build --packages-select uvc_stereo && cd ..     # config 를 설치 폴더로
```
- 출력의 `rectified baseline` 이 **자로 잰 두 렌즈 중심 간격**과 1 cm 안쪽으로 맞는지 확인.

## 6. 평소 실행

```bash
UNO_Q=<IP> ./run_uvc_stereo.sh               # 정렬된 /pc/camera/{left,right}/compressed — run_cam_relay.sh 대신
export SCENE_CAM=ov9281                      # 파이썬 소비 코드가 OV9281 캘리브레이션을 읽게 (gaze_on_scene 등 실행 전에)
./run_gaze_on_scene.sh ...                   # 그대로
./run_slam_ov9281.sh                         # SLAM 이 필요하면 (스테레오 전용)
```

## 7. 카메라를 바꿨으니 다시 해야 하는 것

- **시선↔장면 캘리브레이션** (`calibration/gaze_scene_affine*.json`, `gaze_scene_extrinsic.json`): 장면 영상의 K·화각이
  바뀌었으니 무효. 기존 절차(docs/12)대로 다시.
- **깊이 baseline**: oCamS 는 SGBM 깊이에 실측 보정값 0.105 m 를 썼다. OV9281 은 아직 실측 전이라 기하 baseline 을 쓴다
  (`DEPTH_BASELINE_M` 환경변수로 덮어쓸 수 있음). 1 m·1.5 m 에 물체를 두고 `src/verify_depth_scale.py` 로 확인할 것.
- 태그 크기·AprilTag 앵커 등 카메라와 무관한 설정은 그대로.

## 파일

| 파일 | 하는 일 |
|---|---|
| `edge/uno_q/stereo_relay.py` | UNO Q: 카메라 목록(`--list`), 두 대 캡처·짝 맞춤·TCP 송신 |
| `edge/uno_q/stereo-relay.service` | UNO Q 부팅 자동 시작 (선택) |
| `ros2_ws/src/uvc_stereo/` | PC: `uvc_stereo_bridge` 노드, `config/`(캘리브레이션 결과가 들어갈 곳), `test/` |
| `run_uvc_stereo.sh` | PC 브리지 실행 (`RAW=1` 이면 캘리브레이션 녹화용 원본) |
| `scripts/record_ov9281_calib_bag.sh` | 캘리브레이션 녹화 (카메라 두 대만) |
| `scripts/kalibr/calibrate_cameras.sh` | `FOCAL_GUESS`·`MODELS`·`TOPICS` 환경변수 추가 (기본값은 oCamS 그대로) |
| `scripts/kalibr/camchain_to_opencv_rectify.py` | OpenCV 5 호환 수정(평행이동을 (3,1) 로) |
| `scripts/kalibr/opencv_to_orbslam3_stereo.py` | 정렬 결과 → ORB-SLAM3 스테레오 전용 설정 |
| `run_slam_ov9281.sh` | ORB-SLAM3 스테레오 전용 실행 |
| `src/ocams_calib.py` | `SCENE_CAM=ov9281` 전환 (기본 oCamS 그대로) |
| `scripts/move_docker_to_mnt_data.sh` | 도커 저장 위치를 /mnt/data 로 (Kalibr 이미지용) |

## 아직 확인 안 된 것 (카메라가 와야 알 수 있음)

- UNO Q 리눅스에서 OV9281(UVC)이 잡히는지, MJPG 640×480@30 이 두 대 동시에 나오는지, 허브 하나로 대역폭이 충분한지.
- UNO Q 의 OpenCV 가 V4L2 버퍼 시각(`CAP_PROP_POS_MSEC`)을 주는지 — 못 주면 읽은 시각으로 대신하고, 그만큼 어긋남 측정이 거칠어진다.
- 실제 좌우 어긋남 크기와 그것이 깊이·SLAM 에 주는 영향.
