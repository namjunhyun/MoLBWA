# gaze_hri — 시선으로 로봇팔 제어하기 (MoLBWA 창종설)

MoLBWA 아이트래킹 안경이 만들어낸 **시선 3D 좌표**를 받아,
**SO-ARM101 로봇팔이 응시한 컵을 집어 응시한 자리로 옮기게** 하는 ROS2 스택입니다.

기존 저장소가 `p_W`(SLAM world 기준 시선점)까지 만들어 주면,
이 패키지가 그 다음 전부를 담당합니다.

---

## 1. 전체 구조

```
 [기존 MoLBWA 파이프라인]
  IR 눈 카메라 ─┐
               ├─ 융합 ──> p_W (시선 3D점, SLAM world)
  oCamS 스테레오┘           + T_WS (헤드 pose)
                              │  UDP JSON
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ [1] gaze_bridge         p_W를 ROS2 토픽으로 발행           │
 │        └─> /gaze/point_raw  (30Hz)                        │
 │                                                            │
 │ [2] dwell_detector      "일정 시간 한 곳을 보는가?"         │
 │        └─> /gaze/fixation   (I-DT 알고리즘)                │
 │                                                            │
 │ [3] target_resolver     "무엇을 / 어디에?"                  │
 │        └─> /gaze/target     (물체 스냅 + 테이블 평면 투영)  │
 │                                                            │
 │ [4] task_manager        상태 기계                          │
 │        IDLE → PICK_SELECTED → EXECUTING                    │
 │        └─> /pick_place 액션 목표                           │
 │                                                            │
 │ [5] arm_server          TF 변환 + 역기구학 + 서보 구동      │
 │        └─> SO-ARM101                                       │
 └──────────────────────────────────────────────────────────┘
                              ▲
                    TF: map ──┴── base_link
                    (캘리브레이션으로 구함, 아래 3장)
```

사용자 요청과의 대응 관계:

| 요청한 것 | 담당 노드 |
|---|---|
| 시선/물체/SLAM 좌표를 로봇팔에 **전달**하는 코드 | `gaze_bridge` + `arm_server`의 TF 변환 |
| 어느 물체를 보는지 좌표를 **구독**하는 코드 | `dwell_detector` → `target_resolver` (role=`pick`) |
| 어디로 옮길지 인식하는 좌표를 **구독**하는 코드 | `target_resolver` (role=`place`) + `task_manager` |
| 로봇 **동작** 코드 | `arm_server` (`/pick_place` 액션 서버) |

---

## 2. 설치

ROS2 Humble 또는 Jazzy (Ubuntu 22.04 / 24.04) 기준입니다.

```bash
mkdir -p ~/gaze_ws/src && cd ~/gaze_ws/src
# 이 폴더의 gaze_hri_msgs, gaze_hri 를 src/ 안에 복사

cd ~/gaze_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Feetech 서보를 직접 제어하려면:

```bash
pip install feetech-servo-sdk
sudo usermod -a -G dialout $USER   # 재로그인 필요
```

### 하드웨어 없이 먼저 돌려보기

**이걸 제일 먼저 하세요.** 안경도 로봇도 없이 전체 파이프라인이 돕니다.

```bash
ros2 launch gaze_hri gaze_hri.launch.py
```

`gaze_bridge`의 `fake` 모드가 "컵 A를 4초 응시 → 다른 곳으로 이동 →
놓을 자리를 4초 응시"를 반복 생성합니다.
터미널에 `응시 확정` → `물체 선택됨` → `[APPROACH_PICK]` ... 순서로 찍히면
로직이 정상입니다.

RViz로 보려면:
```bash
rviz2
# Fixed Frame: map
# Add > MarkerArray > /gaze/markers        (시선 커서, 응시하면 초록으로 차오름)
# Add > MarkerArray > /gaze/target_markers (집을 곳 주황 / 놓을 곳 초록)
```

---

## 3. 캘리브레이션 — 가장 중요한 단계

### 왜 필요한가

ORB-SLAM3의 world 원점은 **"SLAM을 시작한 순간의 카메라 위치"**입니다.
로봇 베이스와 아무 관계가 없습니다. `T_BW`를 모르면 `p_W`가 아무리 정확해도
로봇은 엉뚱한 곳으로 갑니다.

### 3-A. 테이블 평면 캘리브레이션 (먼저)

`config/gaze_hri.yaml`에서 `table_calibration: true`로 바꾸고 실행한 뒤,
테이블 위 **서로 다른** 지점 5~10군데를 차례로 응시하세요.
RANSAC 결과가 로그에 찍히면 그 값을 `table_plane:`에 복사하고
`table_calibration: false`로 되돌립니다.

### 3-B. world ↔ base_link 정합 (핵심)

```bash
# 터미널 1
ros2 launch gaze_hri gaze_hri.launch.py source:=udp backend:=feetech
# 터미널 2
ros2 run gaze_hri calibrate_world_to_base
```

로봇팔이 7개 자세를 차례로 잡습니다. 매번 **그리퍼 끝을 가만히 응시**하면 됩니다.

- FK로 `p_B`(로봇 기준 그리퍼 끝)를 정확히 압니다.
- 응시로 `p_W`(시선 기준 같은 지점)를 얻습니다.
- **Umeyama 정합**으로 `T_BW`를 한 번에 풉니다.
- `~/.ros/gaze_hri_calib.yaml`에 저장됩니다.

**RMSE가 20mm 이하면 합격입니다.** 50mm를 넘으면 링크 길이나 SLAM 스케일을 의심하세요.

이 방식의 장점은 SLAM 좌표계 오차뿐 아니라 **아이트래킹의 체계적 편향까지
같이 흡수**한다는 점입니다. 심사 때 설명 포인트로 쓰기 좋습니다.

> 대안(ArUco): 로봇 베이스에 마커를 붙이고 씬 카메라로 검출하면 더 빠릅니다.
> 다만 마커가 항상 보여야 하고 시선 편향은 보정되지 않습니다.
> 두 방법을 비교 실험해서 정확도 표를 만들면 보고서가 훨씬 좋아집니다.

### 3-C. 링크 길이 입력

`config/gaze_hri.yaml`의 `base_height / shoulder_offset / l1 / l2 / l3`는
**반드시 본인 SO-ARM101의 URDF 값으로 바꿔야 합니다.** 기본값은 추정치입니다.

```bash
# URDF에서 joint origin xyz 확인
ros2 param get /robot_state_publisher robot_description
# 또는 SO-ARM101 URDF 파일의 <joint><origin xyz="..."> 값을 직접 읽기
```

---

## 4. 기존 융합 코드를 붙이는 법

`gaze_bridge`가 다리 역할을 합니다. **UDP 방식을 권합니다.**
ORB-SLAM3 + pye3d + OpenCV 파이프라인은 자기만의 의존성 환경이 있어서
rclpy와 한 프로세스에 묶으면 의존성 충돌이 잦습니다.

기존 융합 루프 끝에 이 몇 줄만 추가하면 됩니다:

```python
import json, socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ... 기존 루프 안에서, gaze_point_world() 호출 직후 ...
p_W, origin, ray_dir = gaze_point_world(u, v, D, K, T_WS)

sock.sendto(json.dumps({
    "point": [float(p_W[0]), float(p_W[1]), float(p_W[2])],
    "valid": bool(pupil_detected and depth_ok),   # 깜빡임/추적실패 시 False
    "head": {
        "position": T_WS[:3, 3].tolist(),
        "orientation": rotation_matrix_to_quat(T_WS[:3, :3]),  # [x,y,z,w]
    },
}).encode(), ("127.0.0.1", 55055))
```

그리고 `source:=udp`로 실행합니다.

`valid` 플래그를 성실하게 채우는 게 중요합니다. 깜빡일 때 좌표가 튀면
dwell 판정이 계속 깨집니다.

---

## 5. 물체 인식을 붙이면 좋아지는 이유 (선택이지만 강력 추천)

사람은 컵을 볼 때 컵의 **표면**을 봅니다. 그 점을 그대로 파지점으로 쓰면
그리퍼가 컵 옆구리를 스치거나 테두리를 칩니다.

`target_resolver`는 `/objects/poses` (`geometry_msgs/PoseArray`)를 구독합니다.
여기에 검출된 물체 **중심** 좌표를 넣어주면, 응시점을 `snap_radius`(기본 8cm) 안의
가장 가까운 물체 중심으로 끌어당깁니다.

> **물체 위치를 어디서 얻을지 두 가지 길이 있고, 성질이 다릅니다.**
>
> | | 탑다운 카메라 + 호모그래피 | 씬카메라 + YOLO + 스테레오 깊이 |
> |---|---|---|
> | 좌표계 | `base_link` (로봇 기준) | `map` (SLAM world) |
> | 정확도 | 캘리브가 맞으면 mm 급 | 시선과 **같은 오차 경로**를 탐 |
> | 추가 하드웨어 | 탑다운 카메라 필요 | 없음 |
> | 깊이 의존 | 없음 | 있음 (`docs/12` 기준 아직 제약) |
>
> **탑다운을 쓰면** 시선은 "어느 컵인지"만 고르고 정확한 위치는 탑다운이 담당하므로,
> 시선 오차(수 cm)가 파지 정확도로 넘어가지 않습니다. 팀원_설명서 8장의 설명이
> 성립하는 것은 이 구성일 때입니다.
>
> **씬카메라만 쓰면** 물체 위치도 시선과 같은 경로(깊이+SLAM)를 타므로 그 장점이
> 사라집니다. 어느 쪽을 쓰든 `frame_id`만 정확히 채우면 `target_resolver`가
> `base_link`로 변환해 처리합니다. 자세한 논의는 `docs/13_gaze_to_arm.md`.

가장 간단한 구현 (YOLO + 스테레오 깊이):

```python
# 씬 카메라 프레임마다
boxes = yolo(scene_rgb)                      # class == "cup"
for box in boxes:
    u, v = box.center
    D = median_depth_around(disparity, u, v, window=9)
    p_S = D * (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    p_W = (T_WS @ np.append(p_S, 1.0))[:3]
    poses.append(p_W)
# PoseArray로 /objects/poses 발행
```

컵 2개 구분이 훨씬 안정적으로 되고, "어떤 물체를 보는지 특정한다"는
기획서 문구도 실제로 증명됩니다.

---

## 6. 실행 (실제 하드웨어)

```bash
# 1) 융합 파이프라인 (기존 저장소)
python src/fusion_main.py --udp 127.0.0.1:55055

# 2) ROS2 스택 — 처음엔 반드시 dry_run으로!
ros2 launch gaze_hri gaze_hri.launch.py source:=udp backend:=dry_run

# 3) 좌표가 맞는 걸 확인한 뒤에만 실제 서보로
ros2 launch gaze_hri gaze_hri.launch.py source:=udp backend:=feetech
```

사용 흐름:
1. 컵을 1.2초 응시 → `물체 선택됨` 로그, 주황 마커
2. 시선을 옮겨 놓을 자리를 1.2초 응시 → 로봇 동작 시작
3. 잘못 선택했으면 그냥 15초 기다리거나 `ros2 topic pub --once /task/cancel std_msgs/Empty {}`

---

## 7. 안전장치

코드에 들어 있는 것:

- **미다스의 손 문제 방지**: dwell 1.2초 + 산포 3cm + 쿨다운 1.5초
- **재선택 잠금**: 한 번 선택하면 10cm 밖으로 시선이 나가야 다음 선택 가능
- **pick/place 최소 거리** 10cm — 같은 곳 응시로 동시 발동 방지
- **선택 타임아웃** 15초 자동 취소
- **작업 공간 검사**: 반경 0.42m, z ≥ -0.02m 밖이면 거부
- **관절 한계 검사** + 접근각 자동 대체 (45° 실패 시 60°, 수직 순으로 재시도)
- **코사인 이징 보간**: 저가 서보에서 급가속 진동 억제

코드로 못 막는 것:

> **물리 비상정지를 반드시 두세요.** 전원 스위치를 손 닿는 곳에 두고,
> 첫 실험은 반드시 팔을 손으로 잡을 수 있는 자세에서 하세요.
> 시선 인터페이스는 "취소하려고 쳐다보는 것"조차 입력이 되는 구조라
> 소프트웨어 취소만으로는 부족합니다.

---

## 8. 튜닝 순서 (이 순서대로 하면 시간이 절약됩니다)

1. `fake` 모드로 전체 로직 확인 — 하드웨어 0개
2. 실제 시선 + `dry_run` — **좌표가 맞는지만** 확인. `/gaze/target`을 `ros2 topic echo`로 보면서 자로 재보기
3. `dwell_time`, `dispersion_radius` 튜닝 — 팀원 여러 명에게 시켜보기
4. 테이블 평면 + world↔base 캘리브레이션
5. `grasp_height`를 실제 컵으로 재기 (컵 바꾸면 반드시 다시)
6. `backend: feetech`, 속도 낮춰서 첫 동작
7. `approach_pitch` 조정 — 컵 손잡이 방향에 따라 45°가 안 되면 60°

---

## 9. 자주 겪는 문제

| 증상 | 원인 / 해결 |
|---|---|
| 로봇이 전혀 다른 방향으로 감 | `T_BW` 캘리브레이션 미실행 또는 RMSE 큼. `ros2 run tf2_ros tf2_echo map base_link`로 TF 확인 |
| 로봇이 맞는 방향인데 거리가 안 맞음 | SLAM 스케일 문제. 모노큘러로 돌고 있지 않은지, IMU가 초기화됐는지 확인 |
| dwell이 절대 안 잡힘 | `dispersion_radius`를 5cm로 올려보기. 깊이 `D`가 튀면 시선점이 광선 방향으로 크게 흔들림 → 시선 주변 윈도우의 **중앙값** 깊이 사용 |
| dwell이 너무 자주 잡힘 | `dwell_time` 증가, `cooldown` 증가 |
| 그리퍼가 컵을 스침 | `grasp_height` 재측정, `/objects/poses` 물체 스냅 도입 |
| `역기구학 실패: 작업 공간 밖` | 컵을 로봇 쪽으로 당기기. 링크 길이 값 확인. 베이스 바로 앞(반경 15cm 이내)은 이 팔 구조상 도달 불가 |
| 서보가 반대로 돎 | `servo_signs`의 해당 항목을 -1로 |
| 액션 서버 없음 | `arm_server` 노드가 죽었는지 확인. `ros2 node list` |

---

## 10. 보고서에 쓸 수 있는 평가 지표

심사에서 "그래서 얼마나 정확한데?"는 반드시 나옵니다. 미리 재두세요.

- **시선 3D점 정확도**: 알려진 위치의 마커를 N회 응시, 실제 좌표와의 RMSE
- **좌표계 정합 RMSE**: 캘리브레이션 도구가 자동 출력 (목표 < 20mm)
- **파지 성공률**: 컵 위치를 바꿔가며 30회 시도 (목표 > 80%)
- **선택 정확도**: 컵 2개 중 의도한 것을 고른 비율
- **오작동률**: 의도하지 않은 dwell 발동 횟수 / 분
- **선택 시간**: 응시 시작부터 로봇 동작 시작까지

`dwell_time`을 0.8 / 1.2 / 1.6초로 바꿔가며 "오작동률 vs 선택 시간"
트레이드오프 곡선을 그리면 보고서 품질이 확 올라갑니다.

---

## 11. MoveIt2를 쓰고 싶다면

충돌 회피와 경로 계획이 필요해지면:

```bash
git clone https://github.com/JafarAbdi/ros2_so_arm100.git
git clone https://github.com/JafarAbdi/feetech_ros2_driver.git
```
(`so_arm101_description` 패키지가 포함되어 있습니다)

그 다음 `arm_server`의 `backend: "joint"`로 바꾸고
`JointTrajectoryBackend`의 토픽을 컨트롤러에 맞추면 됩니다.

다만 책상 위 컵 옮기기 정도면 지금의 직선 보간으로 충분하고,
**대회 시연 때 훨씬 덜 깨집니다.** 시연 안정성이 우선이라면 그대로 두세요.
