# molbwa_arm — MoLBWA 시선 → SO-ARM101 음용 보조

시선으로 컵 3개 중 하나를 고르면 팔이 집어서 입에 대고 기울여 준다.
SLAM(ORB-SLAM3)이 헤드캠 자세를 계속 주고, AprilTag 번들이 팔 좌표계를 world 에 latch 한다.

```
T_hc_ab = inv(T_w_hc) @ T_w_ab
          ^SLAM          ^AprilTag latch (한 번 등록, 보일 때마다 갱신)
```

## 파일

| 파일 | 역할 |
|---|---|
| `kinematics.py` | SO-101 5-DOF 정/역기구학. FK 왕복 + 관절한계 검증 포함 |
| `arm.py`        | 속도제한 이동, 그리퍼, 기울이기. `dry_run=True` 로 하드웨어 없이 테스트 |
| `anchor.py`     | AprilTag 번들 PnP + SLAM latch + ANCHORED/STALE/UNANCHORED 상태머신 |
| `perception.py` | YOLOv8-seg 컵 검출 + dwell 기반 시선 선택 + depth 백프로젝션 |
| `task.py`       | 접근→파지→들기→입→기울이기→복귀 시퀀스 + abort |
| `run_demo.py`   | 메인 루프. **MoLBWA 통합 지점은 `GazeSource.frame()` 하나뿐** |
| `teach.py`      | 관절 부호 확인 / 입 위치 티칭 / 작업공간 확인 / 그리퍼 값 탐색 |

## 지금 바로 되는 것 (하드웨어 0개, 서드파티 0개)

`numpy scipy opencv-python pyyaml` 만 있으면 된다. ultralytics / pupil-apriltags /
rclpy / lerobot 없이도 아래 셋은 오늘 그대로 돈다.

```bash
python test_pipeline.py                 # 검증 스위트 13개 (좌표계 사슬 전체)
python run_demo.py --sim                # 합성 장면으로 시선->앵커->파지 전 구간 1회 완주
python teach.py workspace --dry-run     # 작업공간 표
python run_demo.py --dry-run --no-ros   # 실입력 대기 상태로 기동 (검출기는 있으면 켜짐)
```

`--sim` 은 컵의 armbase 진값을 알고 시작해서, 카메라 픽셀로 투영했다가 파이프라인이
되짚어 복원한 값과 비교한다. 좌표계가 한 군데라도 틀리면 여기서 mm 단위로 드러난다.

인식/실기까지 쓰려면: `pip install ultralytics pupil-apriltags` + lerobot 설치 + ROS2.

## 2026-08-19 에 고친 것 — 실기 전에 반드시 읽을 것

| # | 결함 | 증상 | 조치 |
|---|---|---|---|
| 1 | `src/ocams_calib.py` 가 `left.yaml`(2026-07-14 구버전) 을 읽고 있었다 | rectified K 가 fx=433.55/cx=470.35. ORB-SLAM3 는 484.71/326.84 로 pose 를 만들고 있었으므로 **pose 좌표계와 역투영 좌표계가 서로 달랐다**. 0.7m 거리 화면 중앙 컵 기준 횡방향 **약 23cm** 오차 | `*_opencv.yaml`(commit 23ff90e) 로 교체 + import 시점에 ORB-SLAM3 설정과 자동 대조 |
| 2 | `run_demo.py` 에 같은 낡은 K 가 하드코딩 | 위와 동일. 태그 PnP 에도 같이 들어가 앵커까지 틀어짐 | 하드코딩 삭제, `ocams_calib` 단일 출처 |
| 3 | 태그 코너 축이 config 주석과 **반대(거울상)** | 코드는 법선 −x, 문서는 +x. det=−1 이라 회전으로 표현 불가 → PnP 가 안 풀림 (합성 검증 재투영 오차 2086px) | `anchor.tag_axes` 로 명시 + 재투영 오차로 자동 기각 |
| 4 | 번들 4장이 전부 동일평면 | 태그판 기준 **거울 반사된 자세**가 픽셀상 똑같이 투영됨. 재투영 오차 0.000px 인데 팔 베이스가 **120mm**(= 2 × 판 깊이) 틀린 해가 나왔다 | 기본 배치를 계단형(비평면)으로 변경 + `solvePnPGeneric` 다중해/cheirality 검사 |
| 5 | 팔 동작 중 안전 게이트가 죽어 있었다 | `task.run()` 이 블로킹인데 그 안에서 SLAM 을 안 읽어서, 30초 시퀀스 내내 멈춘 옛 pose 를 검사 | rclpy spin 을 별도 스레드로 + `DrinkTask(refresh=...)` 훅 + POUR 단계 게이트 추가 |
| 6 | dwell 후보를 화면 좌→우 인덱스로 추적 | 컵 하나가 한 프레임 빠지면 인덱스가 통째로 밀림 | 화면 **위치**로 추적하고, 인덱스가 아니라 **Cup 객체**를 반환 |
| 7 | `--dry-run --no-ros` 가 실제로는 안 돌았다 | `pupil_apriltags` 미설치 시 즉시 ModuleNotFoundError | 검출기를 선택적으로 생성 |

## depth 없이도 컵 위치가 나온다 — 테이블 평면 교차

StereoSGBM 은 흰 종이컵처럼 텍스처 없는 물체에서 **정확히 그 물체 위에만** 구멍이 뚫린다
(docs/11 기준 유효픽셀 32%). 대신 "컵은 테이블 위에 있다"는 사전정보를 쓴다:
컵-테이블 접점 픽셀에서 광선을 쏴 알려진 테이블 평면과 교차시키면 depth 가 아예 필요 없다.

`perception.cup_position_on_table()`. `--sim` 에서 컵 3개 복원오차 0.0000mm.
depth 맵이 들어오면 그쪽을 우선 쓰고, 없을 때만 이 경로로 떨어진다.

## ★ 하드웨어 배치 — 받침대 없으면 데모가 성립하지 않는다

기구학을 풀어본 결과다. **컵을 세운 채(top-down 파지) 들 수 있는 최대 높이가
팔 베이스 기준 12cm**다. 팔을 테이블에 직접 놓으면 사람 입(테이블 위 ~22cm)에
닿으려면 컵을 46° 기울여야 하고, 가는 도중에 다 쏟는다.

**팔 베이스 아래 12~14cm 받침대를 깔아라.** config 의 `arm.base_height_m`.
14cm 기준 도달 범위(반경 r, 팔 중심에서):

| 절대높이 | 도달 반경 |
|---|---|
| 0.00 m (테이블면) | 0.125 ~ 0.235 m |
| 0.22 m (입 높이)  | 0.095 ~ 0.205 m |
| 0.28 m            | 도달 불가 |

컵 3개도 사람 입도 이 안에 들어와야 한다. `teach.py workspace` 로 확인.

## ★ 기울이기는 wrist_roll 이 아니라 툴 피치다

top-down 파지에서 `wrist_roll` 은 **수직축** 회전이라 아무리 돌려도 컵이 안 기운다.
`arm.tilt_in_place()` 가 TCP 위치를 고정한 채 피치만 -90° → -70° 로 바꾼다.

## ★ ORB-SLAM3 에 map_id 퍼블리시를 추가해야 한다

트래킹을 놓치면 ORB-SLAM3 는 Atlas 에 **새 맵**을 만들고, 그 순간 world 원점이
재정의된다. latch 해둔 `T_w_ab` 는 부정확해지는 게 아니라 **무효**가 된다.
loop closure 를 꺼도 이건 못 막는다.

`patches/orbslam3_ros2.patch` 에서 pose 퍼블리시를 뚫은 자리에 추가:

```cpp
// 현재 Atlas map id 를 /orbslam3/map_id (std_msgs/Int32) 로 퍼블리시
int map_id = pSLAM->GetAtlas()->GetCurrentMap()->GetId();   // API 명은 버전 확인
```

`anchor.py` 가 map_id 변화를 보고 latch 를 무효화하고, `task.py` 가 **매 동작 직전**에
게이트를 다시 확인한다. UNANCHORED/STALE 이면 팔은 움직이지 않고
"팔 쪽 태그를 봐 주세요" 를 띄운다.

## anchor drift — 보고서에 쓸 평가 지표

태그가 보이는 동안에는 두 경로로 `T_hc_ab` 를 동시에 얻을 수 있다.

```
A: 태그 직접 검출                       <- 기준
B: inv(T_w_hc) @ T_w_ab (latch 경유)    <- SLAM 경유
```

`AnchorTracker.drift_history` 에 `(t, ‖A-B‖)` 가 쌓인다. 이게 **SLAM anchor drift**고,
cm/min 으로 뽑으면 "SLAM 썼습니다"보다 훨씬 강한 정량 결과가 된다.
`drift_warn_m`(2cm) / `drift_max_m`(5cm) 임계로 안전 정책까지 연결된다.

## 셋업 순서

1. `pip install ultralytics pupil-apriltags`
2. 받침대 12~14cm 제작, `base_height_m` 기입
3. 태그 **5~6cm** 짜리 4~5장을 팔 베이스 뒤에 **비평면(계단형)** 으로 부착
   → 실측 위치를 `anchor.bundle` 에, 태그 한 변을 `tag_size_m` 에
   - 3cm 는 1m 거리에서 14px 밖에 안 된다 (6cm 면 29px). 코너 잡기가 불안정하다.
   - 전부 한 평면에 붙이면 거울 해 때문에 12cm 틀린 자세가 재투영 0px 로 통과한다.
     깊이가 다른 두 단으로 나누면 사라진다.
   - **태그는 팔에만 붙인다. 컵에는 붙이지 않는다** — 컵은 YOLO 가 보고, 3D 위치는
     테이블 평면 교차로 나온다. 미지수는 컵이 아니라 팔 베이스 자세다.
   - 그리퍼에 태그 1장을 더 붙이면 (a) 번들 좌표를 자로 재는 대신 FK 와 짝지어
     자동 산출(핸드아이), (b) 동작 중 예측 TCP vs 관측 TCP 로 end-to-end 오차 실측이 된다.
4. 링크 길이 L1~L4 실측 → `arm.links`
5. `teach.py signs` → `joint_sign` 확정 → `home_deg` 재계산
6. `teach.py gripper` → `gripper.open/closed` 확정
7. `teach.py pose` → 입 위치 티칭 → `task.mouth_pos`
8. `teach.py workspace` → 컵 배치 확정
9. ORB-SLAM3 map_id 퍼블리시 패치
10. `GazeSource.frame()` 에 MoLBWA 시선/depth 연결
11. `run_demo.py --no-arm` 으로 컵 선택까지 확인 → 전체 실행

## 안전

- 물로 먼저. 종이컵 50~80ml. SO-101 은 뻗은 자세에서 페이로드 여유가 거의 없다.
- 컵을 든 뒤 모든 이동은 `slow_joint_speed_deg_s`(12 deg/s).
- 기울임 20°, 6 deg/s 상한.
- `task.plan()` 이 실행 전에 전 구간 IK 를 검사한다. 하나라도 불가면 아예 시작하지 않는다.
- `abort()` 는 컵을 먼저 **세운 뒤** 테이블에 내려놓고 home 으로 간다.
