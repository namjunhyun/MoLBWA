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

## 지금 바로 되는 것

```bash
conda activate lerobot
cd ~/molbwa_arm
python teach.py workspace --dry-run     # 작업공간 확인
python run_demo.py --dry-run --no-ros   # 로직만
```

`ultralytics`, `pupil-apriltags` 는 lazy import 라 팔/기구학 부분은 지금 그대로 돌아간다.
인식까지 쓰려면: `pip install ultralytics pupil-apriltags`

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
3. `calibration/apriltag_grid.pdf` 에서 태그 4개 잘라 팔 베이스 뒤 평판에 부착
   → 실측 위치를 `anchor.bundle` 에, 태그 한 변을 `tag_size_m` 에
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
