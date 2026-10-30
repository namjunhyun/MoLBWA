# 2026-08-19 — SO-ARM101 브랜치 결함 수리 + 하드웨어 없는 검증 스위트

`feat/so101-arm` 을 실기에 붙이기 전에 코드를 정독하고, 좌표계 사슬을 합성 데이터로
되짚어 검증했다. 결함 7개를 찾아 고쳤고, 그중 3개는 **실기에서 팔이 엉뚱한 데로 가는**
종류였다.

## ① rectified K 가 통째로 틀려 있었다 (제일 큼)

`src/ocams_calib.py` 가 `ros2_ws/src/ocams_ros2/config/left.yaml` 을 읽고 있었다.
그런데 2026-08-11 재캘리브레이션(commit 23ff90e)은 `left_opencv.yaml` / `right_opencv.yaml`
쪽을 교체했다. `left.yaml` 은 2026-07-14 의 순환 캘리브레이션 결과 그대로였다.

| | fx | cx | cy | baseline |
|---|---:|---:|---:|---:|
| `left.yaml` (쓰이고 있던 값) | 433.553 | **470.352** | 236.489 | 12.48 cm |
| `left_opencv.yaml` (정본) | 484.711 | **326.836** | 240.154 | 10.67 cm |
| `orbslam3_stereo_inertial_oCamS.yaml` (SLAM 이 실제로 쓰는 값) | 484.711 | 326.836 | 240.154 | 10.67 cm |

cx 가 **143.5px** 차이난다. 640폭 영상에서 470 은 애초에 광학 중심일 수 없다.
ORB-SLAM3 는 484.71/326.84 로 pose 를 만들고 있었으므로 **pose 의 좌표계와 역투영의
좌표계가 서로 다른 카메라였다.** 0.7m 거리의 화면 중앙 컵을 역투영하면

- 정본: x = −0.0099 m
- 쓰이던 값: x = −0.2427 m
- → 횡방향 **약 23 cm** 오차

영향 범위는 `arm/` 뿐이 아니다. `src/gaze_on_scene.py`, `src/fusion.py` 도 같은
`RECTIFIED_K` 를 쓰므로 시선 3D 점도 같이 틀어져 있었다.

**조치**: `*_opencv.yaml` 을 읽도록 교체(OpenCV FileStorage 포맷이라 `yaml.safe_load` 로는
못 읽는다 → `cv2.FileStorage`). 그리고 import 시점에 ORB-SLAM3 설정과 자동 대조하는
`check_consistency()` 를 넣었다. 다시 어긋나면 조용히 틀리는 대신 예외로 죽는다.

`arm/run_demo.py` 에 같은 낡은 값이 하드코딩돼 있던 것도 삭제하고 `ocams_calib` 단일
출처로 바꿨다.

## ② AprilTag 코너 축이 문서와 반대였다

`anchor.py` 의 로컬 코너가 `[(+y,−z), (−y,−z), (−y,+z), (+y,+z)]` 였다.
right = −y, up = +z 이므로 법선(right × up) = **−x**. 그런데 config 주석은
"태그 평면 법선은 +x(팔 정면) 가정" 이었다. 즉 y 축이 뒤집힌 **거울상** 배치를
PnP 에 넣고 있었다 (det = −1, 회전으로 표현 불가능).

합성 검증에서 예전 규약의 재투영 오차는 **2086 px**.

**조치**: `anchor.tag_axes` (태그별 override 가능) 로 명시하고, 재투영 오차가
`max_reproj_px`(3px) 를 넘으면 그 관측을 버린다.

## ③ 동일평면 번들의 거울 해 — 재투영 0px 인데 12cm 틀림

config 기본 번들 4장이 전부 `x = −0.060` 한 평면이었다. 평면 타깃은 태그판을 기준으로
**거울 반사된 자세**가 픽셀상 똑같이 투영된다. 합성 검증에서 재투영 오차 **0.000 px**
인데 팔 베이스 위치가 **정확히 120 mm**(= 2 × 판 깊이) 틀린 해가 나왔다.
재투영 오차 검사로는 절대 못 잡는다.

추가로 `solvePnP` 가 좌표 전체의 부호를 뒤집은 해(태그가 카메라 **뒤**에 있는 해)를
반환하는 것도 확인했다 — 투영은 (x,y,z) → (−x,−y,−z) 에 불변이라 재투영 오차가 0 이다.

**조치**:
- 기본 번들을 계단형(x = −0.060 두 장 + x = −0.010 두 장)으로 변경, 태그 크기 3cm → 6cm.
- `solvePnPGeneric` 으로 후보해를 전부 받아 (a) cheirality 위반 해 제거,
  (b) 남은 해들이 `ambiguity_max_m`(2cm) 넘게 벌어지면 관측 자체를 버린다.
- 번들이 거의 동일평면이면 경고를 띄운다.

## ④ 팔이 움직이는 동안 안전 게이트가 죽어 있었다

`task.py` 는 "매 동작 직전에 anchor 게이트를 다시 확인한다"고 선언해놓고, `run()` 이
블로킹인 채로 그 안에서 SLAM 을 한 번도 안 읽었다. 30초짜리 시퀀스 내내 **멈춰 있는
옛 pose** 를 검사했다. 액체를 얼굴로 옮기는 동작에서 이건 안전 장치가 아니라 장식이다.

**조치**: rclpy 스핀을 별도 스레드로 분리하고, `DrinkTask(refresh=...)` 훅을 넣어 매
게이트마다 최신 pose 를 끌어오게 했다. `tracking_ok` 도 "최근 0.5초 안에 pose 가 왔는가"
로 실제 판정한다 (ORB_SLAM3_ROS2 는 트래킹을 놓치면 발행을 멈춘다).
게이트가 없던 POUR(얼굴 바로 앞에서 기울이는) 단계에도 추가했다.

## ⑤ dwell 후보를 화면 인덱스로 추적하고 있었다

`perception.py` 가 매 프레임 컵을 화면 x 로 정렬해 0/1/2 를 새로 매기는데, dwell 후보를
그 인덱스로 들고 있었다. 컵 하나가 한 프레임 검출에서 빠지면 나머지 인덱스가 통째로
밀린다. `GazeDwell.update()` 가 sticky 한 인덱스를 반환하는 구조라 다른 컵을 잡을 여지도
있었다.

**조치**: 후보를 화면 **위치**로 추적하고(`track_radius_px`), 인덱스가 아니라 **그 프레임의
Cup 객체**를 반환한다. 호출자가 인덱스를 다시 조회할 일이 없어 이 종류의 사고가
구조적으로 불가능해졌다.

## ⑥ depth 없이 컵 위치 구하기 — 테이블 평면 교차

`docs/09 §7` 이 "스테레오 깊이 D 가 현재 병목"이라고 해놨는데, 대상이 **흰 종이컵**이라
StereoSGBM 은 정확히 그 컵 위에만 구멍이 뚫린다 (docs/11 기준 유효픽셀 32%).
`perception.py:41` 의 "흰 종이컵은 구멍이 뚫리므로 median 필수" 주석이 이미 그걸
예상했지만, median 으로 메울 유효값 자체가 안 나온다.

대신 "컵은 테이블 위에 있다"는 사전정보를 쓴다. 컵-테이블 접점 픽셀(마스크 최하단
중심)에서 광선을 쏴, 앵커로 알고 있는 테이블 평면과 교차시킨다.
`perception.cup_position_on_table()`. 합성 검증에서 컵 3개 복원오차 0.0000 mm.

## ⑦ `--dry-run --no-ros` 는 원래부터 안 돌았다

README 가 "지금 바로 되는 것"으로 적어둔 명령인데 `pupil_apriltags` 미설치 시 즉시
`ModuleNotFoundError` 로 죽었다. 검출기를 선택적으로 생성하도록 바꿨다.

## 검증 스위트 — `arm/test_pipeline.py`

`numpy scipy opencv-python pyyaml` 만으로 도는 13개 검사.
ultralytics / pupil_apriltags / rclpy / lerobot 전부 불필요.

```
13/13 PASS
  intrinsics: ORB-SLAM3 설정과 일치 — fx=484.71 cx=326.84 최대차 0.000px
  intrinsics: cx 가 화면 중앙 부근 — cx=326.84
  intrinsics: baseline 이 물리적으로 그럴듯 — 10.67cm
  kinematics: FK/IK 왕복 — 440/500 성공, 최대오차 1.15e-16 m
  anchor: 좌표변환 왕복 — 오차 1.62e-16 m
  anchor: latch 후 ANCHORED
  anchor: map_id 바뀌면 latch 무효
  태그: 법선이 팔 정면(+x)
  태그: 새 코너 규약으로 PnP 복원 — 재투영 0.000px, 위치오차 0.000mm
  태그: 예전 거울상 규약은 재투영 오차로 걸러짐 — 재투영 2086.24px
  테이블 평면 교차: depth 없이 컵 3D 복원 — 최대오차 0.0000 mm
  dwell: 검출 누락으로 인덱스가 밀려도 같은 컵 유지
  dwell: 컵을 바꾸면 처음부터 다시 센다
```

`arm/sim_source.py` 는 컵의 armbase 진값을 알고 시작해서 카메라 픽셀로 투영했다가,
파이프라인이 되짚어 복원한 값과 비교한다. `python run_demo.py --sim` 으로 시선 →
앵커 → 파지 → 입 전달 → 복귀 전 구간이 하드웨어 없이 1회 완주한다.

> 이 스위트를 짜다가 시뮬레이터 자체의 카메라 축 행렬이 det = −1(반사)인 것도
> 잡았다. `cv2.Rodrigues` 는 회전이 아닌 행렬을 넣어도 에러를 안 내고 엉뚱한 값을 뱉는다.

## 아직 안 된 것 (실기 전 남은 일)

- [ ] `GazeSource.frame()` 에 `src/gaze_on_scene.py` 연결 — **유일한 미구현 통합 지점**
- [ ] `/orbslam3/map_id` 퍼블리시 패치 (없으면 Atlas 맵 전환 감지가 영원히 작동 안 함)
- [ ] 받침대 12~14cm 제작 → `arm.base_height_m`
- [ ] 태그 6cm × 4~5장 비평면 부착 → `anchor.bundle` 실측
- [ ] 그리퍼 태그 + 핸드아이로 번들 좌표 자동 산출 (자로 재지 말 것)
- [ ] `pip install ultralytics pupil-apriltags` + lerobot 설치
- [ ] `teach.py signs / gripper / pose`

## 태그를 어디에 붙이는가 — 결론

**팔에만. 컵에는 안 붙인다.**

- 컵은 카메라가 이미 볼 수 있다 (YOLO-seg + 테이블 평면 교차). 컵에 태그를 붙이면
  "시선으로 임의 물체를 고른다"는 프로젝트 본질이 "마커 붙은 물체만 집는 데모"가 된다.
- 모르는 값은 컵이 아니라 **팔 베이스 자세**다. SLAM 의 world 원점은 초기화 순간의
  카메라 자세로 임의로 잡히므로, 팔이 그 원점 기준 어디 있는지 알 방법이 원리적으로 없다.
- 컵에 붙인 태그는 그리퍼가 잡는 순간 가려지고, 기울이면 같이 돈다.
- 팔 베이스는 안 움직인다 → latch 가 성립한다.

SLAM 의 역할도 다시 정의해둔다. 태그가 보이는 동안에는 태그가 정확도의 **주 공급원**이고,
SLAM 은 (a) 태그가 가려진 구간을 잇는 브릿지, (b) `AnchorTracker.drift_history` 로
측정되는 **anchor drift (cm/min)** 의 대상이다. 데모 성공률을 ORB-SLAM3 안정성과
분리해야 한다 — docs/11 기준 SLAM 은 아직 VI 초기화 1단계 도달, 장시간 안정성 미검증이다.
