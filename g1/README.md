# g1 — 데모 2: 시선으로 G1 휴머노이드와 교감하기

사용자가 G1 의 **얼굴 / 손 / 몸통** 중 한 곳을 3초 응시하면, G1 이 **서 있는 채로** 사용자 쪽으로
전신 제자리 회전(45° 단위)으로 돌아 마주본 뒤 손 흔들기(`wave`)를 하고
(악수 자세 `handshake` / 팔 벌리기 `open_arms` 는 보류 — 클립이 생기면 그 이름 그대로 추가)
대기로 돌아온다.
데모 1(로봇팔)과 같은 시선 인터페이스(글래스 → UDP 55056)를 쓴다.

설계: `docs/superpowers/specs/2026-10-08-g1-emotional-demo-design.md` ·
계획: `docs/superpowers/plans/2026-10-08-g1-emotional-demo.md` · 실행 절차: [`RUNBOOK.md`](RUNBOOK.md)

진단·치료 효과는 주장하지 않는다.

## 전체 구조

```
[글래스]  gaze_on_scene.py --send-gaze-px ─UDP 55056─┐
          씬 영상 /pc/camera/left/compressed ────────┤
                                                      ▼
[데스크탑] g1/g1_gaze_bridge.py
           ├ TagBundleDetector(G1 몸통 4면 태그) → 사용자 머리의 G1 기준 방위각 bearing_deg
           └ yolo_worker(g1_face/g1_hand/g1_torso) → 시선 픽셀이 속한 부위 label
           ─UDP 55057 JSON {t, label, bearing_deg, valid}─▶
          gaze_hri/g1_interaction_node (ROS 2)
           ├ dwell: 같은 label 3.0 초 유지(유효 70%) → 확정
           ├ bearing_deg = int(round(dwell 창 안 마지막 유효 bearing)) 또는 null
           ├ 상태: IDLE → BUSY(Jetson 보고 기준) → IDLE, BUSY 중 이벤트 무시
           ├ 발행 /g1/event, /g1/state, /g1/dwell_progress — rosbag 으로 기록
           └─UDP 55070 {seq, cmd:"act", part, bearing_deg} / {cmd:"ping"} ─▶
[Jetson]  g1/deploy/g1_motion_server.py (50 Hz 온보드, numpy only)
           ├ 정책 1개 + 모션 라이브러리 npz(세그먼트 경계 library_meta.json)
           ├ idle 반복 → 명령 수신 시 plan(part, bearing) = [turn_l*/turn_r*?] + 동작 → idle
           ├ 세그먼트 시작마다 yaw_off 재정렬
           └─UDP 55071 {state, seq} ─▶ 데스크탑
```

**정책→모터 경로는 로봇 내부(Jetson 온보드)만 지난다.** 와이파이는 명령·상태·로그에만 쓴다.
데스크탑에서 정책을 돌리고 와이파이로 `lowcmd` 를 쏘는 것은 금지다(5GHz 부하 시 p99 170~215 ms,
지연 여유 60 ms 의 3~4배 — `~/g1_dance_deploy/POLICY_DEPLOY_PLAYBOOK.md` 12장).

## 규약 (코드 전체가 이걸 공유한다 — `g1_protocol.py`)

| 항목 | 값 |
|---|---|
| UDP 포트 | 브리지→노드 **55057**, 노드→Jetson **55070**, Jetson→노드 **55071** (기존 55055/55056/55059 는 안 건드림) |
| 부위 → 동작 | `g1_face`→`wave`, `g1_hand`→`handshake`, `g1_torso`→`open_arms`. 겹치면 hand > face > torso |
| 방향 맞추기 `plan()` | bearing b(도, **+ 가 왼쪽**, (−180,180] 로 감음)를 원형 거리로 가장 가까운 bin {0, ±45, ±90, ±135, 180} 로 양자화(±180 부근은 180, 잔차 ≤ ±22.5°). bin ≠ 0 → 전신 제자리 회전 `turn_l{b}`/`turn_r{−b}` 뒤 동작, 0 이나 null → 동작만. 동작은 사용자를 마주본 채 **허리 변형 없이** 재생 |
| 세그먼트 | `idle`, `turn_l45/90/135/180`, `turn_r45/90/135`, `wave` — 9개. 회전 = Unitree 사전학습 보행 정책(unitree_rl_gym `motion.pt`)을 MuJoCo 에서 돌려 녹화(`~/molbwa_g1/record_loco_turn.py`: yaw P 제어, wz ≤ 0.6 rad/s, 두 발 지지 위상에서 끝, 앞뒤 1.5 s 정지 패딩). wave = AMASS_Retargeted_for_G1 BMLmovi 48_F_2(**연구용 라이선스**). 라이브러리는 blend 1.5 s + `--seam_joint 0.2`(보행 자체가 ~0.18 rad/프레임 — 원본 클립 최대값과 같음 확인) |
| 모션 csv | 36열 = root pos 3 + root quat **xyzw** 4 + 관절 29, 30 fps (`csv_to_npz.py` 가 wxyz·50 fps 로 바꾼다) |
| 정책 관측 | **154차원**(Wo-State-Estimation)만. 160차원 정책은 실행 거부 |
| 메시지 | bridge `{t, label, bearing_deg, valid}` · cmd `{seq, cmd: act\|ping, part, bearing_deg: int\|null}` · state `{state: idle\|turning\|acting, seq}` |

## 파일

| 파일 | 역할 | 어디서 도나 |
|---|---|---|
| `g1_protocol.py` | 포트·라벨·`plan()`(방위 → 세그먼트 목록)·JSON encode/decode | 전부 (Jetson 은 3.8) |
| `motion/build_motion_library.py` | 클립 csv(segments.yaml, 지금 9개) → 블렌드·yaw/xy 합성 → `library.csv` + `library_meta.json`. 이음새 임계 초과 시 exit 1 | 데스크탑 |
| `motion/segments.yaml` | 세그먼트 이름 → csv 경로, 순서 | 데스크탑 |
| `deploy/segment_player.py` | 세그먼트 커서 상태머신 (idle 반복 / act → plan() 세그먼트 → idle, seq·BUSY·bearing 타입 거부) | Jetson |
| `deploy/g1_motion_server.py` | 50 Hz 정책 실행부 (`deploy_g1_tracking.py` 확장). probe·dry-run 기본·`--arm`·토크 가드·`--backend mujoco --fake_events N` | Jetson / 데스크탑(sim2sim) |
| `deploy/g1_tracking_policy_meta.json` | 관절 순서·게인·스케일 (Isaac 실측 덤프 복사본) | Jetson |
| `deploy/export_policy_npz.py` | 체크포인트 → numpy 정책 npz (torch 출력과 200표본 대조) | 데스크탑 |
| `g1_gaze_bridge.py` | 시선 픽셀 + YOLO 부위 + 태그 번들 방위 → UDP 55057 | 데스크탑 |
| `config.yaml` | 몸통 4면 태그 번들(`anchor:`) + YOLO 부위 모델(`yolo:`) | 데스크탑 |
| `tools/capture_frames.py` | 씬 영상을 N 프레임마다 저장 (YOLO 부위 학습 데이터) | 데스크탑 |
| `tests/` | pytest (하드웨어 없이) | 데스크탑 |
| `../ros2_ws/src/gaze_hri/gaze_hri/g1_interaction_node.py` · `g1_dwell.py` · `launch/g1_demo.launch.py` | 라벨 dwell → 확정 이벤트 → Jetson 명령, `/g1/*` 토픽 | 데스크탑 (ROS 2 Jazzy) |

## 빠른 시작 — 로봇 없이 되는 것

```bash
cd ~/창종설          # 저장소 루트 (worktree 면 그 경로)

# 1) 단위 테스트 — 프로토콜·커서·라이브러리·브리지 순수 함수·모션 서버 인자
/usr/bin/python3 -m pytest g1/tests -q
/usr/bin/python3 -m pytest ros2_ws/src/gaze_hri/test/test_g1_dwell.py -q

# 2) 모션 서버 인자 확인
/usr/bin/python3 g1/deploy/g1_motion_server.py --help | grep -E "library_meta|fake_events"

# 3) sim2sim (정책 학습 후) — 로봇 없이 시나리오 전체: 무작위 이벤트 10회
ART=~/molbwa_g1      # 산출물 폴더 (저장소 밖 — npz 는 커밋하지 않는다, RUNBOOK 0장)
cd g1/deploy
env -u PYTHONPATH -u LD_LIBRARY_PATH ~/miniconda3/envs/g1deploy/bin/python g1_motion_server.py \
    --policy $ART/policy_molbwa.npz --motion ~/whole_body_tracking/motions/molbwa_library.npz \
    --library_meta $ART/library_meta.json \
    --backend mujoco --fake_events 10 --hold_sec 5

# 4) 상호작용 노드 단독 기동 — Jetson 이 없으니 /g1/state 는 offline 이어야 정상
cd ~/창종설/ros2_ws && colcon build --packages-select gaze_hri && source install/setup.bash
ros2 run gaze_hri g1_interaction --ros-args -p jetson_host:=127.0.0.1
ros2 topic echo /g1/state          # 다른 터미널: offline
```

`--fake_events` 는 mujoco 백엔드 전용이다. 6초 간격으로 무작위 part/bearing(−180~180 정수, 10% null) `act` 를 내부 주입하고(`--fake_seed` 로 고정),
마지막 동작이 idle 로 돌아오면 스스로 끝난다.
끝나면 이벤트별 (part, bearing, 세그먼트, 목표 heading = 수락 시 yaw + 계획 회전 bin, heading 오차 = 동작 첫 프레임 torso yaw − 목표(= 회전 정확도), act 수신→세그먼트 첫 프레임 지연 ms) 표를 찍는다.
합격 기준은 `RUNBOOK.md` 5장.

정책·라이브러리 npz 가 아직 없으면 3) 은 못 돈다 — 촬영부터 학습까지 `RUNBOOK.md` 1~4장.

## 안전 — 먼저 읽을 것

- 기본은 **dry-run**. 실제 발행은 `--arm` 일 때만이고, `--arm` 시 MotionSwitcher `CheckMode()` 의 name 이
  비어 있지 않으면(고수준 모드) 발행을 거부한다.
- **약하게 시작하는 길은 없다.** `--gain_ramp` 를 내리면 넘어지고 `--motion_scale` 을 내리면 토크가 2.7배 오른다.
  모션 서버에서는 두 인자를 아예 뺐다. 안전은 하네스(느슨하게)·매트·비상정지 담당으로 확보한다.
- 보간 진입 없음 — 정책을 **첫 스텝부터** 켠다(`--enter_sec 0`). 정책이 균형을 잡는 주체다.
- 명령 채널이 끊겨도 idle 을 계속 재생한다(댐핑하지 않는다 — 정책을 끄면 무조건 주저앉는다).
- 종료 시 정책이 꺼지면 넘어진다 → **사람이 받친다.**
- UDP 55070 에는 인증이 없다 — 같은 와이파이의 누구든 `act` 를 보낼 수 있다. 실기는 전용 AP 에서만.
- **Jetson 이 2026-10-05 이후 ping 이 안 된다.** 실기 단계 전에 먼저 해결해야 한다(`RUNBOOK.md` 8장).
