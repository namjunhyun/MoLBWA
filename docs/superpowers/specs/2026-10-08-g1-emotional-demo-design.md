# 데모 2: 시선으로 G1 휴머노이드와 교감하기 — 설계

2026-10-08 · 브랜치 `feat/g1-emotional-demo` (기반 `feat/gaze-hri-pick-place` f1dd605)

## 목표

사용자가 G1 의 **얼굴 / 손(팔뚝 끝) / 몸통** 중 한 곳을 1초 응시하면, G1 이 **서 있는 상태로**
사용자 쪽으로 제자리 회전한 뒤 부위에 맞는 동작(인사 / 악수 자세 / 팔 벌리기)을 하고 대기로 돌아온다.
사용자가 로봇 옆에 있어도 동작해야 한다. 데모 1(로봇팔)과 같은 시선 인터페이스를 쓴다.

성공 기준 (대회 전 측정, 발표 표에 넣음):

| 지표 | 측정 | 목표 |
|---|---|---|
| sim2sim 시나리오 완주 | MuJoCo, 무작위 이벤트 10회 연속 | 10/10 서 있음 |
| 회전 후 heading 오차 | sim2sim, 목표 방위 대비 | 기록 (bin 양자화 ±22.5° 이내 기대) |
| 첫 반응 시간 | 확정 이벤트 → 회전 세그먼트 첫 프레임 | ≤ 1.5 초 |
| 부위 판정 성공률 | 정면 / 45° / 측면, 부위별 20회 | 팀 합의 |

진단·치료 효과는 주장하지 않는다.

## 결정 사항 (사용자 확정)

- **서 있는 상태 전신 RL 트래킹 정책** (거치대 버전 없음).
- **HMP(humanoid motion prior) 미사용.** 회전도 클립 트래킹으로 푼다(임의 각도 → 가장 가까운 bin).
- **모션 소스는 직접 촬영 영상** → 기존 춤 파이프라인(GVHMR → GMR → BeyondMimic) 재사용.
- 범위는 데모 2 전체: 모션, 학습 설정, Jetson 실행부, 시선 연결, 테스트, 런북.

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
           ├ dwell: 같은 label 1.0 초 유지(유효 70%) → 확정
           ├ bearing → turn_bin (0, ±45, ±90, ±135, ±180)
           ├ 상태: IDLE → BUSY(Jetson 보고 기준) → IDLE, BUSY 중 이벤트 무시
           ├ 발행 /g1/event, /g1/state (std_msgs/String, JSON) — rosbag 으로 정서 지표 기록
           └─UDP 55070 {seq, cmd:"act", part, turn_bin} / {cmd:"ping"} ─▶
[Jetson]  g1/deploy/g1_motion_server.py (50 Hz 온보드, numpy only)
           ├ 정책 1개 + 모션 라이브러리 npz(세그먼트 경계 meta)
           ├ idle 세그먼트 반복 → 명령 수신 시 turn 세그먼트 → 동작 세그먼트 → idle
           ├ 세그먼트 시작마다 yaw_off 재정렬
           └─UDP 55071 {state, seq} ─▶ 데스크탑
```

정책→모터 경로는 로봇 내부(온보드)만 지난다. 와이파이는 명령·상태·로그에만 쓴다
(데스크탑 정책 + 와이파이 lowcmd 는 p99 170~215 ms 로 금지 — `~/g1_dance_deploy/POLICY_DEPLOY_PLAYBOOK.md`).

## 구성 요소

### A. 모션 라이브러리 (`g1/motion/`)

**촬영 클립 11개**, 모두 같은 **차렷 대기 자세로 시작·끝**(앞뒤 1초 정지):

| 세그먼트 | 내용 |
|---|---|
| `idle` | 차렷 대기 4초 (호흡 정도의 미세 움직임만) |
| `turn_l45/90/135/180`, `turn_r45/90/135` | 제자리 회전 (180° 는 왼쪽 하나), 발을 작게 여러 번 디딤 |
| `bow` | 고개+상체 숙여 인사 (얼굴 응시) |
| `handshake` | 오른팔을 앞으로 내밀어 악수 자세, 2초 유지 후 복귀 (손 응시). 뻗는 거리 짧게 |
| `open_arms` | 팔 벌려 안아주는 자세, 2초 유지 후 복귀, 감싸지 않음 (몸통 응시) |

촬영 조건: 고정 카메라, 전신이 프레임 안, 정면 기준 3~4 m, 30 fps 이상, 단색 배경 권장.
처리: GVHMR → GMR(`gvhmr_to_robot.py` → `batch_gmr_pkl_to_csv.py`) → 36열 csv
(3 root pos + 4 quat + 29 joint, 기존 LAFAN1 G1 csv 와 같은 열 규약) → `check_motion.py` 품질 게이트.

**`build_motion_library.py`** (순수 numpy, 테스트 대상):
- 입력: 세그먼트 이름 → csv 경로 목록, fps.
- 각 세그먼트 앞뒤를 공통 대기 자세(`idle` 첫 프레임)로 0.5 초 블렌드(관절 선형, root 회전 slerp).
- 다음 세그먼트의 root yaw·xy 를 앞 세그먼트 끝 자세에 합성해 **이음새에서 root 가 끊기지 않게** 한다.
- 출력: `library.csv`(36열, 이후 `csv_to_npz.py` 1회) + `library_meta.json`
  `{fps, segments: {name: [start_frame, end_frame)}, turn_bins: {name: deg}}`.
- 이음새 검사: 관절 점프 최대값, root 위치 점프, yaw 점프를 출력하고 임계 초과 시 실패.
- npz 가 아니라 **csv 단계에서 잇는 이유**: `csv_to_npz.py` 가 Isaac FK 로 body 위치를 만들기 때문에,
  이어 붙인 뒤 FK 를 한 번 돌려야 body 궤적이 일관된다.

### B. 학습 (코드 추가 없음)

```
~/bin/isaaclab_run.sh -p scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-Delay-v0 \
  --motion_file motions/molbwa_library.npz --num_envs 1024 --max_iterations 6000 --headless \
  env.rewards.action_rate_l2.weight=-0.1 env.actions.joint_pos.max_delay_steps=3
```
- Wo-State-Estimation(154차원 관측)만 실기 가능. 지연 랜덤화 0~60 ms, action_rate −0.1 (파라파라 실기 발산 교훈).
- 산출: `export_policy_npz.py` → `policy_molbwa.npz`.

### C. Jetson 실행부 (`g1/deploy/`)

`~/g1_dance_deploy/deploy_g1_tracking.py`(실기 검증본)를 복사해 확장한 `g1_motion_server.py`:
- `--library_meta library_meta.json`, `--cmd_port 55070`, `--state_port 55071`, `--state_host <데스크탑>`.
- 재생 커서: `idle` 구간 반복 → `act` 명령 수신 시 `turn_<bin>`(bin 0 이면 생략) → 동작 세그먼트 → `idle`.
- **세그먼트 시작마다** `yaw_off = yaw(REF anchor @ seg_start) − yaw(현재 torso)` 재정렬
  (기존 코드는 시작 시 1회만 — 794행).
- BUSY 중 `act` 무시, 같은 `seq` 재수신 무시. 명령 채널 끊김 → idle 유지(댐핑 금지, 넘어진다).
- 기존 안전장치 유지: `--probe`, dry-run 기본, `--arm` 시 CheckMode 이름 비어 있지 않으면 발행 거부,
  토크 폭주 가드, 300 초 유지 후 카운트다운, 종료 시 사람이 받침. `--gain_ramp`/`--motion_scale` 로 약하게 시작하지 않는다.
- `--backend mujoco` + `--fake_events N` 으로 로봇 없이 전체 시나리오 검증.
- 잔차 보정(bin 양자화 ±22.5° 를 허리 yaw 로 흡수)은 학습 분포 밖이라 **기본 꺼짐**, sim2sim 측정 후 결정.

### D. 시선 연결 (`g1/g1_gaze_bridge.py`, `ros2_ws/src/gaze_hri/gaze_hri/g1_interaction_node.py`)

- `g1_gaze_bridge.py`: `arm/gaze_tag_bridge.py` 와 같은 구조(루프에서 떼어 낸 순수 함수 + 얇은 루프).
  - G1 몸통 **앞·좌·우·뒤 4면 태그**(번들, `arm/anchor.py` 의 `TagBundleDetector` 재사용) → 헤드캠의 G1 torso 기준 위치 → `bearing_deg = atan2(y, x)`.
    태그 1장만 쓰면 사용자가 옆에 있을 때 안 보인다.
  - 부위 판정: 씬 프레임을 `gaze_hri/yolo_worker.py` 에 넘기고 시선 픽셀을 포함하는 박스 중 우선순위 **hand > face > torso** (손 박스가 몸통 박스 안에 있기 때문).
    세그멘테이션은 박스로 판정 실패율이 높을 때 추가.
  - 출력 UDP 55057 JSON `{t, label|null, bearing_deg|null, valid}`.
- `g1_interaction_node.py` (ROS 2, `setup.py` entry point `g1_interaction`):
  - dwell 판정은 3D 점이 아니라 **라벨 유지** 기반(같은 label 비율 ≥ 70% 가 `dwell_time`=1.0 초). 확정 후 `cooldown` 동안 재확정 없음.
  - `turn_bin = 가장 가까운 {0, ±45, ±90, ±135, 180}`(+ 는 왼쪽, ±180 은 180 하나).
  - Jetson 상태(55071) 가 BUSY 거나 1 초 이상 미수신이면 확정 이벤트를 보내지 않는다.
  - `/g1/dwell_progress`(Float32) 를 내서 기존 HUD 패턴으로 피드백.
  - 런치 `g1_demo.launch.py`: `g1_interaction` 만(브리지는 별도 프로세스, 기존 패턴).
- YOLO 데이터: `g1/tools/capture_frames.py` 로 씬 영상에서 N 프레임 간격 저장 → 라벨링(외부 도구) → 학습은 런북 절차.

### E. 문서 (`g1/README.md`, `g1/RUNBOOK.md`)

촬영 가이드(클립 목록·조건), 리타게팅·라이브러리·학습·export 명령, sim2sim 검증, Jetson 배포·probe·실기 절차
(하네스·매트·비상정지 담당·시연자 역할), 현장 체크리스트.

## 오류 처리

| 상황 | 동작 |
|---|---|
| 태그 미검출 | `bearing_deg=null` → 확정 이벤트에 turn_bin 0(회전 없이 동작만) |
| 시선 invalid / 부위 없음 | dwell 진행 리셋 |
| Jetson 상태 미수신 | 이벤트 보내지 않음, `/g1/state` 에 `offline` |
| 명령 채널 끊김 (Jetson) | idle 유지 |
| 토크 가드 발동 | 기존 동작(능동 댐핑) — 하네스가 받는다 |

## 테스트

pytest (`g1/tests/`, `ros2_ws/src/gaze_hri/test/`), 하드웨어 없이:
- 라이브러리: 이음새 관절·root·yaw 점프 0 근처, 세그먼트 경계 meta 정확, 블렌드 길이, 대기 자세 일치.
- 부위 판정: 겹친 박스 우선순위, 박스 밖 → None, 빈 검출.
- bearing → turn_bin 경계값(22.5°, 180° 래핑).
- 라벨 dwell: 유지·깜빡임·라벨 전환 리셋·cooldown.
- 모션 서버 커서 상태머신: idle 반복, act → turn → 동작 → idle, BUSY 중 무시, seq 중복, bin 0 회전 생략.
- UDP 패킷 인코드/디코드 왕복.
- sim2sim (MuJoCo, 정책 학습 후): `--fake_events 10` 완주·heading 오차·첫 반응 시간.

## 범위 밖

- HMP / AMP / 잠재 스킬 정책.
- 거치대·허리만 회전 버전.
- 계획서의 "짧은 선반응(상체 살짝 틀기)" — 첫 반응 시간이 1.5 초를 넘으면 추가.
- 새 ROS 메시지 타입(JSON String 사용).
- 정서 지표 시각화 — rosbag 기록까지만.

## 남은 위험

- GVHMR 단안 추정의 제자리 회전 발 미끄러짐 → `check_motion.py` 게이트, 실패 시 재촬영.
- 실기 이식: Jetson 연결이 2026-10-05 이후 막혀 있음(ping 실패). 실기 단계 전 해결 필요.
- 학습 시간: 라이브러리 길이 약 60~80 초, 6000 iter 기준 수 시간 — 촬영 직후 바로 시작해야 4주 일정 안.
