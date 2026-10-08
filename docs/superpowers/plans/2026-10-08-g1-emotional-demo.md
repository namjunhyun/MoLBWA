# 데모 2 G1 정서 교감 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 시선으로 G1 의 얼굴/손/몸통을 응시하면 G1 이 서서 사용자 쪽으로 돌아 인사/악수 자세/팔 벌리기를 하는 전체 경로(모션 라이브러리 → 정책 실행부 → 시선 연결)를 하드웨어 없이 검증 가능한 상태로 만든다.

**Architecture:** 촬영 클립을 하나의 연속 모션 라이브러리 csv 로 이어 BeyondMimic 정책 1개를 학습하고, Jetson 모션 서버가 세그먼트 커서로 idle → turn → 동작 → idle 을 재생한다. 데스크탑은 AprilTag 번들로 방위각, YOLO 로 응시 부위를 구해 라벨 dwell 로 확정하고 UDP 로 명령을 보낸다.

**Tech Stack:** Python 3 (Jetson 은 3.8 — 배포 파일은 3.8 호환), numpy, pupil_apriltags, ultralytics(별도 `~/yolo-env`), ROS 2 Jazzy rclpy, MuJoCo(sim2sim), pytest(`/usr/bin/python3 -m pytest`).

**Spec:** `docs/superpowers/specs/2026-10-08-g1-emotional-demo-design.md`

## Global Constraints

- 정책 관측은 154차원 Wo-State-Estimation 전용. 160차원 정책 거부(기존 검사 유지).
- 정책→모터는 Jetson 온보드 50 Hz. 데스크탑에서 lowcmd 발행 금지.
- 기본은 dry-run, 실제 발행은 `--arm` 일 때만. `--arm` 시 CheckMode 이름이 비어 있지 않으면 발행 거부(기존 유지).
- `--gain_ramp`/`--motion_scale` 로 약하게 시작하는 경로를 새로 만들지 않는다.
- 명령 채널이 끊겨도 idle 유지(댐핑하지 않는다).
- UDP 포트: 브리지→노드 55057, 노드→Jetson 55070, Jetson→노드 55071. 기존 55055/55056/55059 건드리지 않음.
- 세그먼트 이름: `idle`, `turn_l45`, `turn_l90`, `turn_l135`, `turn_l180`, `turn_r45`, `turn_r90`, `turn_r135`, `bow`, `handshake`, `open_arms`.
- 부위 라벨: `g1_face`, `g1_hand`, `g1_torso` → 동작 `bow`, `handshake`, `open_arms`. 판정 우선순위 hand > face > torso.
- turn_bin ∈ {0, 45, 90, 135, 180, -45, -90, -135}, + 가 왼쪽, 180 은 `turn_l180`.
- csv 36열 = root pos 3 + root quat **xyzw** 4 + 관절 29 (기존 LAFAN1/GMR csv 규약, `csv_to_npz.py` 가 wxyz 로 바꿈). fps 30 입력.
- 하위 에이전트는 **커밋하지 않는다** (같은 worktree 공유) — 코디네이터가 리뷰 후 커밋.

## Review Focus

1. 사용자가 G1 바로 뒤(±180° 경계)에 있을 때 bearing 이 179°/−179° 사이에서 흔들림 → 둘 다 turn_bin 180 이어야 한다. (Task 4 테스트)
2. 동작 중(BUSY) 같은 부위를 계속 응시 → 동작이 끝난 직후 곧바로 재실행되지 않아야 한다(cooldown + 라벨 이탈 필요). (Task 5 테스트)
3. 명령 UDP 패킷 중복/역순 도착(와이파이 재전송) → 같은/작은 seq 는 무시. (Task 2 테스트)
4. 태그가 한 장만 보이는 정면 응시 → bearing 이 나와야 한다(번들 min_tags 1). 안 보이면 turn_bin 0 으로 동작만. (Task 4 테스트)
5. 촬영 클립 끝이 대기 자세와 몇 도 어긋남 → 블렌드로 이음새 관절 점프가 프레임당 임계 이하. (Task 1 테스트)

---

### Task 1: UDP 프로토콜 (코디네이터 직접)

**Files:**
- Create: `g1/g1_protocol.py`
- Test: `g1/tests/test_g1_protocol.py`

**Interfaces:**
- Produces:
  - `PORT_BRIDGE = 55057`, `PORT_CMD = 55070`, `PORT_STATE = 55071`
  - `PART_TO_SEGMENT = {"g1_face": "bow", "g1_hand": "handshake", "g1_torso": "open_arms"}`
  - `TURN_BINS = (0, 45, 90, 135, 180, -45, -90, -135)`; `turn_segment(bin: int) -> str | None` (0 → None, 180 → "turn_l180", 45 → "turn_l45", −45 → "turn_r45")
  - `nearest_bin(deg: float) -> int` — TURN_BINS 중 원형 거리 최소, ±180 근방은 180 (Task 4·5 가 재사용)
  - `encode(msg: dict) -> bytes`, `decode(data: bytes) -> dict | None` (JSON utf-8, 깨진 패킷 None)
  - 메시지: bridge `{"t", "label", "bearing_deg", "valid"}`, cmd `{"seq", "cmd": "act"|"ping", "part", "turn_bin"}`, state `{"state": "idle"|"turning"|"acting", "seq"}`

- [ ] 테스트: `turn_segment` 전 bin 매핑, encode/decode 왕복, 깨진 바이트 → None.
- [ ] 구현, `/usr/bin/python3 -m pytest g1/tests/test_g1_protocol.py -q` PASS.

### Task 2: 세그먼트 플레이어 (그룹 B)

**Files:**
- Create: `g1/deploy/segment_player.py`
- Test: `g1/tests/test_segment_player.py`

**Interfaces:**
- Consumes: `g1_protocol.turn_segment`, `PART_TO_SEGMENT`
- Produces: `class SegmentPlayer(segments: dict[str, tuple[int, int]])`
  - `step() -> tuple[int, bool]` — 이번 제어 스텝의 라이브러리 프레임 인덱스와 "세그먼트 첫 프레임인가"(True 면 호출측이 yaw_off 재정렬).
  - `command(msg: dict) -> bool` — `cmd=="act"` 이고 idle 이며 `seq > last_seq` 일 때만 수락(True). 수락하면 큐 = [turn_segment(bin) (None 이면 생략), PART_TO_SEGMENT[part]]. 큐는 **현재 idle 반복이 끝난 뒤가 아니라 다음 step 에서 즉시** 시작(첫 반응 시간). 거부해도 `last_seq` 는 수락분만 갱신.
  - `state -> str` ("idle"|"turning"|"acting"), `last_seq -> int`.
  - 라이브러리 fps 50(npz) = 제어 50 Hz 이므로 step 당 1프레임.

- [ ] 테스트 `test_idle_loops`: idle [0,10) 에서 25 step → 인덱스 0..9,0..9,0..4, 첫 프레임 플래그는 0 에서만 True.
- [ ] 테스트 `test_act_turn_then_gesture_then_idle`: act(part g1_face, bin 90) → 다음 step 이 turn_l90 시작(flag True), 끝나면 bow 시작(flag True), 끝나면 idle 시작, state 전이 turning→acting→idle.
- [ ] 테스트 `test_bin0_skips_turn`, `test_busy_ignored`(turning 중 act → False), `test_seq_duplicate_and_old_ignored`(seq 5 수락 후 5·3 거부, 6 은 idle 에서 수락), `test_ping_ignored`, `test_unknown_part_rejected`.
- [ ] 구현 후 PASS. Python 3.8 호환(`from __future__ import annotations`).

### Task 3: Jetson 모션 서버 (그룹 B, Task 2 후)

**Files:**
- Create: `g1/deploy/g1_motion_server.py` (← `~/g1_dance_deploy/deploy_g1_tracking.py` 복사 후 수정), `g1/deploy/g1_tracking_policy_meta.json` (복사), `g1/deploy/export_policy_npz.py` (복사)
- Test: `g1/tests/test_motion_server_smoke.py`

**Interfaces:**
- Consumes: `SegmentPlayer`, `g1_protocol.*`, `library_meta.json`(Task 6 산출, 형식은 Task 6 Interfaces)
- 추가 인자: `--library_meta` (필수), `--cmd_port 55070`, `--state_host 127.0.0.1`, `--state_port 55071`, `--run_sec 0`(0=무한, 정지 신호까지), `--fake_events N`(mujoco 백엔드 전용: 6 초 간격으로 무작위 part/bin act 를 내부 주입).
- 수정점:
  - 메인 루프 `for t in range(T_ALL)` → `while`: `k, first = player.step()`; `first` 이면 `yaw_off = yaw_of(REF_BQ[k, B_ANCHOR]) - yaw_of_mat(torso_rot_world(imu, q))`; 관측·로그는 `REF_*[k]`.
  - 비차단 UDP 수신 소켓(`setblocking(False)`, 스텝마다 `recvfrom` 을 BlockingIOError 날 때까지) → `player.command(decode(...))`.
  - 5 Hz(10 스텝마다) state 송신 `{"state", "seq"}`.
  - 기존 토크 가드·late 주기 가드·SIGTERM/PID 파일·`--probe`·dry-run·CheckMode 거부·종료 경로(유지/카운트다운/댐핑) 그대로. 유지 단계의 `rbq_last` 는 마지막 실행 프레임 `k` 기준.
  - 요약에 `--fake_events` 일 때 이벤트별 (part, bin, 목표 heading, 실제 heading 오차 deg, act 수신→세그먼트 첫 프레임 지연 ms) 표 출력.
  - 모든 import·경로 하드코딩 금지(SDK 경로 후보 순회, `--iface auto` 유지).

- [ ] 테스트 `test_parses_args`: `python3 g1/deploy/g1_motion_server.py --help` 종료코드 0, 출력에 `--library_meta`, `--fake_events`.
- [ ] 테스트 `test_obs_dim_guard`: 160차원 가짜 정책 npz 로 실행 시 비정상 종료 + "154" 메시지(기존 검사 회귀 방지).
- [ ] `/usr/bin/python3 -m pytest g1/tests/test_motion_server_smoke.py -q` PASS.
- [ ] 실정책 sim2sim 은 학습 후(Task 7 런북 절차) — 이 태스크 범위 아님.

### Task 4: 시선→부위·방위 브리지 (그룹 C)

**Files:**
- Create: `g1/g1_gaze_bridge.py`, `g1/config.yaml`
- Test: `g1/tests/test_g1_gaze_bridge.py`

**Interfaces:**
- Consumes: `arm/anchor.py` `TagBundleDetector(cfg, K)`.detect(gray) → `T_hc_ab`(4×4, 태그번들 기준 프레임 → 헤드캠) 또는 None; `arm/run_demo.py` `GazeSource`, `load_intrinsics`; `gaze_hri/yolo_worker.py` 프로토콜(stdin JPEG, stdout `{"dets": [[cls, conf, x1, y1, x2, y2], ...]}`); `g1_protocol.encode`, `PORT_BRIDGE`.
- Produces (순수 함수):
  - `pick_part(dets: list, uv: tuple[float, float] | None) -> str | None` — uv 를 포함하는 박스 중 우선순위 hand > face > torso, 같은 클래스면 conf 큰 것.
  - `bearing_deg(T_hc_torso: np.ndarray) -> float` — 헤드캠 원점의 torso 좌표 `p = inv(T)[:3, 3]`, `degrees(atan2(p[1], p[0]))` ∈ (−180, 180]. (torso +x 정면, +y 왼쪽)
  - `nearest_bin` 은 `g1_protocol` 것을 재사용(re-export).
- `g1/config.yaml`: `anchor:` 블록(arm 형식 그대로) — tag36h11, `tag_size_m: 0.10`, 4장 id 10(앞)/11(왼)/12(오른)/13(뒤), pos = 몸통 표면 중심(실측 전 자리값: 앞 [0.10,0,0.20], 왼 [0,0.11,0.20], 오른 [0,−0.11,0.20], 뒤 [−0.10,0,0.20], torso_link 기준), 면마다 right/up, `min_tags_for_latch: 1`, `require_noncoplanar: false` (`# ponytail: 단일 태그 flip 허용 — bin 이 45° 라 정면 flip 오차는 작다. 판정 실패율 높으면 면당 2장+비동일평면으로`), `max_reproj_px: 3.0`. 그리고 `yolo: {model: "~/yolo-env/g1_parts.pt", conf: 0.25, classes: [g1_face, g1_hand, g1_torso]}`.
- 루프(`main`): `gaze_tag_bridge.py` 와 같은 인자 패턴(`--config`, `--scene-topic /pc/camera/left/compressed`, `--gaze-px-port 55056`, `--udp-host`, `--udp-port 55057`). 매 프레임 YOLO + 태그 → `{"t","label","bearing_deg","valid"}` 송신. 태그 실패 시 `bearing_deg: null`.

- [ ] 테스트: `pick_part` 겹친 박스(hand ⊂ torso → hand), 박스 밖 → None, uv None → None, 빈 dets → None.
- [ ] 테스트: `bearing_deg` — 사용자가 정면 2 m(헤드캠 torso 좌표 (2,0,0.5)) → 0, 왼쪽 → 90, 뒤 → 180, 오른쪽 → −90 (T 는 헤드캠이 torso 를 바라보도록 구성).
- [ ] 테스트: `nearest_bin` 22.4→0, 22.6→45, 179→180, −179→180, −157→180, −158→−135 경계 포함.
- [ ] 테스트: `g1/config.yaml` 이 `build_bundle_obj_pts` 로 로드되고 태그 4장.
- [ ] PASS.

### Task 5: ROS 2 상호작용 노드 (그룹 C, Task 4 와 독립)

**Files:**
- Create: `ros2_ws/src/gaze_hri/gaze_hri/g1_interaction_node.py`, `ros2_ws/src/gaze_hri/gaze_hri/g1_dwell.py`, `ros2_ws/src/gaze_hri/launch/g1_demo.launch.py`
- Modify: `ros2_ws/src/gaze_hri/setup.py` (entry `g1_interaction = gaze_hri.g1_interaction_node:main`), `ros2_ws/src/gaze_hri/config/gaze_hri.yaml` (`g1_interaction:` 블록)
- Test: `ros2_ws/src/gaze_hri/test/test_g1_dwell.py`

**Interfaces:**
- Consumes: `g1_protocol` (노드는 `sys.path` 에 저장소 `g1/` 를 넣어 import — 런치에 `g1_dir` 인자, 기본은 패키지 기준 상대경로 `../../../../g1`).
- Produces: `class LabelDwell(dwell_time=1.0, min_ratio=0.7, cooldown=2.0)` (순수, `g1_dwell.py`)
  - `update(t: float, label: str | None) -> str | None` — 최근 `dwell_time` 창에서 같은 label 비율 ≥ min_ratio 이고 창이 꽉 찼으면 확정 label 반환(1회). 확정 후 cooldown 동안 None, 그리고 그 label 이 창에서 사라질 때까지(비율 < 0.3) 같은 label 재확정 없음.
  - `progress(t) -> float` 0~1.
- 노드: 55057 수신 → `LabelDwell` → 확정 시 Jetson state 가 idle 이고 1 초 내 수신했을 때만 `{"seq": n, "cmd": "act", "part", "turn_bin": nearest_bin(bearing) 또는 0}` 를 55070 으로 송신, 1 Hz ping. 발행: `/g1/event`(String JSON: t, part, bearing_deg, turn_bin, seq), `/g1/state`(String: idle|turning|acting|offline), `/g1/dwell_progress`(Float32). 파라미터 `jetson_host`, `dwell_time 1.0`, `min_ratio 0.7`, `cooldown 2.0`.
  - `nearest_bin` 은 Task 4 의 `g1_gaze_bridge` 가 아니라 `g1_protocol` 에 두어야 노드가 cv2/apriltag 없이 import 된다 → **Task 1 에 `nearest_bin` 포함**, Task 4 는 재사용.

- [ ] 테스트 `test_confirms_after_dwell`(30 Hz, 1.0 s 동안 g1_face → 1.0 s 근처 1회 확정), `test_blink_tolerated`(20% None 섞여도 확정), `test_switch_resets`(0.6 s face 후 hand → face 확정 없음), `test_no_retrigger_while_staring`(3 s 계속 face → 1회만), `test_retrigger_after_leave`(face 확정 → 0.5 s None → face 1 s → 2회째 확정, cooldown 지난 경우).
- [ ] `colcon build --packages-select gaze_hri` 후 `ros2 run gaze_hri g1_interaction --ros-args -p jetson_host:=127.0.0.1` 기동 3 초 내 에러 없음, `/g1/state` = offline.
- [ ] PASS.

### Task 6: 모션 라이브러리 빌더 (그룹 A)

**Files:**
- Create: `g1/motion/build_motion_library.py`, `g1/motion/segments.yaml`
- Test: `g1/tests/test_build_motion_library.py`

**Interfaces:**
- Produces:
  - `load_csv(path) -> np.ndarray (T, 36)`; `build_library(clips: dict[str, np.ndarray], fps: int = 30, blend_s: float = 0.5) -> tuple[np.ndarray, dict]` → (library (N,36), meta).
  - meta (= `library_meta.json`, 모션 서버가 읽음): `{"fps_csv": 30, "fps_npz": 50, "segments": {name: [start, end]}}` — **npz(50 fps) 프레임 인덱스**로 저장(csv 인덱스 × 50/30 반올림, `csv_to_npz.py` 가 50 fps 로 리샘플). `end` 배타.
  - `seam_report(lib, meta) -> dict` — 프레임 간 최대 관절 점프(rad), root 위치 점프(m), yaw 점프(rad).
  - CLI: `python3 build_motion_library.py --segments segments.yaml --out_csv library.csv --out_meta library_meta.json`; 이음새 임계(관절 0.05 rad/프레임, root 0.02 m, yaw 0.05 rad) 초과 시 exit 1.
- 알고리즘(시그니처가 정하지 않는 부분):
  - 대기 자세 = `idle` 첫 프레임의 관절 29 + root 높이·roll·pitch.
  - 각 클립: 앞 `blend_s` 동안 대기→클립 시작, 뒤 `blend_s` 동안 클립 끝→대기 (관절 cos 보간, root 회전 slerp, root 높이 보간). 클립 자체 yaw·xy 는 보존.
  - 이어 붙이기: 클립 i 의 root 를 "클립 i 시작 yaw·xy 를 0 으로 만든 뒤 앞 클립 끝 yaw·xy 로 합성" (z 축 회전 + xy 평행이동). 쿼터니언 xyzw.
  - 순서는 `segments.yaml` 순서(기본: idle, turn_l45 … turn_r135, bow, handshake, open_arms).

- [ ] 테스트 `test_seams_continuous`: 합성 클립(관절 사인파, 끝이 대기 자세와 0.2 rad 어긋남, 회전 클립은 yaw 0→π/2) 3개 → `seam_report` 관절 ≤ 0.05, root ≤ 0.02, yaw ≤ 0.05.
- [ ] 테스트 `test_meta_indices`: 세그먼트 경계가 겹치지 않고 연속, 마지막 end = round(N·50/30) 이하, 각 길이 > 0.
- [ ] 테스트 `test_turn_yaw_accumulates`: turn_l90 다음 클립 시작 yaw = turn_l90 끝 yaw (±1e-6).
- [ ] 테스트 `test_quat_unit`: 전 프레임 쿼터니언 노름 1 ± 1e-6.
- [ ] 테스트 `test_cli_exit1_on_bad_seam`: blend_s=0 + 어긋난 클립 → CLI exit 1.
- [ ] PASS.

### Task 7: 문서와 데이터 수집 도구 (그룹 D)

**Files:**
- Create: `g1/README.md`, `g1/RUNBOOK.md`, `g1/tools/capture_frames.py`
- Modify: `docs/00_overview.md` 끝에 "데모 2: G1" 한 단락 + 링크

**Interfaces:**
- Consumes: 위 모든 태스크의 CLI·포트·파일명(이 문서의 Global Constraints 와 Task 별 Interfaces 그대로 인용).
- `capture_frames.py`: `--scene-topic /pc/camera/left/compressed --out DIR --every 10` — rclpy 로 CompressedImage 를 받아 N 프레임마다 jpg 저장, 종료 시 저장 수 출력. 테스트 없음(얇은 I/O).
- README: 구조도(스펙 그림), 파일 표, 빠른 시작(fake 경로).
- RUNBOOK 절차(명령 그대로):
  1. 촬영 가이드(스펙 A 표·조건, 클립마다 앞뒤 1 초 차렷).
  2. GVHMR → GMR → csv (`~/para_pipeline/run_all.sh` 의 명령 인용, 경로는 변수로).
  3. `build_motion_library.py` → `csv_to_npz.py --input_fps 30 --output_name molbwa_library` → `~/para_pipeline/check_motion.py motions/molbwa_library.npz` exit 0 확인 (csv_to_npz 는 저장 후 안 끝남 → npz 생성 확인 후 kill).
  4. 학습(스펙 B 명령), export_policy_npz.
  5. sim2sim: `g1_motion_server.py --backend mujoco --fake_events 10 --library_meta ...` — 합격: 10/10 서 있음(골반 최소 > 0.6 m), 첫 반응 ≤ 1.5 s, heading 오차 기록.
  6. YOLO 부위 데이터: capture_frames → 라벨링 → `yolo train` (정면/45°/측면/위아래, 수백 장).
  7. 태그: 몸통 4면 부착, `g1/config.yaml` pos 실측으로 교체.
  8. 실기: Jetson scp(server·segment_player·g1_protocol·meta·library·policy) → `--probe 5` → dry-run → 하네스·매트·비상정지 담당 배치 → `--arm`. 역할 고정(시연자·비상정지·설명).
  9. 현장 체크리스트(계획서 그대로) + 백업 영상.
  - "Jetson 연결이 10-05 이후 ping 실패 — 실기 전 해결" 경고 박스.

- [ ] 문서의 모든 명령의 파일·인자가 실제 존재하는지 grep 으로 대조(존재하지 않는 인자 0).
- [ ] `python3 -m py_compile g1/tools/capture_frames.py` 성공.

## 실행 순서

Task 1(코디네이터) → 그룹 A(Task 6) · B(Task 2→3) · C(Task 4→5) · D(Task 7) 병렬 → 코디네이터 리뷰·전체 테스트·커밋·푸시.
