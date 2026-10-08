# 데모 2 G1 — 실행 절차 (촬영 → 학습 → sim2sim → 실기)

작성 2026-10-08. 구조·규약은 [`README.md`](README.md). 춤 정책을 실기에 올리며 배운 것은
`~/g1_dance_deploy/POLICY_DEPLOY_PLAYBOOK.md` 에 전부 있고, 이 문서는 그중 이번 데모에 걸리는 것만 옮겼다.
**명령은 그대로 복사해 쓴다.** 인자를 손으로 바꾸고 싶어지면 먼저 그 이유가 플레이북에 기각돼 있지 않은지 본다.

> **⚠ Jetson 연결이 2026-10-05 이후 ping 실패다. 실기(8장) 전에 반드시 먼저 해결한다.**
> 1~7장은 로봇 없이 된다. 8장은 Jetson 에 SSH 가 되고 `uname -m` = `aarch64` 가 나와야 시작한다.

---

## 0. 공통 변수

```bash
REPO=~/창종설                    # 이 저장소 (worktree 에서 작업 중이면 그 경로)
ART=~/molbwa_g1                  # 산출물 — 저장소 밖. 영상·pkl·csv·npz 는 커밋하지 않는다
RAW=$ART/raw                     # 촬영 원본 mp4 (세그먼트 이름 그대로: idle.mp4, turn_l45.mp4, ...)
mkdir -p "$RAW" "$ART/gmr_out"
```

기존 산출물은 덮어쓰지 않는다 — 같은 이름이 있으면 먼저 `mv x x.bak_$(date +%m%d_%H%M)` 로 비킨다.

**GPU 작업(2·3·4·6장의 GVHMR·Isaac·학습·YOLO 학습)은 다른 GPU 작업(VLA 학습 등)이 돌고 있으면 시작하지 않는다.**
시작 전 `nvidia-smi` 로 확인한다. 같이 돌리면 둘 다 느려지고 RAM 16GB 가 먼저 바닥난다.

### 0-1) 장비 배치 — 데스크탑 2대 (2026-10-08)

| | 데스크탑 A (데모 1) | 데스크탑 B (데모 2, 이 문서) |
|---|---|---|
| 글래스 | A (oCamS) | B (컴팩트 스테레오, 엣지 = Pi 5 또는 Arduino UNO Q) |
| 프로세스 | gaze_hri 로봇팔 스택 | `g1_gaze_bridge.py` + YOLO 워커 + `g1_interaction` |
| 로봇 | SO-ARM101 2대 | Jetson `g1_motion_server.py` (명령은 B 에서만) |
| `ROS_DOMAIN_ID` | 예: 11 | 예: 22 |

**두 데스크탑이 같은 네트워크에 있으면 `ROS_DOMAIN_ID` 를 반드시 다르게 둔다.** 같으면 두 데모의
`/gaze/*` 토픽이 섞여 데모 1 시선이 데모 2 에 들어온다. 각 데스크탑 `~/.bashrc` 에 `export ROS_DOMAIN_ID=22` 처럼 고정하고,
글래스 엣지(Pi/UNO Q)도 짝 데스크탑과 같은 값으로 맞춘다. UDP 포트(55056/55057/55070/55071)는 데스크탑 B 안에서만 쓰이므로 겹치지 않는다.

**글래스 B 엣지가 Arduino UNO Q 인 경우:** 데스크탑 B 코드는 엣지 기종을 모른다 — 시선 픽셀 UDP 55056 과
씬 영상 `CompressedImage` 토픽만 오면 된다. 확인이 필요한 건 카메라 쪽이다.
UNO Q(Qualcomm QRB2210, Debian)는 공식 문서상 **4-lane MIPI-CSI-2 + ISP 2개**가 있다. USB(UVC) 카메라 동작은 확인된 자료가 없다.
- Arducam OV9281 동기화 스테레오 키트는 라즈베리파이 CSI 용이라 UNO Q 에 그대로 붙는다는 보장이 없다.
- OAK-FFC-4P 는 USB3 + 보드 자체 하드웨어 동기화라 엣지 의존이 적지만, UNO Q 에서 depthai(USB) 가 도는지 실측이 먼저다.
- 실측 순서: UNO Q 에 카메라 1대 연결 → 30 fps 캡처 확인 → ROS 2 로 `CompressedImage` 발행 → 데스크탑 B 에서 수신 지연 측정.

---

## 1. 촬영

클립 **11개**, 전부 **같은 기본자세(편하게 선 자세 — 팔은 자연스럽게 내리고 무릎은 펴되 잠그지 않음)로 시작하고 끝난다. 클립마다 앞뒤 1초는 그 자세로 정지.**
라이브러리 빌더가 클립 앞뒤를 `idle` 첫 프레임 자세로 0.5초 블렌드하는데, 클립 끝이 대기 자세에서
멀수록 블렌드 구간의 관절이 빨리 움직여 이음새 검사(관절 0.05 rad/프레임)에 걸린다.

| 세그먼트 (= 파일 이름) | 내용 |
|---|---|
| `idle` | 편하게 선 기본자세로 4초 대기 (호흡 정도의 미세 움직임만) |
| `turn_l45`, `turn_l90`, `turn_l135`, `turn_l180` | 제자리 **왼쪽** 회전. 발을 작게 여러 번 디딤 |
| `turn_r45`, `turn_r90`, `turn_r135` | 제자리 **오른쪽** 회전 (180° 는 왼쪽 하나만 쓴다) |
| `bow` | 고개+상체 숙여 인사 (얼굴 응시) |
| `handshake` | 오른팔을 앞으로 내밀어 악수 자세, 2초 유지 후 복귀 (손 응시). 뻗는 거리는 짧게 |
| `open_arms` | 팔 벌려 안아주는 자세, 2초 유지 후 복귀. 감싸지 않는다 (몸통 응시) |

촬영 조건:
- **고정 카메라**(삼각대) — GVHMR 을 `-s`(static cam)로 돌린다
- 전신이 처음부터 끝까지 프레임 안, 정면 기준 3~4 m
- 30 fps 이상 (더 높으면 `batch_gmr_pkl_to_csv.py` 가 30 fps 로 솎는다)
- 단색 배경 권장, 헐렁한 옷 피하기
- 회전 클립은 **각도를 정확히** — 바닥에 테이프로 0/45/90/135/180° 표시. 이 각도가 곧 로봇의 회전량이다
- 회전 중 발을 끌지 않는다(디뎌서 돈다). GVHMR 단안 추정은 제자리 회전에서 발이 미끄러지기 쉽다

---

## 2. GVHMR → GMR → csv

`~/para_pipeline/run_all.sh` 1~2단계와 같은 명령을 클립마다 돌린다.

```bash
SEGS="idle turn_l45 turn_l90 turn_l135 turn_l180 turn_r45 turn_r90 turn_r135 bow handshake open_arms"

# 2-1) GVHMR — 영상 → SMPL (outputs/demo/<이름>/hmr4d_results.pt)
for s in $SEGS; do
  ( cd ~/GVHMR && env -u PYTHONPATH ~/miniconda3/envs/gvhmr/bin/python tools/demo/demo.py \
      --video="$RAW/$s.mp4" -s ) > "$ART/gvhmr_$s.log" 2>&1 || echo "실패: $s ($ART/gvhmr_$s.log)"
done

# 2-2) GMR 리타게팅 — SMPL → G1 pkl
for s in $SEGS; do
  ( cd ~/GMR && env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
      ~/miniconda3/envs/gmr/bin/python scripts/gvhmr_to_robot.py \
      --gvhmr_pred_file ~/GVHMR/outputs/demo/$s/hmr4d_results.pt \
      --robot unitree_g1 --save_path "$ART/gmr_out/$s.pkl" ) > "$ART/gmr_$s.log" 2>&1
  [ -f "$ART/gmr_out/$s.pkl" ] || echo "실패: $s ($ART/gmr_$s.log)"
done

# 2-3) pkl → csv (폴더 안 pkl 전부 → $ART/gmr_out/csv/<이름>.csv)
( cd ~/GMR && env -u PYTHONPATH ~/miniconda3/envs/gmr/bin/python \
    scripts/batch_gmr_pkl_to_csv.py --folder "$ART/gmr_out" )

# 2-4) 11개 다 있고 전부 36열인지
for s in $SEGS; do
  f="$ART/gmr_out/csv/$s.csv"
  [ -f "$f" ] && echo "$s $(head -1 "$f" | awk -F, '{print NF}')열 $(wc -l < "$f")행" || echo "없음: $s"
done
```

빠진 클립이 있으면 여기서 멈춘다. 라이브러리 빌더는 없는 파일을 건너뛰지 않고 예외로 죽는다.

---

## 3. 라이브러리 → npz → 품질 게이트

### 3-1) 라이브러리 빌드

`g1/motion/segments.yaml` 은 csv 를 **yaml 파일 기준 `clips/<이름>.csv`** 로 찾는다. `$ART` 로 복사하고
`clips` 를 2장 출력 폴더에 링크하면 고칠 것이 없다. 형식은 `segments:` 아래 `{name, csv}` 목록 + 최상위 `fps`(30)·`blend_s`(0.5),
이어 붙이는 순서는 yaml 순서 그대로다.

```bash
cp "$REPO/g1/motion/segments.yaml" "$ART/segments.yaml"
ln -sfn "$ART/gmr_out/csv" "$ART/clips"
/usr/bin/python3 "$REPO/g1/motion/build_motion_library.py" \
    --segments "$ART/segments.yaml" \
    --out_csv ~/whole_body_tracking/motions_csv/molbwa_library.csv \
    --out_meta "$ART/library_meta.json"
echo "exit $?"
```

- 이음새 리포트(관절 rad/프레임, root m, yaw rad)와 세그먼트 경계를 찍는다.
- **exit 1 = 이음새 임계 초과**(관절 0.05 rad/프레임, root 0.02 m, yaw 0.05 rad). 출력 파일을 쓰지 않는다.
  리포트의 `worst_seam` 세그먼트를 재촬영하거나(끝 자세가 대기와 멀다), yaml 의 `blend_s` 를 늘린다.
- `library_meta.json` 의 `segments` 는 **npz(50 fps) 프레임 인덱스** `[start, end)` 다. 모션 서버가 이걸 읽는다.

### 3-2) csv → npz (Isaac FK)

**`csv_to_npz.py` 는 npz 를 저장한 뒤에도 끝나지 않는다** (Isaac Sim 앱이 계속 돈다 — 제로투 때 9시간 48분 낭비).
npz 가 생기고 크기가 멈추면 우리가 끊는다 (`~/para_pipeline/run_fix3.sh` 와 같은 방식).

```bash
NPZ=~/whole_body_tracking/motions/molbwa_library.npz       # 저장 위치는 스크립트가 정한다(motions/<output_name>.npz)
[ -f "$NPZ" ] && mv "$NPZ" "$NPZ.bak_$(date +%m%d_%H%M)"
( cd ~/whole_body_tracking && ~/bin/isaaclab_run.sh -p scripts/csv_to_npz.py \
    --input_file motions_csv/molbwa_library.csv --input_fps 30 --output_name molbwa_library --headless ) \
    > "$ART/npz.log" 2>&1 &
NPID=$!
for i in $(seq 1 180); do
  if [ -f "$NPZ" ]; then
    sleep 4; s1=$(stat -c%s "$NPZ"); sleep 2; s2=$(stat -c%s "$NPZ")
    [ "$s1" = "$s2" ] && break
  fi
  sleep 5
done
kill $NPID 2>/dev/null; sleep 2; kill -9 $NPID 2>/dev/null
grep "Motion saved locally" "$ART/npz.log"
ps -eo pid,args | grep '[c]sv_to_npz'      # 남은 게 있으면 그 PID 만 kill (pkill -f 금지 — 자기 셸이 죽는다)
```

### 3-3) npz 길이가 meta 와 맞는지

csv(30 fps) → npz(50 fps) 리샘플 후 프레임 수가 meta 의 마지막 `end` 이상이어야 한다.
모자라면 모션 서버가 마지막 세그먼트 끝에서 배열 밖을 읽는다.

```bash
/usr/bin/python3 -c "
import json, numpy as np
n = np.load('$NPZ')['joint_pos'].shape[0]
seg = json.load(open('$ART/library_meta.json'))['segments']
end = max(e for s, e in seg.values())
print('npz', n, '프레임 · meta 마지막 end', end); assert n >= end, '라이브러리 npz 가 meta 보다 짧다'"
```

### 3-4) 품질 게이트

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH ~/miniconda3/envs/g1deploy/bin/python \
    ~/para_pipeline/check_motion.py "$NPZ"
echo "exit $?"        # 0 이어야 학습으로 간다
```

NaN·관절한계 초과(>0.5° 가 5% 프레임 이상)·골반 < 0.40 m·골반 > 1.20 m·관절속도 > 40 rad/s 중 하나면 exit 1.
**이 게이트는 발 미끄러짐을 보지 않는다.** 회전 클립의 발 미끄러짐은 2장 이후 GMR 결과 영상을 눈으로 확인한다
(`gvhmr_to_robot.py --record_video`).

---

## 4. 학습 → 정책 내보내기

**태스크는 `Tracking-Flat-G1-Wo-State-Estimation-Delay-v0` 하나뿐이다.** 기본 `Tracking-Flat-G1-v0`(관측 160차원)은
world 위치·선속도를 요구해 실기에 올릴 수 없고, 모션 서버가 160차원 정책을 거부한다.
`action_rate -0.1`·지연 0~60 ms 는 파라파라 실기 발산(2026-10-05, 다리 목표가 20 ms 마다 ±100° 튐)을 고친 설정이다.

```bash
nvidia-smi        # 다른 GPU 작업이 없을 때만
( cd ~/whole_body_tracking && ~/bin/isaaclab_run.sh -p scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-Wo-State-Estimation-Delay-v0 \
    --motion_file motions/molbwa_library.npz --num_envs 1024 --max_iterations 6000 --headless \
    env.rewards.action_rate_l2.weight=-0.1 env.actions.joint_pos.max_delay_steps=3 \
    "env.events.push_robot.params.velocity_range.x=[-1.0,1.0]" \
    "env.events.push_robot.params.velocity_range.y=[-1.0,1.0]" ) \
    > "$ART/train.log" 2>&1

RUN=$(ls -dt ~/whole_body_tracking/logs/rsl_rl/g1_flat/*/ | head -1)
CKPT=$(ls -t "$RUN"model_*.pt | head -1); echo "$CKPT"
# 설정이 실제로 먹었는지 (run_smooth.sh 와 같은 확인)
grep -A3 'action_rate_l2:' "$RUN"params/env.yaml | grep weight | head -1
grep max_delay_steps "$RUN"params/env.yaml | head -1
grep -A8 "push_robot:" "$RUN"params/env.yaml | grep -A2 "x:"   # [-1.0, 1.0] 이어야 한다 (기본 ±0.5)
grep -oE "Mean reward:[[:space:]]*[-0-9.]+" "$ART/train.log" | tail -1
```

라이브러리 60~80초 · 6000 iter 기준 수 시간. 촬영 직후 바로 시작해야 일정 안에 든다.

```bash
( cd "$REPO/g1/deploy" && env -u PYTHONPATH ~/miniconda3/envs/isaaclab/bin/python \
    export_policy_npz.py --ckpt "$CKPT" --out "$ART/policy_molbwa.npz" \
    --title "molbwa" --note "데모2 G1 정서 교감 · 154차원 · action_rate -0.1 · 지연 0~60ms" )
```

내보낼 때 torch 경로와 numpy 경로를 200표본 대조한다. **이 대조가 통과하지 않으면 실기에 올리지 않는다.**
(`g1/deploy/catalog.json` 이 생긴다 — 로컬 카탈로그이므로 커밋하지 않는다.)

---

## 5. sim2sim (MuJoCo, 로봇 없음)

### 5-0) 학습 전 서버 스모크 (정책 없이도 가능)

기존 제로투 정책 + 서 있기만 하는 정지 라이브러리로 서버 경로(세그먼트 전환·yaw 재정렬·UDP·이벤트 표·종료)를 먼저 확인한다.

```bash
python3 "$REPO/g1/tools/make_static_library.py" --out /tmp/g1_static
cd "$REPO/g1/deploy"
env -u PYTHONPATH -u LD_LIBRARY_PATH ~/miniconda3/envs/g1deploy/bin/python g1_motion_server.py \
    --policy ~/g1_dance_deploy/policy_final.npz --motion /tmp/g1_static/static_library.npz \
    --library_meta /tmp/g1_static/static_library_meta.json --backend mujoco --fake_events 4 --hold_sec 2
```
합격: exit 0, "판정: 서 있음", 이벤트 표의 heading 오차 ≈ −bin (회전 클립이 없으니 안 도는 게 정상).
2026-10-08 실측: 34.9 s 완주, 골반 최소 0.753 m, 오차 −45→+42.8° · +90→−91.2° · 0→0.0°.

### 5-1) 학습한 정책으로

```bash
cd "$REPO/g1/deploy"
env -u PYTHONPATH -u LD_LIBRARY_PATH ~/miniconda3/envs/g1deploy/bin/python g1_motion_server.py \
    --policy "$ART/policy_molbwa.npz" --motion "$NPZ" --library_meta "$ART/library_meta.json" \
    --backend mujoco --fake_events 10 --hold_sec 5 \
    --log_csv "$ART/sim2sim_events10.csv" | tee "$ART/sim2sim_events10.log"
```

- 6초 간격으로 무작위 part/bin `act` 10회를 내부 주입하고, 마지막 동작이 idle 로 돌아오면 끝난다(`--fake_seed` 로 고정).
  **mujoco 전용** — `--backend dds` 에 주면 서버가 거부한다(실기에 가짜 명령을 넣지 않는다).
- `--hold_sec 5` 를 빼면 이벤트가 끝난 뒤 유지 단계 300초를 그대로 기다린다.
- `--xml` 은 기본값(`G1_MUJOCO_XML` 환경변수)을 쓴다. `g1_29dof.xml` 을 주면 스크립트가 바닥 있는 `scene_29dof.xml` 로 바꿔 연다.
- 눈으로 보려면 `--viewer`, 영상은 `--sim_video $ART/sim2sim.mp4`.

**합격 기준** (발표 표에 그대로 넣는다):

| 지표 | 기준 | 어디서 읽나 |
|---|---|---|
| 시나리오 완주 | **10/10 서 있음**, 골반 최소 > 0.6 m | 요약 `골반 높이 최소` |
| 첫 반응 시간 | **≤ 1.5 s** | 이벤트 표의 act 수신 → 세그먼트 첫 프레임 지연 |
| 회전 후 heading 오차 | 기록 (bin 양자화로 ±22.5° 안 기대) | 이벤트 표 |

- 권장: 시드 하나로 끝내지 말고 `--fake_seed 1`, `--fake_seed 2` 로도 돌려 본다(이벤트 순서가 바뀐다).
- sim2sim 의 첫 반응 시간은 **서버 내부 지연만**이다. 실기에서는 dwell 확정 → UDP(와이파이) 지연이 더해진다.
- 넘어지면 실기도 넘어진다. 이 단계를 건너뛰고 8장으로 가지 않는다.

### 5-2) 밀어도 버티는가 (2026-10-08 요구: 사람이 밀어도 그 자리에서 선 자세 유지)

```bash
for N in 100 150 200 250; do
  env -u PYTHONPATH -u LD_LIBRARY_PATH ~/miniconda3/envs/g1deploy/bin/python g1_motion_server.py \
      --policy "$ART/policy_molbwa.npz" --motion "$NPZ" --library_meta "$ART/library_meta.json" \
      --backend mujoco --run_sec 30 --hold_sec 1 --sim_push $N --sim_push_every 4 \
      --log_csv "$ART/push_$N.csv" | tee "$ART/push_$N.log" | grep -E "판정|밀기|중단"
done
```
idle 중에만 4초마다 골반을 0.2초 민다(앞→왼→뒤→오른). 요약에 `자리 이탈 최대 (cm)` 가 나온다.

| 지표 | 기준 |
|---|---|
| 넘어지지 않는 최대 힘 | 기록. 최소 150 N 은 서 있어야 한다(사람이 툭 미는 정도) |
| 자리 이탈 | 기록. 위치 관측이 없는 정책이라 밀리면 원위치로 돌아오지 못한다 — 작을수록 좋다 |

**⚠ 토크 가드와 충돌한다.** 기존 가드(1초 평균 토크 > 50% → 관절 풀기)는 "하네스를 너무 조여 발버둥" 을 잡으려고
만든 것인데, 사람이 밀어 버틸 때도 토크가 올라 **가드가 로봇을 주저앉힌다.** 2026-10-08 실측: 춤 정책 + 정지 자세에
150 N 을 미니 51% 로 가드 발동 → 골반 0.086 m. 학습한 정책으로 위 시험을 돌린 뒤, 버틴 경우들의 `push_*.csv` 에서
1초창 평균 토크 최대를 재고 **그보다 위로 `--hot_frac` 를 정한다**(그래도 하네스 과조임 시험 64~84% 와 겹치는지 확인).
기존 춤 정책 결과는 기준이 아니다 — 정지 자세를 학습한 적이 없는 정책이라 150 N 에서도 넘어졌다.

**"안 움직이고" 버틸 수 있는 한계는 물리로 정해진다.** 0.2초 밀기의 충격량 F·0.2 를 로봇 35 kg 으로 나누면 무게중심
속도 변화 Δv, 발을 떼지 않고 버티려면 capture point 이동 Δv·√(h/g) (h ≈ 0.7 m) 이 발바닥 안이어야 한다.

| 단발 밀기 | Δv | capture point 이동 | 발 안 떼고 |
|---|---|---|---|
| 50 N | 0.29 m/s | ≈ 7.6 cm | 가능 |
| 100 N | 0.57 m/s | ≈ 15 cm | 불가 — 디뎌야 한다 |

2026-10-08 실측 (임시 대기 정책, 2단계 = ±0.5 1000 iter → ±0.8 1000 iter, MuJoCo, 가드 0.95 로 분리 측정):
50 N 은 4방향 9회 연속에도 서 있음 · 자리 이탈 7.3 cm, 100 N 은 단발에도 넘어짐. 정상 서기 1초창 토크 최대 33%, 50 N 밀기 중 38%.
→ "밀어도 안 움직임" 은 50 N 급(툭 치기)까지. 더 센 밀기는 **디뎌서 안 넘어지기**를 학습해야 한다(3단계 학습 중).

**3단계 후 최종 임시 대기 정책(4997 iter, 2026-10-09)** — `~/molbwa_g1/policy_idle_4997.npz`:

| 시험 | 결과 | 1초창 토크 최대 |
|---|---|---|
| 가만히 서기 15 s | 서 있음, 골반 최소 0.754 m | 16% |
| 단발 50 / 100 / 150 N | 서 있음, 이탈 3.6 / 9.4 / 42.7 cm | — |
| 단발 200 N | 넘어짐 | — |
| 반복 100 N × 7회 (4방향) | 서 있음, 이탈 64.6 cm | **54%** |
| 반복 150 N | 넘어짐 | 84% |

같은 시험에서 실기 검증된 제로투 춤 정책: 단발 100 N 서 있음 / 200 N 넘어짐 → 시험 장치 정상, 대기 정책이 같은 수준.
**가드 결정 필요:** 기본 `--hot_frac 0.5` 는 반복 100 N(54%)에서 버티는 로봇을 주저앉힌다. 권장 0.6 (서기 16%·반복 100 N 54% 위,
반복 150 N 84% 아래). 하네스 과조임 검출 목적과 겹치므로 실기 전에 정하고, 정해지면 서버 기본값을 바꾼다.

측정 주의(2026-10-08 수정): 밀기는 재생 루프의 idle 에서만 건다. 처음 구현은 유지·카운트다운 단계에서도 밀어서
100 N 이상 결과가 오염됐다 — 그 숫자는 쓰지 않는다. 요약의 `재생 루프 골반 최소` 와 `자리 이탈` 이 루프 구간 값이다.

---

## 6. YOLO 부위 데이터 (`g1_face` / `g1_hand` / `g1_torso`)

### 6-1) 수집

글래스 씬 영상이 나오는 상태(`run_gaze_on_scene.sh`)에서, G1 을 여러 각도로 보며 저장한다.

```bash
cd "$REPO"
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 g1/tools/capture_frames.py --scene-topic /pc/camera/left/compressed \
    --out "$ART/yolo_raw/$(date +%m%d_%H%M)" --every 10
# Ctrl-C 로 끝내면 저장 장수를 찍는다
```

- 압축 바이트를 그대로 저장한다 — 브리지가 YOLO 에 넣는 것과 같은 원본 영상(회전·반전 없음).
- 각도 분포: **정면 / 좌우 45° / 좌우 측면 / 뒤**, 높이 **위·아래**(앉은 사람·선 사람), 거리 1~3 m.
  부위 판정 성공률을 정면 / 45° / 측면에서 재므로 그 셋은 반드시 충분히.
- 실제 데모 장소의 조명·배경에서도 한 번 찍는다. 수백 장(최소 300장) 목표.

### 6-2) 라벨링

외부 도구(CVAT · Label Studio · Roboflow 등)로 YOLO 형식 박스를 단다. 클래스 순서 고정:
`0 g1_face`, `1 g1_hand`, `2 g1_torso`.
- `g1_hand` = 팔뚝 끝~손. `g1_torso` = 몸통(손 박스가 몸통 박스 안에 들어가도 된다 — 판정은 hand > face > torso).
- 가려진 부위는 보이는 만큼만. 안 보이면 달지 않는다.

### 6-3) 학습

`~/yolo-env` 는 시스템 numpy 와 torch 충돌을 피하려고 분리한 환경이다(`gaze_hri/yolo_worker.py` 가 이 파이썬으로 뜬다).
없으면 먼저 만든다. GPU 를 쓰므로 다른 GPU 작업이 없을 때.

```bash
# data.yaml: path / train / val / names: [g1_face, g1_hand, g1_torso]
~/yolo-env/bin/yolo detect train data="$ART/yolo/data.yaml" model=yolo11s.pt imgsz=640 epochs=100 \
    project="$ART/yolo/runs" name=g1_parts
cp "$ART/yolo/runs/g1_parts/weights/best.pt" ~/yolo-env/g1_parts.pt   # g1/config.yaml 의 yolo.model
```

val mAP 만 보지 않는다 — 7장 뒤 실제 응시로 부위별 20회 판정 성공률을 잰다(정면 / 45° / 측면).

---

## 7. 태그 (몸통 4면 번들)

- tag36h11, 한 변 **0.10 m**, id **10 앞 / 11 왼 / 12 오른 / 13 뒤** (G1 기준 왼쪽·오른쪽).
- 몸통 표면에 평평하게 붙인다. 팔이 가리지 않는 높이, 하네스 줄과 겹치지 않게.
- `g1/config.yaml` 의 `anchor:` pos 는 **자리값**(앞 [0.10,0,0.20] 등)이다. 붙인 뒤 **torso_link 기준 태그 중심 위치를 실측**해 교체한다
  (+x 정면, +y 왼쪽, +z 위). 각 면의 right/up 축도 실제 붙인 방향과 맞는지 본다 — 축이 거울상이면 PnP 가 안 풀린다
  (`arm/README.md` 2026-08-19 표 3번).
- 확인: 노드 없이 브리지만 띄우고 55057 을 직접 받아, 사용자가 정면 / 왼쪽 / 뒤 / 오른쪽에 섰을 때
  `bearing_deg` 가 0 / 90 / 180 / −90 근처인지 본다.

```bash
cd "$REPO"
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV; source /opt/ros/jazzy/setup.bash
python3 -u g1/g1_gaze_bridge.py --config g1/config.yaml --scene-topic /pc/camera/left/compressed \
    --gaze-px-port 55056 --udp-host 127.0.0.1 --udp-port 55057 &
python3 -c "
import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('127.0.0.1', 55057))
while True: print(s.recv(4096).decode())"
```

태그가 한 장만 보여도 bearing 이 나온다(`min_tags_for_latch: 1`). 전부 안 보이면 `bearing_deg: null` → 회전 없이 동작만.

---

## 8. 실기

### 8-0) 실행 전 관문 — 하나라도 아니면 중단

- [ ] **Jetson 연결 복구됨** (2026-10-05 이후 ping 실패). SSH 가 되고 `uname -m` = `aarch64`
- [ ] **하네스** 설치, **줄이 처져 있다**(조이면 정책이 발버둥친다 — 발에 체중의 절반 이상이 실려야 한다). 발바닥이 바닥에 온전히 닿는다
- [ ] **매트**
- [ ] **비상정지 담당**이 따로 있다(시연자와 겸하지 않는다). 전원 차단 스위치가 그 사람 손에 닿는다
- [ ] 주변 2 m 안에 사람·물건 없음
- [ ] 5장 sim2sim 합격

**역할 고정** (시행 내내 바꾸지 않는다):

| 역할 | 하는 일 |
|---|---|
| 시연자 | 글래스 착용, G1 의 한 부위를 **3초** 응시(진행률은 `/g1/dwell_progress`). 로봇에 손대지 않는다 |
| 비상정지 담당 | 로봇 옆. SSH 비상정지 창 + 전원 차단 스위치. **종료 때 로봇을 받친다** |
| 설명 담당 | 데스크탑·관객 설명. `/g1/state`·로그 확인 |

### 8-1) Jetson 찾기·파일 올리기

Jetson 무선 IP 는 **DHCP 로 바뀐다**(192.168.50.119 → .118 이었다). 박지 말고 매번 찾는다 — ARP 이웃 → 공유기 DHCP 목록.

```bash
ip neigh | grep 192.168.50.             # 후보. FAILED/INCOMPLETE 는 건너뛴다
J=unitree@192.168.50.<x>                # 계정은 unitree (ubuntu 아님)
ssh $J 'uname -m; python3 --version'    # aarch64, Python 3.8.x
DESK=$(ip -4 -o addr | awk '$4 ~ /^192\.168\.50\./ {sub(/\/.*/, "", $4); print $4; exit}')   # 데스크탑 IP (상태 수신처)

ssh $J 'mkdir -p ~/molbwa_g1'
scp "$REPO/g1/deploy/g1_motion_server.py" "$REPO/g1/deploy/segment_player.py" "$REPO/g1/g1_protocol.py" \
    "$REPO/g1/deploy/g1_tracking_policy_meta.json" \
    "$ART/library_meta.json" "$NPZ" "$ART/policy_molbwa.npz" $J:~/molbwa_g1/
```

Jetson 에는 numpy + cyclonedds 가 이미 있고 torch·MuJoCo 는 필요 없다. SDK 는 `/home/unitree/unitree_sdk2_python` 을
스크립트가 후보 경로로 찾는다. 내부망 인터페이스 이름도 바뀐다(`eth0` → `eth1`) — `--iface auto`(기본값)가 192.168.123.x 주소로 찾는다.
**`--iface` 를 이름으로 주지 않는다.**

### 8-2) R&D 모드로 내리기 — CheckMode 의 name 이 빈 값이어야 한다

**이 순간부터 로봇은 스스로 서지 못한다. 하네스가 받친다.**

```bash
ssh $J
cd ~/molbwa_g1 && python3 - <<'EOF'
import sys, subprocess
sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
iface = next(f.split()[1] for f in out.splitlines() if " 192.168.123." in f)
ChannelFactoryInitialize(0, iface)
m = MotionSwitcherClient(); m.SetTimeout(5.0); m.Init()
print("before", m.CheckMode())
m.ReleaseMode()
print("after ", m.CheckMode())       # name 이 '' 여야 한다. 'ai' 면 아직 고수준
EOF
```

R&D 판정의 유일한 근거는 `CheckMode()` 의 name 이다. "`ps` 에 loco 가 없다", "lowstate 가 온다"는 근거가 아니다.
모션 서버도 `--arm` 시 name 이 비어 있지 않으면 발행을 거부한다.

### 8-3) probe → dry-run → 발행

전부 **Jetson 에서**, `~/molbwa_g1` 에서 시스템 `python3` 로 돌린다. 데스크탑에서 정책을 돌리지 않는다.

```bash
A="--policy policy_molbwa.npz --motion molbwa_library.npz --library_meta library_meta.json --state_host $DESK"

# (1) probe — lowstate 만 받는다. 발행자를 만들지 않아 실수로 쏠 수 없다
python3 g1_motion_server.py $A --probe 5
#     통과: 수신 > 100 Hz, mode_pr == 0, 자세·토크 정상

# (2) dry-run — 발행 없음. 데스크탑 노드를 띄워 놓고 /g1/state 가 idle 로 오는지(55071 경로) 같이 본다
python3 g1_motion_server.py $A --run_sec 30
#     유효한 숫자는 제어주기뿐: p99 < 60 ms. 추종 오차·관절속도·토크는 자기참조 루프라 무의미 — 인용 금지

# (3) 발행 — 8-0 관문을 한 번 더 소리 내어 확인한 뒤
python3 g1_motion_server.py $A --arm --log_csv ~/molbwa_g1/arm_$(date +%m%d_%H%M).csv
```

$DESK 는 데스크탑에서 정한 값을 Jetson 셸에 그대로 적어 넣는다.

- **세기 인자는 없다.** `--gain_ramp`/`--motion_scale` 로 약하게 시작하면 넘어지거나 토크가 오른다(플레이북 5장). 모션 서버에서 뺐다.
- **보간 진입 없음.** 정책을 첫 스텝부터 켠다(`--enter_sec 0` 기본값 그대로). 첫 1초에 토크 75~80%·616°/s 가 나오는데, 그건 하네스가 받는다.
- `--imu_frame` 은 기본값 `pelvis`(실측). 시작 자세는 무관하다(늘어진 자세·발 들림 포함).
- 명령이 끊겨도 idle 을 계속 재생한다. 토크 가드(1초 창 평균 > 50%)가 걸리면 능동 댐핑 → 하네스가 받는다.
- 서버는 마지막으로 수락한 seq **이하**의 `act` 를 무시한다(와이파이 중복·역순 대비). 노드는 시각 기반 seq 를 쓰지만,
  확정 이벤트가 계속 무시되면(노드 `/g1/event` 는 나가는데 `/g1/state` 가 idle 그대로) **서버와 노드를 둘 다 재시작**한다.

데스크탑 쪽 (시선 → 명령):

```bash
# 터미널 1: 글래스 시선   ~/창종설/run_gaze_on_scene.sh  (시선 픽셀 UDP 55056)
# 터미널 2: 브리지         7장 명령에서 수신 확인용 python 은 빼고 브리지만
# 터미널 3: 상호작용 노드
cd "$REPO/ros2_ws" && source install/setup.bash
ros2 run gaze_hri g1_interaction --ros-args -p jetson_host:=${J#unitree@}
# 터미널 4: 기록 (정서 지표)
ros2 bag record /g1/event /g1/state /g1/dwell_progress -o "$ART/bag_$(date +%m%d_%H%M)"
```

### 8-4) 비상정지·종료

- **비상정지** (비상정지 담당): `ssh $J 'kill -TERM $(cat /tmp/g1_deploy.pid)'` → 약 0.26 s 후 `kp=0 kd=2` 댐핑 3초.
  비상정지는 "붙잡기"가 아니라 "힘 빼기"다 — 로봇이 주저앉는다. 넘어짐을 막는 건 하네스뿐이다.
- 와이파이가 끊기면 SSH 비상정지도 안 간다 → **전원 차단 스위치**가 최후 수단.
- **정상 종료**: `ssh $J 'kill -INT $(cat /tmp/g1_deploy.pid)'` → 유지 단계(기본 300초, 30초마다 경과 출력) → 만료 시 5초 카운트다운 후 관절을 푼다.
  **정책이 꺼지면 로봇은 무조건 주저앉는다(골반 0.73 → 0.15 m). 카운트다운 동안 비상정지 담당이 받친다.**
- `pkill -f` 쓰지 않는다 — 명령줄에 패턴이 있는 자기 셸까지 죽는다.

---

## 9. 현장 체크리스트

전날
- [ ] Jetson 연결 확인 (SSH, `uname -m`), 무선 IP 기록
- [ ] Jetson `~/molbwa_g1` 파일 7개가 데스크탑 것과 같다 (`md5sum` 대조)
- [ ] 5장 sim2sim 합격 로그 보관
- [ ] 태그 4장 부착·`g1/config.yaml` pos 실측값, bearing 4방향 확인
- [ ] `~/yolo-env/g1_parts.pt` 로 부위 판정 성공률 측정 (정면 / 45° / 측면, 부위별 20회)
- [ ] **백업 영상**: sim2sim 영상(`--sim_video`) + 리허설 실기 영상. 현장에서 로봇이 안 되면 이걸 튼다
- [ ] 충전: G1 배터리, 글래스, 노트북

현장
- [ ] 하네스(줄이 처짐)·매트·전원 스위치 위치, 주변 2 m 비움
- [ ] 역할 배치: 시연자 / 비상정지 담당 / 설명 담당
- [ ] 와이파이: 명령·상태만 지난다. 정책은 Jetson 온보드
- [ ] **전용 AP(공유기)** 사용. UDP 55070 에는 인증이 없다 — 같은 와이파이의 누구든 `act` 를 보내 로봇을 움직일 수 있다. 행사장 공용망 금지
- [ ] CheckMode name 빈 값 → probe 통과 → dry-run 제어주기 p99 < 60 ms → 발행
- [ ] `/g1/state` 가 idle 로 오는지(offline 이면 Jetson→데스크탑 55071 경로 확인)
- [ ] rosbag 기록 시작
- [ ] 끝: 정상 종료 → 카운트다운 동안 받치기 → 전원

---

## 부록) 촬영 전 임시 대기 정책 (2026-10-08)

촬영 클립 없이 "편하게 선 자세 + 밀어도 버티기" 를 먼저 학습한다. 촬영 라이브러리가 생기면 이 런에서 `agent.resume=true` 로 이어 학습한다.
```bash
python3 "$REPO/g1/motion/make_idle_csv.py" --out "$ART/idle_synth.csv" --seconds 10     # G1 기본자세, 이름으로 관절 매핑
"$ART/run_csv_to_npz.sh" "$ART/idle_synth.csv" molbwa_idle                               # npz 크기 안정되면 PID 만 종료
"$ART/train_idle.sh" > "$ART/train_idle.log" 2>&1                                        # 512 env · 2000 iter · push ±1.0 m/s
```
