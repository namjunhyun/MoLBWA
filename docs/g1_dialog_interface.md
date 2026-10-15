# 지원이(G1) ↔ 대화 모듈 연결 규약

2026-10-15 · 대상: 대화(CLOVA) 담당 팀원 · 코드: `g1/g1_protocol.py`, `g1/g1_say.py`

대화 모듈은 **UDP JSON 세 가지**만 쓰면 된다. 로봇 SDK·DDS 를 직접 다루지 않는다
(정책이 도는 중에 다른 프로세스가 DDS 를 열면 제어주기가 깨진다 — 실기에서 확인).

## 1. 받는 것: 시선 정보 (데스크탑, UDP 55057)

`g1/g1_gaze_bridge.py` 가 매 프레임 보낸다.

```json
{"t": 1760500000.1, "label": "g1_face", "bearing_deg": -32.0, "obj_bearing_deg": 41.5, "valid": true}
```

| 키 | 뜻 |
|---|---|
| `label` | 시선이 닿은 지원이 부위 `g1_face`/`g1_hand`/`g1_torso` 또는 `null` |
| `bearing_deg` | 지원이 기준 **착용자** 방위(도, + 왼쪽). 태그가 안 보이면 `null` |
| `obj_bearing_deg` | 지원이 기준 **착용자가 보는 점** 방위(도). 거리 1.5 m 가정(config `attend.object_dist_m`) |
| `valid` | 이번 프레임에 시선이 있었는가 |

물체 이름(가족사진·달력 등)은 대화 쪽 YOLO 가 정한다. 같은 프레임의 `obj_bearing_deg` 를 같이 쓰면 된다.

## 2. 보내는 것: 지원이 명령 (젯슨, UDP 55070)

| 명령 | JSON | 하는 일 |
|---|---|---|
| 같이 보기 | `{"seq": S, "cmd": "attend", "obj_bearing_deg": 45, "user_bearing_deg": -30, "gesture": "celebrate"}` | 물체 쪽 회전 → 손바닥 위 가리키기 → 착용자 쪽 회전 → 박수(`celebrate`) 또는 손 흔들기(`wave`) |
| 인사 | `{"seq": S, "cmd": "act", "part": "g1_face", "bearing_deg": -30}` | 착용자 쪽 회전 → 손 흔들기 (`g1_hand` 면 악수) |
| 말하기 | `{"cmd": "say", "path": "/home/unitree/molbwa_g1/tts/x.wav"}` | 지원이 스피커로 WAV 재생 (seq 불필요, 동작과 동시 가능) |

- **방위는 int(도)**, 15° 단위로 반올림돼 실행된다. float·bool 은 거부.
- **seq** 는 이전보다 큰 int (밀리초 시각 `int(time.time()*1000)` 권장). 같은/작은 seq 는 무시(와이파이 중복 대비).
- **동작 중에 온 act/attend 는 무시**된다 — 상태가 `idle` 일 때 보낼 것.
- 말하기는 `g1/g1_say.py` 를 쓰면 WAV 복사 + 명령까지 한 번에:
  ```python
  from g1_say import say
  say("tts_out.wav", jetson="192.168.50.107")   # 젯슨 IP 는 DHCP 라 바뀔 수 있음
  ```
  WAV 는 아무 샘플레이트·채널이어도 되고 서버가 16 kHz mono 로 바꾼다. **(스피커 재생은 실기 미검증)**

## 3. 받는 것: 지원이 상태 (데스크탑, UDP 55071, 5 Hz)

```json
{"state": "idle" | "turning" | "acting", "seq": 1760500000123}
```
웹 패널(`g1_panel.py`)도 같은 포트를 쓰므로 **둘을 동시에 띄우지 말 것**.

## 4. 타이밍 (라이브러리 v10, 시뮬 기준)

| 구간 | 길이 |
|---|---|
| 회전 15° / 45° / 90° / 135° / 180° | 2.8 / 3.6 / 4.4 / 6.0 / 7.6 s |
| 가리키기 (point) | 7.1 s |
| 박수 (celebrate) | 4.5 s |
| 손 흔들기 / 악수 | 4.3 / 5.0 s |

같이 보기 한 번 = 회전 + 가리키기 7.1 s + 회전 + 박수 4.5 s ≈ **18~25 초**.

**권장 흐름** (말과 동작 맞추기):
1. 3초 응시 확정 → 즉시 `attend` 전송 + 동시에 CLOVA 호출 시작 (지원이가 도는 동안 말 생성·TTS 완료).
2. 상태가 두 번째 `turning` → `acting`(박수)로 바뀌는 순간 `say` 전송 → 박수와 함께 "다음 주에 손녀 생일이에요".
3. 상태가 `idle` 로 돌아오면 다음 턴(대답 듣기) 시작.

## 5. 시험 (로봇 없이)

```bash
bash g1/tools/sim_panel.sh          # MuJoCo 지원이 + 패널(:8080), 명령 포트 55070 동일
python3 g1/g1_say.py hello.wav      # 시뮬에선 서버 로그에 "[say] (스피커 없음 …)" 만 찍힘
```
