# G1 교감 데모 통합 인터페이스 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 시선/YOLO로 로봇 얼굴=인사·손=악수·물체=CLOVA 대화를 자동 발동하는 통합 오케스트레이터를 웹 패널에 넣는다.

**Architecture:** 기존 `LabelDwell`(dwell 확정)·`g1_protocol`(명령)·`g1_dialog.Dialog`(대화)를 재사용하고, 새로 ① 대화용 물체 YOLO 검출기와 ② 부위/물체 분기 오케스트레이터를 만든다. 오케스트레이터는 별도 ROS 노드가 아니라 **웹 패널 프로세스 안의 스레드**로 돌려 포트(55071) 충돌을 없앤다. 웹은 `/origin`(기존 전부)·`/demo`(발표용) 2페이지.

**Tech Stack:** Python 3.12 표준 라이브러리(패널·소켓), ultralytics YOLO(`yolo11l.pt`, 기존 `yolo_worker.py` subprocess), 기존 `g1/` 모듈.

**Spec:** `docs/superpowers/specs/2026-10-29-g1-gaze-interaction-integration-design.md`

## Global Constraints

- 모든 사용자 대면 텍스트·주석·로그는 **한국어**.
- 오케스트레이터는 **UDP만** 사용, 젯슨 DDS를 직접 열지 않는다(정책 구동 중 제어주기 깨짐 방지).
- 포트 고정: 시선픽셀 55056, 부위이벤트 55057(`PORT_BRIDGE`), 명령 55070(`PORT_CMD`), state 55071(`PORT_STATE`), 패널 8080.
- `g1_interaction_node`(별도 ROS 노드)와 패널을 **동시에 띄우지 않는다** — 오케스트레이터가 그 역할을 흡수한다.
- 기존 파일 패턴·의존성 유지: 패널은 표준 라이브러리만.

## Review Focus

- **BUSY 중 새 트리거**: 로봇 동작·대화 진행 중 다른 부위/물체를 3초 봐도 새 명령이 나가면 안 됨(동작 겹침). → Task 2 테스트.
- **라벨 전환 중 dwell**: 얼굴 2초 보다 손으로 옮기면 얼굴이 확정되면 안 됨(dwell 리셋). → Task 2, `LabelDwell` 재사용으로 커버하되 전환 시나리오 명시 테스트.
- **부위 vs 물체 동시 후보**: 같은 프레임에 부위(얼굴/손)와 물체가 둘 다 응시 후보면 **부위 우선**. → Task 2 테스트.
- **시선 끊김(valid=false 연속)**: dwell 리셋되고 로봇은 새 명령 없이 마지막 상태 유지. → Task 2 테스트.
- **알 수 없는 YOLO 라벨**: 매핑에 없는 물체(예: person)는 대화 트리거 안 함. → Task 1 테스트.

---

### Task 1: 대화용 물체 검출 + kind 매핑 — `g1/obj_gaze_detector.py`

**Files:**
- Create: `g1/obj_gaze_detector.py`
- Test: `g1/tests/test_obj_gaze_detector.py`

**Interfaces:**
- Consumes: 기존 `ros2_ws/src/gaze_hri/gaze_hri/yolo_worker.py`(subprocess: stdin `[uint32 len][JPEG]`, stdout `{"dets":[[cls,conf,x1,y1,x2,y2]]}`), `g1_gaze_bridge.object_bearing_deg`(방위 재사용) 또는 동형 계산.
- Produces:
  - `OBJ_TO_KIND: dict[str,str]` — YOLO 라벨→대화 kind.
  - `pick_object(dets: list, uv: tuple[float,float] | None) -> tuple[str, list[float]] | None` — 시선픽셀 uv를 포함하는 대화대상 박스 중 최우선을 `(kind, box_xyxy)`로. 없으면 None.
  - `object_bearing(box_xyxy, K, dist_m) -> int` — 박스 중심을 역투영한 torso 방위(도, 정수). (gaze_bridge의 `object_bearing_deg` 로직 재사용.)

**매핑 값(spec §3.3 그대로):**
`OBJ_TO_KIND = {"potted plant":"plant", "vase":"plant", "bottle":"bottle", "cup":"bottle", "cell phone":"phone", "teddy bear":"photo"}`

- [ ] **Step 1: 실패 테스트 작성** — `g1/tests/test_obj_gaze_detector.py`

```python
def test_pick_object_returns_kind_for_gazed_box():
    dets = [["potted plant", 0.9, 100, 100, 200, 200], ["person", 0.8, 0, 0, 640, 480]]
    assert pick_object(dets, (150, 150)) == ("plant", [100, 100, 200, 200])

def test_pick_object_ignores_unmapped_labels():
    dets = [["person", 0.9, 0, 0, 640, 480]]            # 매핑에 없음
    assert pick_object(dets, (320, 240)) is None

def test_pick_object_none_when_gaze_outside_boxes():
    dets = [["vase", 0.9, 100, 100, 200, 200]]
    assert pick_object(dets, (10, 10)) is None

def test_pick_object_none_when_no_gaze():
    assert pick_object([["cup", 0.9, 0, 0, 640, 480]], None) is None
```

- [ ] **Step 2: 테스트 실패 확인** — Run: `python3 -m pytest g1/tests/test_obj_gaze_detector.py -v` · Expected: FAIL(import/NameError)
- [ ] **Step 3: 구현** — `OBJ_TO_KIND`, `pick_object`, `object_bearing`를 `g1/obj_gaze_detector.py`에 작성. `pick_object`은 uv를 포함하고 `OBJ_TO_KIND`에 있는 박스 중 conf 최고를 고른다. 박스 포함 판정은 `pick_part`(gaze_bridge)와 동일 방식.
- [ ] **Step 4: 테스트 통과 확인** — Run: 위 명령 · Expected: PASS
- [ ] **Step 5: 커밋** — `git add g1/obj_gaze_detector.py g1/tests/test_obj_gaze_detector.py && git commit -m "feat(g1): 대화용 물체 검출·kind 매핑"`

주: subprocess로 `yolo_worker.py`를 띄워 씬영상 프레임을 넣고 dets를 받는 **런타임 래퍼**(`ObjectDetector` 클래스: start/feed(jpeg)->dets/stop)도 이 파일에 둔다. 순수함수 `pick_object`/`object_bearing`만 단위 테스트하고, 래퍼는 Task 3 통합에서 실물 확인(모델 로딩은 테스트에서 제외).

---

### Task 2: 오케스트레이터 상태기계 — `g1/g1_orchestrator.py`

**Files:**
- Create: `g1/g1_orchestrator.py`
- Test: `g1/tests/test_g1_orchestrator.py`

**Interfaces:**
- Consumes: `g1_dwell.LabelDwell(dwell_time, min_ratio, cooldown)` · `.update(t, label|None)->확정라벨|None`; `g1_protocol.plan`, `plan_attend`, `encode`, `PART_TO_SEGMENT`(`{g1_face,g1_hand,g1_torso}`), `PORT_CMD`; `obj_gaze_detector.OBJ_TO_KIND`.
- Produces:
  - `class Orchestrator` with:
    - `__init__(self, send_cmd, start_dialog, is_busy, dwell_time=3.0)` — `send_cmd(msg: dict)` UDP 명령 전송 콜백, `start_dialog(kind: str, bearing_deg: int)` 대화 시작 콜백, `is_busy() -> bool` 로봇/대화 중 판정.
    - `feed_part(t: float, label: str | None, bearing_deg: int | None)` — 55057 부위 이벤트 1개 투입.
    - `feed_object(t: float, kind: str | None, bearing_deg: int | None)` — 물체 이벤트 1개 투입.
    - 내부: 부위 dwell·물체 dwell 각각 `LabelDwell`. 부위 우선.

**동작 규칙(spec §3.2):**
- 부위 확정(얼굴/손) → `send_cmd({"seq":n,"cmd":"act","part":label,"bearing_deg":b})`.
- 물체 확정 → `start_dialog(kind, b)` (호출자가 attend+Dialog.start(with_attend=True) 수행).
- `is_busy()`가 True면 어떤 확정도 명령/대화를 트리거하지 않는다.
- 같은 틱에 부위·물체 둘 다 확정 가능하면 **부위 먼저**, 물체는 버린다.
- seq는 매 전송 증가(`int(time.time()*1000)` 또는 단조 증가 카운터).

- [ ] **Step 1: 실패 테스트 작성** — `g1/tests/test_g1_orchestrator.py` (가짜 콜백으로 호출 기록)

```python
def test_face_dwell_sends_act_wave():
    sent=[]; orc=Orchestrator(sent.append, lambda k,b: sent.append(("dlg",k)), lambda: False, dwell_time=1.0)
    for t in _ticks(0, 1.2, 0.1): orc.feed_part(t, "g1_face", -30)
    assert any(m.get("cmd")=="act" and m["part"]=="g1_face" for m in sent if isinstance(m,dict))

def test_hand_dwell_sends_act_handshake():
    ...  # g1_hand → act part g1_hand

def test_object_dwell_starts_dialog():
    sent=[]; orc=Orchestrator(lambda m: sent.append(m), lambda k,b: sent.append(("dlg",k,b)), lambda: False, dwell_time=1.0)
    for t in _ticks(0,1.2,0.1): orc.feed_object(t, "plant", 45)
    assert ("dlg","plant",45) in sent

def test_busy_blocks_all_triggers():
    sent=[]; orc=Orchestrator(sent.append, lambda k,b: sent.append(("dlg",k)), lambda: True, dwell_time=1.0)
    for t in _ticks(0,1.2,0.1): orc.feed_part(t,"g1_face",-30); orc.feed_object(t,"plant",45)
    assert sent == []

def test_part_priority_over_object():
    # 같은 구간에 얼굴·물체 둘 다 dwell → act(face)만, dialog 없음
    ...

def test_label_switch_resets_dwell():
    # 얼굴 0.6초 보다 손으로 전환 → 얼굴 확정 안 됨(LabelDwell min_ratio)
    ...

def test_invalid_frames_reset():
    # 중간에 label=None 연속 → dwell 리셋, 명령 없음
    ...
```

- [ ] **Step 2: 테스트 실패 확인** — Run: `python3 -m pytest g1/tests/test_g1_orchestrator.py -v` · Expected: FAIL
- [ ] **Step 3: 구현** — `Orchestrator`를 `g1/g1_orchestrator.py`에 작성. 부위/물체 각각 `LabelDwell`, 매 feed에서 `is_busy()` 가드 후 확정 시 콜백. 부위 우선은 같은 틱 처리 순서로(부위 feed가 물체보다 먼저 확정되면 물체 확정 무시 플래그).
- [ ] **Step 4: 테스트 통과 확인** — Run: 위 명령 · Expected: PASS
- [ ] **Step 5: 커밋** — `git add g1/g1_orchestrator.py g1/tests/test_g1_orchestrator.py && git commit -m "feat(g1): 시선 부위/물체 분기 오케스트레이터"`

---

### Task 3: 웹 패널 통합 — `/origin`·`/demo` + 자동 교감 스레드

**Files:**
- Modify: `g1/g1_panel.py`
- Test: `g1/tests/test_g1_panel_routes.py`

**Interfaces:**
- Consumes: Task 1 `ObjectDetector`·`object_bearing`, Task 2 `Orchestrator`; 기존 패널의 `dialog`(g1_dialog.Dialog), `tx.sendto`(55070), state 수신(55071), `dialog_attend`.
- Produces: HTTP 라우트 `/origin`(기존 PAGE), `/demo`(새 발표 페이지), `/auto`(자동 교감 ON/OFF POST), `/state`에 `auto_on`·현재 교감 상태 문구 추가.

**통합 방식:**
- 패널 시작 시 55057 부위 이벤트 수신 스레드 + (ObjectDetector 통한) 물체 이벤트를 `Orchestrator`에 투입하는 스레드. `Orchestrator`의 콜백:
  - `send_cmd` = 기존 명령 전송(`tx.sendto(encode(dict(msg,seq=seq)), (jetson, PORT_CMD))`).
  - `start_dialog(kind,b)` = `dialog_attend` 방위 세팅 후 `dialog.start(mode, with_attend=True, kind=kind)`.
  - `is_busy` = 로봇 state가 idle이 아니거나 `dialog` 진행 중.
- `/auto` ON일 때만 오케스트레이터가 명령을 낸다(OFF면 feed해도 무시). 수동 버튼은 `/origin`에서 auto OFF일 때만 동작.
- `/demo` HTML: 큰 상태 문구(`/state`의 교감 상태), 대화 말풍선(기존 대화로그 재사용), 자동 ON 표시, 비상정지 버튼. 세부 버튼·SSH 없음.

- [ ] **Step 1: 실패 테스트 작성** — `g1/tests/test_g1_panel_routes.py` (핸들러 라우팅 단위 테스트; 소켓/젯슨은 가짜)

```python
def test_demo_route_served():          # GET /demo → 200, 발표 HTML(상태 문구 포함)
def test_origin_route_served():        # GET /origin → 기존 PAGE
def test_auto_toggle_endpoint():       # POST /auto {on:true} → auto_on True
def test_state_includes_auto_flag():   # GET /state JSON에 auto_on 키
```

- [ ] **Step 2: 테스트 실패 확인** — Run: `python3 -m pytest g1/tests/test_g1_panel_routes.py -v` · Expected: FAIL
- [ ] **Step 3: 구현** — `do_GET`에 `/origin`·`/demo` 분기 추가(기존 `/`는 `/origin`으로 리다이렉트 또는 동일), `do_POST`에 `/auto`, `/state`에 `auto_on`·상태문구. 오케스트레이터 스레드 기동/정지. `DEMO_PAGE` HTML 상수 추가.
- [ ] **Step 4: 테스트 통과 확인** — Run: 위 명령 · Expected: PASS
- [ ] **Step 5: 수동 통합 확인(로봇 없이)** — `fake` 젯슨/relay로 패널 기동, `/demo`·`/origin` 브라우저 확인, `/auto` 토글, 가짜 55057 이벤트로 오케스트레이터가 55070 명령 내는지 로그 확인. (스크립트는 `g1/tests/`에 둔다.)
- [ ] **Step 6: 커밋** — `git add g1/g1_panel.py g1/tests/test_g1_panel_routes.py && git commit -m "feat(g1): 패널 /origin·/demo 분리 + 자동 교감 오케스트레이터 통합"`

---

## 재사용(새로 안 만듦)

- dwell 확정: `g1_dwell.LabelDwell` (그대로)
- 명령 프로토콜: `g1_protocol.plan/plan_attend/encode/PART_TO_SEGMENT` (그대로)
- 대화: `g1_dialog.Dialog.start(with_attend=True, kind=...)` (그대로)
- 발화: `g1_say.say` (그대로)
- 부위 라벨+방위: `g1_gaze_bridge` (그대로, 55057 발행)
- 물체 YOLO 실행기: `yolo_worker.py` (그대로, subprocess)
