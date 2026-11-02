"""시선 안경 없이 쓰는 데모 2 조작 패널 (웹). 각도와 동작을 골라 g1_motion_server 에 act 명령을 보낸다.

    python3 g1/g1_panel.py [--jetson 127.0.0.1] [--port 8080]   → 브라우저에서 http://<이 PC>:8080

아무것도 안 누르면 서버는 idle(서 있기) 세그먼트를 계속 재생한다 — 패널은 명령만 보낸다.
g1_interaction_node 를 대신하므로 같이 띄우지 말 것(상태 포트 55071 이 겹친다). 의존성 없음(표준 라이브러리).
"""
import argparse
import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from g1_protocol import (PART_TO_SEGMENT, PORT_BRIDGE, PORT_CMD, PORT_STATE, _quant, _wrap, decode, encode, plan,
                         plan_attend)
from g1_orchestrator import Orchestrator
import g1_dialog
import g1_say
import g1_tts

AP = argparse.ArgumentParser()
AP.add_argument("--jetson", default="127.0.0.1", help="g1_motion_server 가 도는 곳 (sim 이면 127.0.0.1)")
AP.add_argument("--port", type=int, default=8080)
AP.add_argument("--mic", default="default", help="대화 마이크 ALSA 장치(arecord -L). 핀마이크를 데스크탑에 꽂아 쓴다")
AP.add_argument("--tts", choices=["edge", "clova"], default="edge",
                help="목소리: edge(무료, 기본) / clova(CLOVA Voice Premium — 월 기본요금 9만원, 앱에서 체크 필요)")
AP.add_argument("--volume", type=int, default=100, help="정책 꺼졌을 때 직접 재생 음량(0~100)")
AP.add_argument("--ssh", default="", help="실기: unitree@<젯슨 IP>. 주면 비상정지·정지·모드·정책 ON 버튼이 켜진다 (젯슨 ~/molbwa_g1/g1_ctl.py)")
args, _ = AP.parse_known_args()   # 테스트에서 import 만 해도 안 죽게 — 모르는 인자는 흘린다
os.environ["TTS_ENGINE"] = args.tts

tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state = {"state": "?", "seq": 0, "t": 0.0}
last = {"seq": 0}


def listen_state():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", PORT_STATE))
    while True:
        m = decode(rx.recvfrom(4096)[0])
        if m:
            state.update(m, t=time.time())


STOP_CMDS = {"damp", "stop", "release", "shutdown"}   # 멈춤/정책 내림 성격 → 자동 교감 해제
CTL = {"damp", "stop", "release", "highlevel", "start", "start_dry", "handover", "handback", "shutdown", "status"}
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "-o", "ControlMaster=auto",
       "-o", "ControlPath=/tmp/g1_panel_%r@%h", "-o", "ControlPersist=600"]


def ctl(cmd):
    """젯슨에서 g1_ctl.py <cmd>. SSH 연결을 재사용해 비상정지가 빠르게 간다."""
    if not args.ssh:
        return "실기 연결 없음(--ssh 미지정)"
    try:
        r = subprocess.run(SSH + [args.ssh, "cd ~/molbwa_g1 && python3 g1_ctl.py " + cmd],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=15)
        return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"응답 없음(코드 {r.returncode})"
    except subprocess.TimeoutExpired:
        return "시간 초과 — 와이파이 확인. 안 되면 전원 차단"


SAY = {"last": "", "direct": False}


def _user():
    return args.ssh.split("@")[0] if "@" in args.ssh else "unitree"


def server_up():
    """모션 서버(정책)가 상태를 보내고 있나. 꺼져 있으면 말은 g1_play_pcm 직접 재생으로 간다."""
    return time.time() - state["t"] < 1.5


def speak(wav):
    """정책 중이면 모션 서버(UDP say), 아니면 젯슨 g1_play_pcm 직접. 반환 = 화면에 띄울 메시지."""
    dur = g1_tts.duration(wav)
    if _local():
        subprocess.Popen(["aplay", "-q", wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   # 시뮬: 데스크탑 스피커
        if server_up():
            g1_say.say(wav, args.jetson)
        return f"시뮬 재생 {dur:.1f}s (데스크탑 스피커)"
    if server_up():
        g1_say.say(wav, args.jetson, _user())
        return f"지원이 스피커로 보냄 {dur:.1f}s (모션 서버)"
    threading.Thread(target=_direct, args=(wav,), daemon=True).start()
    return f"지원이 스피커로 보냄 {dur:.1f}s (정책 꺼짐 → 직접 재생)"


def _direct(wav):
    SAY["direct"] = True
    try:
        SAY["last"] = "직접 재생: " + g1_say.say_direct(wav, args.jetson, _user(), volume=args.volume)
    except Exception as e:
        SAY["last"] = f"직접 재생 실패: {e}"
    finally:
        SAY["direct"] = False


def _local():
    return args.jetson in ("127.0.0.1", "localhost")


def send_attend(obj, user, gesture="celebrate", say_text=""):
    """같이 보기 명령. say_text 가 있으면 합성·업로드 후 마지막 동작(박수·손 흔들기) 첫 프레임에 서버가 말한다.
    반환 (보낸 seq, 세그먼트 목록, say_at 또는 None). 잘못된 각도는 ValueError."""
    msg = {"cmd": "attend", "obj_bearing_deg": int(obj), "user_bearing_deg": int(user), "gesture": gesture}
    segs = plan_attend(msg["obj_bearing_deg"], msg["user_bearing_deg"], gesture)
    if str(say_text).strip():
        msg["say_path"] = g1_say.say(g1_tts.synth(say_text), args.jetson, _user(), play=False)
        msg["say_at"] = segs[-1]
    return send(msg), segs, msg.get("say_at")


def send(msg):
    last["seq"] = seq = max(last["seq"] + 1, state["seq"] + 1, int(time.time() * 1000))   # g1_dwell.next_seq 와 같은 규칙
    tx.sendto(encode(dict(msg, seq=seq)), (args.jetson, PORT_CMD))
    return seq


def say_text(text):
    """대화 모듈용: 합성 → 지원이 스피커. 반환 = 재생 길이(초)."""
    wav = g1_tts.synth(text)
    SAY["last"] = speak(wav)
    return g1_tts.duration(wav)


def robot_busy():
    """동작 중이거나 말하는 중 — 대화 모듈이 이 동안 듣지 않는다."""
    return SAY["direct"] or (server_up() and (state.get("state") != "idle" or bool(state.get("speaking"))))


DLG = {"bear": (60, -30)}


def dialog_attend(text):
    if not server_up():
        return False                      # 정책 꺼짐 → 동작 없이 말만
    send_attend(DLG["bear"][0], DLG["bear"][1], "look", text)   # 2026-10-31: 꽃 쪽으로 돌아보기만 하고 첫 질문(박수·가리키기 없음)
    _turned(DLG["bear"][0])                        # 꽃 쪽으로 돈 만큼 착용자 방위가 바뀐다
    return True


def _dialog_ended():
    """대화가 "대화해줘서 고마워"로 끝났을 때만 초기화 버튼과 같이 — 응시 누적·재확정 잠금을 비우고 다음 시나리오를 받는다.
    대답 없음·최대 횟수·위험 신호·오류·중지로 끝난 대화는 넘기지 않는다(초기화 버튼으로) (2026-10-31)."""
    if not dialog.thanked:
        print("[auto] 대화 끝 (고마워 아님) — 다음 시나리오로 넘기지 않음. 넘기려면 초기화", flush=True)
        return
    orc.reset()
    if AUTO.get("held_off"):              # 비상정지·정지·직접 끔 뒤라면 다시 켜지 않는다(시선만으로 로봇이 움직이게 되므로)
        print("[auto] 대화 끝 → 초기화 (자동 교감은 꺼진 채 — 정지·직접 끔 뒤)", flush=True)
        return
    AUTO["on"] = True                     # 대화 끝 → 다른 걸 보면 다음 시나리오로 바로 이어지게 (2026-10-31)
    AUTO["status"] = "자동 교감 켜짐 — 시선을 기다려요"
    print("[auto] 대화 끝 → 초기화 · 자동 교감 ON", flush=True)


dialog = g1_dialog.Dialog(say_text, dialog_attend, robot_busy, mic_dev=args.mic, synth=g1_tts.synth,
                          on_end=_dialog_ended)


# ── 자동 교감(시선 → 명령) ───────────────────────────────────────────────
# 브리지(55057)가 부위 라벨·물체 kind·방위를 보낸다. 오케스트레이터가 dwell 로 확정해
# 머리 → 손 흔들기 인사 뒤 안부 대화(greet), 손 → 악수(act)만, 물체 → 같이 보기 + 물체 대화를 낸다. auto OFF 면 아무 명령도 안 나간다(수동 버튼이 대신 쓴다).
AUTO = {"on": False, "status": "대기", "held_off": False}        # status = /demo 큰 문구(현재 교감 상태)
TALK = {"on": False}                           # 대화 모드: 모션(act·attend) 없이 대화만 — 웹 토글(/talk)
LAST = {"user_bearing": 0, "t": 0.0,           # 최근 착용자 방위(태그 놓쳤을 때·같이 보기 복귀용)와 그 시각
        "obj_bearing": None, "obj_t": 0.0,     # 등록한 꽃(물체) 방위와 그 시각
        "obj_recent": []}                      # 등록 후보: 최근 1초 꽃 방위 (t, deg)
BEARING_MEMORY_S = 5.0                          # 태그를 이만큼 못 보면 착용자 방위를 모르는 것으로(0°). 30초는 자리를 옮긴 뒤
                                                # 예전 방위로 엉뚱하게 돌았다(2026-10-31) — 잠깐 빠지는 것만 메운다
OBJ_MEMORY_S = 1800.0                           # 꽃 위치 등록: 로봇·꽃이 한 번 같이 보이면 30분 기억(돌면 돈 만큼 빼서 갱신, 2026-10-31)


def _turned(by_deg):
    """지원이가 by_deg 만큼(양자화된 회전) 돌았다 → 기억한 착용자 방위를 그만큼 빼 둔다(지원이 기준이 바뀌므로)."""
    LAST["user_bearing"] = int(round(_wrap(LAST["user_bearing"] - _quant(by_deg))))
    LAST["t"] = time.time()
    if LAST["obj_bearing"] is not None:            # 꽃도 지원이 기준이 바뀐 만큼 빼 둔다
        LAST["obj_bearing"] = int(round(_wrap(LAST["obj_bearing"] - _quant(by_deg))))
AUTO_DIALOG_MODE = "mic"                        # 자동 대화 듣기 방식(mic/keyboard)
PART_STATUS = {"g1_face": "👋 머리를 보고 손 흔들기 인사", "g1_hand": "🤝 손을 보고 악수",
               "g1_torso": "🙆 몸을 보고 양팔 벌리기"}


def _auto_send(cmd):
    """오케스트레이터 send_cmd 콜백. cmd = {seq, cmd:'act', part, bearing_deg}. auto ON 일 때만 발사."""
    if not AUTO["on"]:
        return
    if TALK["on"]:                       # 대화 모드: 인사·악수 동작 생략
        AUTO["status"] = PART_STATUS.get(cmd["part"], cmd["part"]) + " (대화 모드 — 동작 생략)"
        print(f"[auto] act 생략(대화 모드): {cmd['part']}", flush=True)
        return
    seq = send({"cmd": cmd["cmd"], "part": cmd["part"], "bearing_deg": cmd["bearing_deg"]})   # send() 가 seq 재부여
    if isinstance(cmd["bearing_deg"], (int, float)):
        LAST["user_bearing"] = int(cmd["bearing_deg"])
        _turned(cmd["bearing_deg"])                # 착용자 쪽으로 돌았으니 이제 착용자는 거의 정면
    AUTO["status"] = PART_STATUS.get(cmd["part"], cmd["part"])
    print(f"[auto] act seq {seq}: {cmd['part']} @ {cmd['bearing_deg']}°", flush=True)


def _auto_dialog(kind, bearing, after_act=False):
    """오케스트레이터 start_dialog 콜백. kind + 방위. 물체면 같이 보기 방위 세팅 후,
    손 인사(greet, after_act)면 손 흔들기가 끝난 뒤 안부로 대화 시작."""
    if not AUTO["on"]:
        return
    greet = kind == "greet"
    look = not greet and bearing is not None                   # 물체 방위를 알 때만 같이 보기(돌아보기)
    if look:
        DLG["bear"] = (int(bearing), LAST["user_bearing"])     # 같이 보기: (물체, 착용자) 방위
    talk = TALK["on"]                                          # 대화 모드: 같이 보기·손 흔들기 없이 → 기다릴 동작도 없다
    msg = dialog.start(AUTO_DIALOG_MODE, with_attend=look and not talk, kind=kind, after_act=after_act and not talk)
    AUTO["status"] = ("👋 손 흔들어 인사하고 안부 나누는 중" if after_act and not talk else
                      "👋 안녕하세요~ 안부 나누는 중" if greet else
                      f"💬 {g1_dialog.SCENARIOS.get(kind, {}).get('name', kind)} 이야기 나누는 중")
    print(f"[auto] dialog {kind} @ {bearing}°: {msg}", flush=True)


def is_busy():
    """오케스트레이터 is_busy 콜백 — 로봇이 동작·말하는 중이거나 대화 진행 중."""
    return robot_busy() or dialog.running


orc = Orchestrator(_auto_send, _auto_dialog, is_busy)


def _bridge_feed(m):
    """브리지 메시지 1건 → 오케스트레이터 투입. 부위 먼저(부위 우선), 물체 다음."""
    t = m.get("t") or time.time()
    ub = m.get("bearing_deg")
    if isinstance(ub, (int, float)):
        ub_i = LAST["user_bearing"] = int(round(ub))
        LAST["t"] = time.time()
    else:   # 태그를 놓친 순간 — 0° 로 보내면 지원이가 안 돈다(2026-10-31). 최근 본 방위가 있으면 그걸로
        ub_i = LAST["user_bearing"] if time.time() - LAST["t"] <= BEARING_MEMORY_S else 0
    label = m.get("label")
    orc.feed_part(t, label if label in PART_TO_SEGMENT else None, ub_i)   # 같은 틱 부위 먼저
    ob = m.get("obj_bearing_deg")
    # 몸통 태그 소실(방위 None)이면 물체 방향을 모른다 → 등록해 둔 꽃 방위(로봇·꽃이 같이 보였을 때), 없으면 None:
    # 돌아보기 없이 대화만 (2026-10-31: 꽃을 보는 동안 로봇이 화면 밖이라 '@ None°'로 시작해 꽃 쪽으로 안 돌았다)
    now = time.time()
    if isinstance(ob, (int, float)):               # 등록은 엄격하게: 1초 안 3번 이상, 흔들림 15° 이내일 때만(뒤집힘 값 배제)
        rec = [x for x in LAST["obj_recent"] if now - x[0] <= 1.0] + [(now, float(ob))]
        LAST["obj_recent"] = rec
        vals = [v for _, v in rec]
        if len(vals) >= 3 and max(vals) - min(vals) <= 15.0:
            LAST["obj_bearing"], LAST["obj_t"] = int(round(sorted(vals)[len(vals) // 2])), now
    ob_i = LAST["obj_bearing"] if LAST["obj_bearing"] is not None and now - LAST["obj_t"] <= OBJ_MEMORY_S else None
    orc.feed_object(t, m.get("obj_kind"), ob_i)


def listen_bridge():
    """브리지(55057) 수신 → 오케스트레이터 투입. 예외가 나도 스레드는 죽지 않는다."""
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", PORT_BRIDGE))
    seen, next_t = {}, time.time() + 2.0      # 2초마다 받은 판정·바쁨 요약 (자동 교감이 왜 안 나가는지 볼 때)
    while True:
        try:
            m = decode(rx.recvfrom(4096)[0])
            if m:
                _bridge_feed(m)
                k = m.get("label") or (m.get("obj_kind") and f"obj:{m['obj_kind']}") or "-"
                seen[k] = seen.get(k, 0) + 1
            if time.time() >= next_t and AUTO["on"]:
                print(f"[bridge] 2초 수신 {seen} · 바쁨 {is_busy()} (로봇 {state.get('state')}, 말 {bool(state.get('speaking'))}, "
                      f"대화 {dialog.running}) · 기억 착용자 {LAST['user_bearing']}° 꽃 {LAST['obj_bearing']}°", flush=True)
                seen, next_t = {}, time.time() + 2.0
        except Exception as e:   # 자동 콜백·디코드 예외로 데몬 스레드가 조용히 죽는 것 방지
            print(f"[bridge] 예외 무시하고 계속: {e!r}", flush=True)


def save_upload(data: bytes) -> str:
    """올린 오디오 → 16 kHz mono WAV (ffmpeg 가 형식을 가리지 않고 바꾼다)."""
    os.makedirs(g1_tts.CACHE, exist_ok=True)
    src = os.path.join(g1_tts.CACHE, f"up_{int(time.time() * 1000)}")
    with open(src, "wb") as f:
        f.write(data)
    out = src + ".wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", out],
                   check=True, timeout=30)
    os.remove(src)
    return out


def server_log():
    if not args.ssh:
        return ""
    try:
        return subprocess.run(SSH + [args.ssh, "tail -n 12 ~/molbwa_g1/server.log 2>/dev/null"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True, timeout=5).stdout
    except subprocess.TimeoutExpired:
        return "(로그 시간 초과)"

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>지원이 조작 패널</title>
<style>
body{font-family:system-ui,sans-serif;max-width:520px;margin:24px auto;padding:0 16px;background:#111;color:#eee}
h1{font-size:20px}.row{display:flex;gap:8px;margin:12px 0}button{flex:1;padding:18px;font-size:18px;border:0;border-radius:10px;cursor:pointer}
.wave{background:#2d7ff9;color:#fff}.shake{background:#16a34a;color:#fff}button:disabled{opacity:.35}
input[type=range]{width:100%}#deg{font-size:32px;font-weight:700}#st{padding:10px;border-radius:8px;background:#222}
.pre button{padding:8px;font-size:14px;background:#333;color:#eee}svg{display:block;margin:auto}
.estop{background:#dc2626;color:#fff;font-size:24px;font-weight:800;padding:22px}.ctl button{font-size:15px;padding:14px;background:#374151;color:#eee}
.on{background:#7c3aed!important}#log{background:#000;color:#9ca3af;font-size:11px;white-space:pre-wrap;padding:8px;border-radius:8px;max-height:180px;overflow:auto}
#robot{padding:10px;border-radius:8px;background:#1f2937;margin-top:8px}
</style>
<h1>지원이 조작 패널 <small style="color:#888">(시선 안경 대신)</small></h1>
<div class="row"><button class="estop" onclick="c('damp')">⛔ 비상정지</button></div>
<div class="row ctl"><button id="talk" onclick="ttalk()">🗣 대화 모드 꺼짐 — 동작+대화</button></div>
<div class="row ctl"><button onclick="fetch('/reset',{method:'POST'}).then(r=>r.json()).then(j=>document.getElementById('msg').textContent=j.msg)">🔄 초기화 — 대화 끊고 응시 누적 비움 + 자동 교감 시작</button></div>
<div class="row ctl"><button onclick="c('stop','정책 중이면: 기본자세로 최대 5분 서 있다가 5초 카운트다운 후 힘이 빠집니다(받쳐야 함). 고수준이면 제자리 정지. 계속할까요?')">■ 정지(기본자세)</button>
<button onclick="c('release','디벨롭 모드로 내리면 로봇이 스스로 못 섭니다. 하네스가 받치고 있나요?')">디벨롭 모드</button>
<button onclick="c('highlevel','고수준(기본 보행) 모드로 올립니다. 계속할까요?')">고수준 모드</button></div>
<div class="row ctl"><button class="on" onclick="c('start','균형 정책을 켭니다(발행). 하네스·비상정지 담당·주변 2m 확인했나요?')">▶ 정책 ON (균형 + 인사)</button>
<button onclick="c('start_dry')">dry-run 20초</button></div>
<div class="row ctl"><button style="background:#b45309!important" onclick="c('handover','고수준으로 서 있는 상태에서 바로 우리 정책으로 넘깁니다(실기 미검증). 하네스를 걸었나요? 비상정지 담당과 주변 2m 확인했나요? 로봇이 가만히 서 있나요?')">⇄ 고수준 → 정책 인계 (하네스 필수)</button>
<button style="background:#0e7490!important" onclick="c('handback','우리 정책에서 고수준(보행 컨트롤러)으로 넘깁니다(실기 미검증). 로봇이 가만히 서 있나요? 하네스·비상정지 담당 확인했나요?')">⇄ 정책 → 고수준 반납 (하네스 필수)</button></div>
<div class="row ctl"><button style="background:#4d7c0f!important" onclick="c('shutdown','고수준에 넘긴 뒤 서기 → 앉기 → 제로토크로 끕니다(실기 미검증). 하네스는 앉을 만큼 길게 걸었나요? 주변 2m 확인했나요?')">🏠 종료: 반납 → 앉기 → 제로토크 (하네스 필수)</button></div>
<div id="robot">로봇: ?</div>
<svg width="200" height="200" viewBox="-100 -100 200 200"><circle r="90" fill="none" stroke="#444"/>
<path d="M-8 6 L0 -14 L8 6 Z" fill="#aaa"/><text y="30" fill="#888" font-size="11" text-anchor="middle">지원이(정면=위)</text>
<circle id="u" r="9" fill="#f59e0b"/></svg>
<div style="text-align:center"><span id="deg">0</span>° <span id="side" style="color:#888"></span></div>
<input id="b" type="range" min="-180" max="180" step="15" value="0">
<div class="row pre"></div>
<div class="row"><button class="wave" onclick="act('g1_face')">👋 손 흔들기</button>
<button class="shake" onclick="act('g1_hand')">🤝 악수</button></div>
<h3 style="margin:18px 0 4px">같이 보기 <small style="color:#888">(물체 쪽으로 돌아 가리킨 뒤 착용자 쪽으로 돌아와 축하)</small></h3>
<div>물체 방위 <b id="odeg">60</b>° <input id="o" type="range" min="-180" max="180" step="15" value="60"></div>
<div style="color:#888;font-size:13px">착용자 방위 = 위 원의 주황 점(슬라이더). 젯슨에 정책 v16 22900 + 라이브러리 v12 가 올라가 있어야 한다(그 전 라이브러리는 가리키기·박수가 없어 로봇이 거부하고 서 있기만 함)</div>
<div class="row"><button class="wave" style="background:#d97706" onclick="attend('celebrate')">🎉 같이 보고 축하</button>
<button class="wave" style="background:#0891b2" onclick="attend('wave')">👀 같이 보고 손 흔들기</button></div>
<h3 style="margin:18px 0 4px">말하기 <small style="color:#888">(edge-tts → 지원이 스피커)</small></h3>
<textarea id="txt" rows="2" style="width:100%;font-size:16px;background:#222;color:#eee;border:1px solid #444;border-radius:8px;padding:8px" placeholder="지원이가 할 말"></textarea>
<div class="row pre" id="phr"></div>
<div class="row"><button class="wave" style="background:#9333ea" onclick="say()">🔊 말하기</button></div>
<label style="display:block;margin:6px 0"><input type="checkbox" id="clapsay"> 같이 보기 때 <b>마지막 동작(박수·손 흔들기) 시작에</b> 위 문장 말하기</label>
<div style="font-size:14px;color:#aaa">WAV 파일: <input type="file" id="wavf" accept=".wav,audio/wav"> <button style="padding:6px 10px;font-size:14px" onclick="upwav()">올려서 재생</button></div>
<div id="spk" style="margin-top:8px;color:#888">🔇 조용</div>
<h3 style="margin:18px 0 4px">대화 <small style="color:#888">(물건 → 고정 질문 → 대답 ↔ 공감·질문 … 고마워 → 마무리)</small></h3>
<div class="row"><select id="dkind" style="flex:1;font-size:16px;padding:10px;background:#222;color:#eee;border-radius:8px">
<option value="plant">🪴 화분</option><option value="greet">👋 안부 (손 인사 뒤 안녕하세요~)</option></select>
<select id="dmode" style="flex:1;font-size:16px;padding:10px;background:#222;color:#eee;border-radius:8px">
<option value="mic">🎤 마이크로 듣기</option><option value="keyboard">⌨ 대답을 여기 입력</option></select></div>
<label style="display:block;margin:6px 0"><input type="checkbox" id="dattend"> 첫 질문을 <b>같이 보기</b>(위 물체·착용자 방위)의 마지막 동작에 맞춰 말하기 <small style="color:#888">(정책 켜졌을 때만 · 양팔 벌리기 학습 전에는 마지막 동작이 박수)</small></label>
<div class="row"><button class="wave" style="background:#059669" onclick="dlg('start')">💬 대화 시작</button>
<button style="background:#374151;color:#eee" onclick="dlg('stop')">■ 대화 중지</button>
<button style="background:#374151;color:#eee" onclick="dlg('mictest')">🎤 마이크 시험(3초)</button></div>
<div class="row"><input id="dans" style="flex:3;font-size:16px;padding:10px;background:#222;color:#eee;border:1px solid #444;border-radius:8px" placeholder="(키보드 모드) 어르신 대답">
<button style="flex:1;background:#374151;color:#eee" onclick="dlg('answer')">보내기</button></div>
<div id="dalert" style="display:none;background:#7f1d1d;color:#fff;font-weight:700;padding:10px;border-radius:8px;margin-bottom:6px"></div>
<div id="dlog" style="background:#0b1220;color:#cbd5e1;font-size:14px;white-space:pre-wrap;padding:8px;border-radius:8px;min-height:40px">대화 기록</div>
<div id="st">상태: ?</div><p id="msg" style="color:#888"></p><div id="log"></div>
<script>
const b=document.getElementById('b'),pre=document.querySelector('.pre');
[-135,-90,-45,0,45,90,135,180].forEach(v=>{const e=document.createElement('button');e.textContent=v;e.onclick=()=>{b.value=v;draw()};pre.appendChild(e)});
function draw(){const v=+b.value,r=v*Math.PI/180;document.getElementById('deg').textContent=v;
document.getElementById('side').textContent=v>0?'왼쪽':v<0?'오른쪽':'정면';
const u=document.getElementById('u');u.setAttribute('cx',-80*Math.sin(r));u.setAttribute('cy',-80*Math.cos(r))}
b.oninput=draw;draw();
let talk=false;
async function ttalk(){talk=!talk;const r=await fetch('/talk',{method:'POST',body:JSON.stringify({on:talk})});
document.getElementById('msg').textContent=(await r.json()).msg}
let ctlUntil=0;
async function c(cmd,warn){if(warn&&!confirm(warn))return;ctlUntil=Date.now()+60000;document.getElementById('robot').textContent='로봇: '+cmd+' 보내는 중…';
const r=await fetch('/ctl/'+cmd,{method:'POST'});ctlUntil=Date.now()+10000;document.getElementById('robot').textContent='로봇: '+(await r.json()).msg}
setInterval(async()=>{const r=await(await fetch('/robot')).json();if(r.log)document.getElementById('log').textContent=r.log;
if(r.status&&Date.now()>ctlUntil)document.getElementById('robot').textContent='로봇: '+r.status},2000);
const o=document.getElementById('o');o.oninput=()=>document.getElementById('odeg').textContent=o.value;
const txt=document.getElementById('txt'),phr=document.getElementById('phr');
['안녕하세요, 저는 지원이에요.','저기 보세요!','다음 주에 손녀 생일이에요. 축하해요!','오늘 이야기 나눠서 즐거웠어요.'].forEach(v=>{const e=document.createElement('button');e.textContent=v;e.onclick=()=>{txt.value=v};phr.appendChild(e)});
async function say(){if(!txt.value.trim())return;document.getElementById('msg').textContent='합성 중…';
const r=await fetch('/say',{method:'POST',body:JSON.stringify({text:txt.value})});document.getElementById('msg').textContent=(await r.json()).msg}
async function dlg(op){const q={};if(op==='start'){q.mode=document.getElementById('dmode').value;q.kind=document.getElementById('dkind').value;q.with_attend=document.getElementById('dattend').checked;
q.obj_bearing_deg=+o.value;q.user_bearing_deg=+b.value}if(op==='answer'){q.text=document.getElementById('dans').value;document.getElementById('dans').value=''}
if(op==='mictest')document.getElementById('msg').textContent='마이크 3초 녹음 중… 말해 보세요';
const r=await fetch('/dialog/'+op,{method:'POST',body:JSON.stringify(q)});document.getElementById('msg').textContent=(await r.json()).msg}
async function upwav(){const f=document.getElementById('wavf').files[0];if(!f)return;document.getElementById('msg').textContent='올리는 중…';
const r=await fetch('/say_wav',{method:'POST',body:f});document.getElementById('msg').textContent=(await r.json()).msg}
async function attend(g){const q={obj_bearing_deg:+o.value,user_bearing_deg:+b.value,gesture:g};
if(document.getElementById('clapsay').checked&&txt.value.trim())q.say_text=txt.value;
document.getElementById('msg').textContent='보내는 중…';const r=await fetch('/attend',{method:'POST',body:JSON.stringify(q)});
document.getElementById('msg').textContent=(await r.json()).msg}
async function act(p){const r=await fetch('/act',{method:'POST',body:JSON.stringify({part:p,bearing_deg:+b.value})});
document.getElementById('msg').textContent=(await r.json()).msg}
setInterval(async()=>{const s=await(await fetch('/state')).json(),busy=s.state!=='idle';
document.getElementById('st').textContent='상태: '+(s.stale?'서버 응답 없음':({idle:'서 있는 중',turning:'도는 중',acting:'동작 중'}[s.state]||s.state));
{const a=document.getElementById('dalert');a.style.display=s.dlg_alert?'block':'none';a.textContent=s.dlg_alert||''}
if(s.dlg_log&&s.dlg_log.length)document.getElementById('dlog').textContent=(s.dlg_running?'● 대화 중\\n':'')+s.dlg_log.join('\\n');
document.getElementById('spk').textContent=s.speaking?'🔊 말하는 중 (이 동안 음성 인식은 끄기)':'🔇 조용'+(s.say_last?' · '+s.say_last:'');
talk=!!s.talk_only;{const t=document.getElementById('talk');t.classList.toggle('on',talk);
t.textContent=talk?'🗣 대화 모드 — 모션 없이 대화만 (끄기)':'🗣 대화 모드 꺼짐 — 동작+대화'}
document.querySelectorAll('.wave,.shake').forEach(x=>{if(x.textContent.indexOf('말하기')<0&&x.textContent.indexOf('대화')<0)x.disabled=(busy&&!s.stale)||talk})},300);
</script>"""

# 발표용 페이지(/demo): 큰 상태 문구 + 대화 말풍선 + 자동 교감 토글/표시 + 비상정지만. 세부 버튼·SSH 없음.
DEMO_PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>지원이 시선 교감</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:24px 16px;background:#0b1020;color:#f1f5f9}
h1{font-size:22px;margin:0 0 16px}
#auto{width:100%;padding:28px;font-size:26px;font-weight:800;border:0;border-radius:16px;cursor:pointer;background:#334155;color:#fff}
#auto.on{background:#7c3aed}#dot{display:inline-block;width:16px;height:16px;border-radius:50%;background:#64748b;margin-right:10px;vertical-align:middle}
#dot.on{background:#22c55e;box-shadow:0 0 14px #22c55e}
#talk,#reset{width:100%;margin-top:10px;padding:16px;font-size:19px;font-weight:700;border:0;border-radius:14px;cursor:pointer;background:#1f2937;color:#cbd5e1}
#talk.on{background:#0d9488;color:#fff}
#status{font-size:34px;font-weight:800;text-align:center;padding:30px 16px;margin:18px 0;border-radius:16px;background:#111827;min-height:44px;line-height:1.35}
#bubbles{margin:18px 0;min-height:80px;display:flex;flex-direction:column}
.b{margin:6px 0;padding:12px 16px;border-radius:14px;font-size:17px;max-width:82%;white-space:pre-wrap}
.me{background:#1e293b;align-self:flex-start}.you{background:#2563eb;align-self:flex-end;text-align:right}
#alert{display:none;background:#7f1d1d;color:#fff;font-weight:800;font-size:20px;padding:16px;border-radius:12px;margin:12px 0}
.estop{width:100%;padding:22px;font-size:22px;font-weight:800;border:0;border-radius:14px;background:#dc2626;color:#fff;cursor:pointer;margin-top:12px}
</style>
<h1>지원이 — 시선 교감 데모 <small style="color:#888">(세부 조작은 <a href="/origin" style="color:#60a5fa">/origin</a>)</small></h1>
<button id="auto" onclick="toggle()"><span id="dot"></span><span id="al">자동 교감 시작</span></button>
<button id="talk" onclick="ttalk()">🗣 대화 모드 꺼짐 — 동작+대화</button>
<button id="reset" onclick="fetch('/reset',{method:'POST'})">🔄 초기화 — 대화 끊고 처음부터 + 자동 교감 시작</button>
<div id="status">대기</div>
<div id="alert"></div>
<div id="bubbles"></div>
<button class="estop" onclick="if(confirm('비상정지: 로봇 힘을 뺍니다(받쳐야 함). 계속할까요?'))fetch('/ctl/damp',{method:'POST'})">⛔ 비상정지</button>
<script>
let on=false,talk=false;
async function toggle(){on=!on;await fetch('/auto',{method:'POST',body:JSON.stringify({on:on})});draw()}
async function ttalk(){talk=!talk;await fetch('/talk',{method:'POST',body:JSON.stringify({on:talk})});drawT()}
function drawT(){const t=document.getElementById('talk');t.classList.toggle('on',talk);
t.textContent=talk?'🗣 대화 모드 — 모션 없이 대화만 (끄기)':'🗣 대화 모드 꺼짐 — 동작+대화'}
function draw(){document.getElementById('auto').classList.toggle('on',on);document.getElementById('dot').classList.toggle('on',on);
document.getElementById('al').textContent=on?'자동 교감 중 — 정지':'자동 교감 시작'}
setInterval(async()=>{const s=await(await fetch('/state')).json();on=!!s.auto_on;draw();talk=!!s.talk_only;drawT();
document.getElementById('status').textContent=s.auto_status||(on?'시선을 기다려요':'대기');
{const a=document.getElementById('alert');a.style.display=s.dlg_alert?'block':'none';a.textContent=s.dlg_alert||''}
const bb=document.getElementById('bubbles');bb.innerHTML='';
(s.dlg_log||[]).forEach(line=>{const m=line.match(/^\\d\\d:\\d\\d:\\d\\d \\[(.*?)\\] (.*)$/);if(!m)return;
const you=/대답/.test(m[1]),me=/질문|생성|대체|마무리|고정|위험|기다림/.test(m[1]);if(!you&&!me)return;
const e=document.createElement('div');e.className='b '+(you?'you':'me');e.textContent=m[2];bb.appendChild(e)});
bb.scrollTop=bb.scrollHeight},500);
</script>"""


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json"):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/robot":
            self._send(json.dumps({"status": robot["status"], "log": server_log()}))
        elif self.path == "/state":
            self._send(json.dumps(dict(state, stale=time.time() - state["t"] > 1.0, say_last=SAY["last"],
                                        speaking=bool(state.get("speaking")) or SAY["direct"],
                                        dlg_running=dialog.running, dlg_log=dialog.log[-14:], dlg_alert=dialog.alert,
                                        auto_on=AUTO["on"], auto_status=AUTO["status"], talk_only=TALK["on"])))
        elif self.path == "/demo":
            self._send(DEMO_PAGE, "text/html")
        else:                                  # "/", "/origin", 그 외 → 기존 조작 패널
            self._send(PAGE, "text/html")

    def do_POST(self):
        if self.path.startswith("/ctl/"):
            cmd = self.path[5:]
            if cmd not in CTL:
                self.send_error(400)
                return
            if cmd in STOP_CMDS:   # 멈춤/정책 내림 → 자동 교감 해제(정책 되살릴 때 다음 응시에 로봇이 또 움직이지 않게)
                AUTO["on"] = False
                AUTO["held_off"] = True   # 대화가 끝나도 다시 켜지 않는다 — 사람이 켜거나 초기화를 눌러야
                AUTO["status"] = "자동 교감 꺼짐 (정지)"
            msg = ctl(cmd)
            print(f"[ctl] {cmd}: {msg}", flush=True)
            self._send(json.dumps({"msg": msg}))
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/auto":
            try:
                on = bool(json.loads(body or b"{}").get("on"))
            except ValueError:
                self.send_error(400)
                return
            AUTO["on"] = on
            AUTO["held_off"] = not on        # 직접 끄면 대화가 끝나도 다시 켜지 않는다
            AUTO["status"] = "자동 교감 켜짐 — 시선을 기다려요" if on else "자동 교감 꺼짐"
            print(f"[auto] {'ON' if on else 'OFF'}", flush=True)
            self._send(json.dumps({"auto_on": AUTO["on"], "msg": AUTO["status"]}))
            return
        if self.path == "/reset":           # 초기화: 대화 중지 + 응시 누적·재확정 금지 비움. 로봇엔 아무 명령도 안 보낸다
            dialog.stop()
            orc.reset()
            AUTO["on"], AUTO["held_off"] = True, False   # 초기화하면 자동 교감 시작 (2026-10-31)
            AUTO["status"] = "🔄 초기화 — 자동 교감 켜짐, 시선을 기다려요"
            print("[auto] 초기화 · 자동 교감 ON", flush=True)
            self._send(json.dumps({"msg": AUTO["status"]}))
            return
        if self.path == "/talk":
            try:
                TALK["on"] = bool(json.loads(body or b"{}").get("on"))
            except ValueError:
                self.send_error(400)
                return
            print(f"[talk] 대화 모드 {'ON' if TALK['on'] else 'OFF'}", flush=True)
            self._send(json.dumps({"talk_only": TALK["on"],
                                   "msg": "대화 모드 — 모션 없이 대화만" if TALK["on"] else "대화 모드 꺼짐 — 동작+대화"}))
            return
        if AUTO["on"] and (self.path in ("/act", "/attend") or self.path == "/dialog/start"):
            self._send(json.dumps({"msg": "자동 교감 중 — 수동 버튼은 자동을 끈 뒤에 쓰세요"}))
            return
        if TALK["on"] and self.path in ("/act", "/attend"):
            self._send(json.dumps({"msg": "대화 모드라 동작은 보내지 않아요 — 대화 모드를 끈 뒤에 쓰세요"}))
            return
        if self.path in ("/say", "/say_wav"):
            try:
                if self.path == "/say":
                    wav = g1_tts.synth(json.loads(body)["text"])
                else:
                    wav = save_upload(body)
                msg = speak(wav)
            except Exception as e:
                msg = f"말하기 실패: {type(e).__name__} {e}"
            print("[say]", msg)
            self._send(json.dumps({"msg": msg}))
            return
        if self.path.startswith("/dialog/"):
            try:
                req = json.loads(body or b"{}")
                op = self.path[8:]
                if op == "start":
                    DLG["bear"] = (int(req.get("obj_bearing_deg", 60)), int(req.get("user_bearing_deg", -30)))
                    msg = dialog.start(req.get("mode", "mic"), bool(req.get("with_attend")) and not TALK["on"],
                                       str(req.get("kind", "plant")))
                elif op == "stop":
                    msg = dialog.stop()
                elif op == "answer":
                    msg = dialog.answer(str(req.get("text", "")))
                elif op == "mictest":
                    msg = dialog.mic_test()
                else:
                    raise ValueError(op)
            except Exception as e:
                msg = f"대화 오류: {type(e).__name__} {e}"
            self._send(json.dumps({"msg": msg}))
            return
        try:
            req = json.loads(body)
            if self.path == "/attend":
                try:
                    seq, segs, at = send_attend(req["obj_bearing_deg"], req["user_bearing_deg"],
                                                req.get("gesture", "celebrate"), req.get("say_text", ""))
                except (ValueError, KeyError, TypeError):
                    raise
                except Exception as e:
                    self._send(json.dumps({"msg": f"말 합성 실패 — 동작 안 보냄: {e}"}))
                    return
                extra = f" (🔊 {at} 시작에 말함)" if at else ""
                self._send(json.dumps({"msg": f"보냄 seq {seq}: {' → '.join(segs)}{extra}"}))
                return
            else:
                msg = {"cmd": "act", "part": req["part"], "bearing_deg": int(req["bearing_deg"])}
                segs = plan(msg["part"], msg["bearing_deg"])   # 잘못된 부위·각도는 여기서 ValueError
        except (ValueError, KeyError, TypeError):
            self.send_error(400)
            return
        seq = send(msg)
        self._send(json.dumps({"msg": f"보냄 seq {seq}: {' → '.join(segs)}"}))

    def log_message(self, *a):
        pass


robot = {"status": ""}


def poll_status():
    while args.ssh:
        robot["status"] = ctl("status")
        time.sleep(10)   # 젯슨 부하 최소화 (정책 루프 20 ms 주기)


assert set(PART_TO_SEGMENT) >= {"g1_face", "g1_hand"}


def main():
    # 스레드·소켓 바인드·서버는 실행할 때만(테스트는 import 만 한다).
    threading.Thread(target=listen_state, daemon=True).start()
    threading.Thread(target=dialog.warm, daemon=True).start()   # 고정·대비 문장 미리 합성(캐시)
    threading.Thread(target=poll_status, daemon=True).start()
    threading.Thread(target=listen_bridge, daemon=True).start()  # 시선 브리지(55057) → 자동 교감
    print(f"패널: http://0.0.0.0:{args.port}  → 명령 {args.jetson}:{PORT_CMD} · 자동 교감 수신 :{PORT_BRIDGE}")
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
