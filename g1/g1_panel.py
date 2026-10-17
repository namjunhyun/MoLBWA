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

from g1_protocol import PART_TO_SEGMENT, PORT_CMD, PORT_STATE, decode, encode, plan, plan_attend
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
args = AP.parse_args()
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


threading.Thread(target=listen_state, daemon=True).start()
CTL = {"damp", "stop", "release", "highlevel", "start", "start_dry", "status"}
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
    send_attend(DLG["bear"][0], DLG["bear"][1], "celebrate", text)
    return True


dialog = g1_dialog.Dialog(say_text, dialog_attend, robot_busy, mic_dev=args.mic)


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
<div class="row ctl"><button onclick="c('stop','정책 중이면: 기본자세로 최대 5분 서 있다가 5초 카운트다운 후 힘이 빠집니다(받쳐야 함). 고수준이면 제자리 정지. 계속할까요?')">■ 정지(기본자세)</button>
<button onclick="c('release','디벨롭 모드로 내리면 로봇이 스스로 못 섭니다. 하네스가 받치고 있나요?')">디벨롭 모드</button>
<button onclick="c('highlevel','고수준(기본 보행) 모드로 올립니다. 계속할까요?')">고수준 모드</button></div>
<div class="row ctl"><button class="on" onclick="c('start','균형 정책을 켭니다(발행). 하네스·비상정지 담당·주변 2m 확인했나요?')">▶ 정책 ON (균형 + 인사)</button>
<button onclick="c('start_dry')">dry-run 20초</button></div>
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
<div style="color:#888;font-size:13px">착용자 방위 = 위 원의 주황 점(슬라이더). ⚠ 가리키기·축하 동작은 학습 전 — 그 전엔 로봇이 이 명령을 거부하고 서 있기만 한다</div>
<div class="row"><button class="wave" style="background:#d97706" onclick="attend('celebrate')">🎉 같이 보고 축하</button>
<button class="wave" style="background:#0891b2" onclick="attend('wave')">👀 같이 보고 손 흔들기</button></div>
<h3 style="margin:18px 0 4px">말하기 <small style="color:#888">(edge-tts → 지원이 스피커)</small></h3>
<textarea id="txt" rows="2" style="width:100%;font-size:16px;background:#222;color:#eee;border:1px solid #444;border-radius:8px;padding:8px" placeholder="지원이가 할 말"></textarea>
<div class="row pre" id="phr"></div>
<div class="row"><button class="wave" style="background:#9333ea" onclick="say()">🔊 말하기</button></div>
<label style="display:block;margin:6px 0"><input type="checkbox" id="clapsay"> 같이 보기 때 <b>마지막 동작(박수·손 흔들기) 시작에</b> 위 문장 말하기</label>
<div style="font-size:14px;color:#aaa">WAV 파일: <input type="file" id="wavf" accept=".wav,audio/wav"> <button style="padding:6px 10px;font-size:14px" onclick="upwav()">올려서 재생</button></div>
<div id="spk" style="margin-top:8px;color:#888">🔇 조용</div>
<h3 style="margin:18px 0 4px">대화 <small style="color:#888">(첫 마디 → 질문 → 대답 듣기 → 공감 → 마무리)</small></h3>
<div class="row"><select id="dmode" style="flex:1;font-size:16px;padding:10px;background:#222;color:#eee;border-radius:8px">
<option value="mic">🎤 마이크로 듣기</option><option value="keyboard">⌨ 대답을 여기 입력</option></select></div>
<label style="display:block;margin:6px 0"><input type="checkbox" id="dattend"> 첫 마디를 <b>같이 보고 축하</b>(위 물체·착용자 방위)의 박수에 맞춰 말하기 <small style="color:#888">(정책 켜졌을 때만)</small></label>
<div class="row"><button class="wave" style="background:#059669" onclick="dlg('start')">💬 대화 시작</button>
<button style="background:#374151;color:#eee" onclick="dlg('stop')">■ 대화 중지</button>
<button style="background:#374151;color:#eee" onclick="dlg('mictest')">🎤 마이크 시험(3초)</button></div>
<div class="row"><input id="dans" style="flex:3;font-size:16px;padding:10px;background:#222;color:#eee;border:1px solid #444;border-radius:8px" placeholder="(키보드 모드) 어르신 대답">
<button style="flex:1;background:#374151;color:#eee" onclick="dlg('answer')">보내기</button></div>
<div id="dlog" style="background:#0b1220;color:#cbd5e1;font-size:14px;white-space:pre-wrap;padding:8px;border-radius:8px;min-height:40px">대화 기록</div>
<div id="st">상태: ?</div><p id="msg" style="color:#888"></p><div id="log"></div>
<script>
const b=document.getElementById('b'),pre=document.querySelector('.pre');
[-135,-90,-45,0,45,90,135,180].forEach(v=>{const e=document.createElement('button');e.textContent=v;e.onclick=()=>{b.value=v;draw()};pre.appendChild(e)});
function draw(){const v=+b.value,r=v*Math.PI/180;document.getElementById('deg').textContent=v;
document.getElementById('side').textContent=v>0?'왼쪽':v<0?'오른쪽':'정면';
const u=document.getElementById('u');u.setAttribute('cx',-80*Math.sin(r));u.setAttribute('cy',-80*Math.cos(r))}
b.oninput=draw;draw();
async function c(cmd,warn){if(warn&&!confirm(warn))return;document.getElementById('robot').textContent='로봇: '+cmd+' 보내는 중…';
const r=await fetch('/ctl/'+cmd,{method:'POST'});document.getElementById('robot').textContent='로봇: '+(await r.json()).msg}
setInterval(async()=>{const r=await(await fetch('/robot')).json();if(r.log)document.getElementById('log').textContent=r.log;
if(r.status&&!document.getElementById('robot').textContent.includes('보내는 중'))document.getElementById('robot').textContent='로봇: '+r.status},2000);
const o=document.getElementById('o');o.oninput=()=>document.getElementById('odeg').textContent=o.value;
const txt=document.getElementById('txt'),phr=document.getElementById('phr');
['안녕하세요, 저는 지원이에요.','저기 보세요!','다음 주에 손녀 생일이에요. 축하해요!','오늘 이야기 나눠서 즐거웠어요.'].forEach(v=>{const e=document.createElement('button');e.textContent=v;e.onclick=()=>{txt.value=v};phr.appendChild(e)});
async function say(){if(!txt.value.trim())return;document.getElementById('msg').textContent='합성 중…';
const r=await fetch('/say',{method:'POST',body:JSON.stringify({text:txt.value})});document.getElementById('msg').textContent=(await r.json()).msg}
async function dlg(op){const q={};if(op==='start'){q.mode=document.getElementById('dmode').value;q.with_attend=document.getElementById('dattend').checked;
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
if(s.dlg_log&&s.dlg_log.length)document.getElementById('dlog').textContent=(s.dlg_running?'● 대화 중\n':'')+s.dlg_log.join('\n');
document.getElementById('spk').textContent=s.speaking?'🔊 말하는 중 (이 동안 음성 인식은 끄기)':'🔇 조용'+(s.say_last?' · '+s.say_last:'');
document.querySelectorAll('.wave,.shake').forEach(x=>{if(x.textContent.indexOf('말하기')<0&&x.textContent.indexOf('대화')<0)x.disabled=busy&&!s.stale})},300);
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
                                        dlg_running=dialog.running, dlg_log=dialog.log[-14:])))
        else:
            self._send(PAGE, "text/html")

    def do_POST(self):
        if self.path.startswith("/ctl/"):
            cmd = self.path[5:]
            if cmd not in CTL:
                self.send_error(400)
                return
            msg = ctl(cmd)
            print(f"[ctl] {cmd}: {msg}")
            self._send(json.dumps({"msg": msg}))
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
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
                    msg = dialog.start(req.get("mode", "mic"), bool(req.get("with_attend")))
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


threading.Thread(target=poll_status, daemon=True).start()
assert set(PART_TO_SEGMENT) >= {"g1_face", "g1_hand"}
print(f"패널: http://0.0.0.0:{args.port}  → 명령 {args.jetson}:{PORT_CMD}")
ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
