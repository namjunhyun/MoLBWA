"""시선 안경 없이 쓰는 데모 2 조작 패널 (웹). 각도와 동작을 골라 g1_motion_server 에 act 명령을 보낸다.

    python3 g1/g1_panel.py [--jetson 127.0.0.1] [--port 8080]   → 브라우저에서 http://<이 PC>:8080

아무것도 안 누르면 서버는 idle(서 있기) 세그먼트를 계속 재생한다 — 패널은 명령만 보낸다.
g1_interaction_node 를 대신하므로 같이 띄우지 말 것(상태 포트 55071 이 겹친다). 의존성 없음(표준 라이브러리).
"""
import argparse
import json
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from g1_protocol import PART_TO_SEGMENT, PORT_CMD, PORT_STATE, decode, encode, plan, plan_attend

AP = argparse.ArgumentParser()
AP.add_argument("--jetson", default="127.0.0.1", help="g1_motion_server 가 도는 곳 (sim 이면 127.0.0.1)")
AP.add_argument("--port", type=int, default=8080)
AP.add_argument("--ssh", default="", help="실기: unitree@<젯슨 IP>. 주면 비상정지·정지·모드·정책 ON 버튼이 켜진다 (젯슨 ~/molbwa_g1/g1_ctl.py)")
args = AP.parse_args()

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
async function attend(g){const r=await fetch('/attend',{method:'POST',body:JSON.stringify({obj_bearing_deg:+o.value,user_bearing_deg:+b.value,gesture:g})});
document.getElementById('msg').textContent=(await r.json()).msg}
async function act(p){const r=await fetch('/act',{method:'POST',body:JSON.stringify({part:p,bearing_deg:+b.value})});
document.getElementById('msg').textContent=(await r.json()).msg}
setInterval(async()=>{const s=await(await fetch('/state')).json(),busy=s.state!=='idle';
document.getElementById('st').textContent='상태: '+(s.stale?'서버 응답 없음':({idle:'서 있는 중',turning:'도는 중',acting:'동작 중'}[s.state]||s.state));
document.querySelectorAll('.wave,.shake').forEach(x=>x.disabled=busy&&!s.stale)},300);
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
            self._send(json.dumps(dict(state, stale=time.time() - state["t"] > 1.0)))
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
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/attend":
                msg = {"cmd": "attend", "obj_bearing_deg": int(req["obj_bearing_deg"]),
                       "user_bearing_deg": int(req["user_bearing_deg"]), "gesture": req.get("gesture", "celebrate")}
                segs = plan_attend(msg["obj_bearing_deg"], msg["user_bearing_deg"], msg["gesture"])
            else:
                msg = {"cmd": "act", "part": req["part"], "bearing_deg": int(req["bearing_deg"])}
                segs = plan(msg["part"], msg["bearing_deg"])   # 잘못된 부위·각도는 여기서 ValueError
        except (ValueError, KeyError, TypeError):
            self.send_error(400)
            return
        last["seq"] = seq = max(last["seq"] + 1, state["seq"] + 1, int(time.time() * 1000))   # g1_dwell.next_seq 와 같은 규칙
        tx.sendto(encode(dict(msg, seq=seq)), (args.jetson, PORT_CMD))
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
