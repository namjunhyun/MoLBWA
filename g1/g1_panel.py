"""시선 안경 없이 쓰는 데모 2 조작 패널 (웹). 각도와 동작을 골라 g1_motion_server 에 act 명령을 보낸다.

    python3 g1/g1_panel.py [--jetson 127.0.0.1] [--port 8080]   → 브라우저에서 http://<이 PC>:8080

아무것도 안 누르면 서버는 idle(서 있기) 세그먼트를 계속 재생한다 — 패널은 명령만 보낸다.
g1_interaction_node 를 대신하므로 같이 띄우지 말 것(상태 포트 55071 이 겹친다). 의존성 없음(표준 라이브러리).
"""
import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from g1_protocol import PART_TO_SEGMENT, PORT_CMD, PORT_STATE, decode, encode, plan

AP = argparse.ArgumentParser()
AP.add_argument("--jetson", default="127.0.0.1", help="g1_motion_server 가 도는 곳 (sim 이면 127.0.0.1)")
AP.add_argument("--port", type=int, default=8080)
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

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>지원이 조작 패널</title>
<style>
body{font-family:system-ui,sans-serif;max-width:520px;margin:24px auto;padding:0 16px;background:#111;color:#eee}
h1{font-size:20px}.row{display:flex;gap:8px;margin:12px 0}button{flex:1;padding:18px;font-size:18px;border:0;border-radius:10px;cursor:pointer}
.wave{background:#2d7ff9;color:#fff}.shake{background:#16a34a;color:#fff}button:disabled{opacity:.35}
input[type=range]{width:100%}#deg{font-size:32px;font-weight:700}#st{padding:10px;border-radius:8px;background:#222}
.pre button{padding:8px;font-size:14px;background:#333;color:#eee}svg{display:block;margin:auto}
</style>
<h1>지원이 조작 패널 <small style="color:#888">(시선 안경 대신)</small></h1>
<svg width="200" height="200" viewBox="-100 -100 200 200"><circle r="90" fill="none" stroke="#444"/>
<path d="M-8 6 L0 -14 L8 6 Z" fill="#aaa"/><text y="30" fill="#888" font-size="11" text-anchor="middle">지원이(정면=위)</text>
<circle id="u" r="9" fill="#f59e0b"/></svg>
<div style="text-align:center"><span id="deg">0</span>° <span id="side" style="color:#888"></span></div>
<input id="b" type="range" min="-180" max="180" step="15" value="0">
<div class="row pre"></div>
<div class="row"><button class="wave" onclick="act('g1_face')">👋 손 흔들기</button>
<button class="shake" onclick="act('g1_hand')">🤝 악수</button></div>
<div id="st">상태: ?</div><p id="msg" style="color:#888"></p>
<script>
const b=document.getElementById('b'),pre=document.querySelector('.pre');
[-135,-90,-45,0,45,90,135,180].forEach(v=>{const e=document.createElement('button');e.textContent=v;e.onclick=()=>{b.value=v;draw()};pre.appendChild(e)});
function draw(){const v=+b.value,r=v*Math.PI/180;document.getElementById('deg').textContent=v;
document.getElementById('side').textContent=v>0?'왼쪽':v<0?'오른쪽':'정면';
const u=document.getElementById('u');u.setAttribute('cx',-80*Math.sin(r));u.setAttribute('cy',-80*Math.cos(r))}
b.oninput=draw;draw();
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
        if self.path == "/state":
            self._send(json.dumps(dict(state, stale=time.time() - state["t"] > 1.0)))
        else:
            self._send(PAGE, "text/html")

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            part, b = req["part"], int(req["bearing_deg"])
            segs = plan(part, b)   # 잘못된 부위·각도는 여기서 ValueError
        except (ValueError, KeyError, TypeError):
            self.send_error(400)
            return
        last["seq"] = seq = max(last["seq"] + 1, state["seq"] + 1, int(time.time() * 1000))   # g1_dwell.next_seq 와 같은 규칙
        tx.sendto(encode({"seq": seq, "cmd": "act", "part": part, "bearing_deg": b}), (args.jetson, PORT_CMD))
        self._send(json.dumps({"msg": f"보냄 seq {seq}: {' → '.join(segs)}"}))

    def log_message(self, *a):
        pass


assert set(PART_TO_SEGMENT) >= {"g1_face", "g1_hand"}
print(f"패널: http://0.0.0.0:{args.port}  → 명령 {args.jetson}:{PORT_CMD}")
ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
