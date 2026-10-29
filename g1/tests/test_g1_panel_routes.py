"""패널 HTTP 라우팅 단위 테스트. 소켓·젯슨·YOLO 없이 핸들러(H)를 직접 호출한다.

g1_panel 은 모듈을 import 하면 바로 돌 수 있어야 한다(argparse 는 parse_known_args, 스레드·서버는 __main__).
각 테스트는 BaseHTTPRequestHandler 를 소켓 없이 세워(BytesIO rfile/wfile) do_GET/do_POST 를 부른다."""
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import g1_panel  # noqa: E402


def _call(method, path, body=b""):
    """핸들러를 소켓 없이 세워 라우트 하나를 태운다. 반환 = (상태코드, 바디 bytes)."""
    h = g1_panel.H.__new__(g1_panel.H)
    h.rfile = io.BytesIO(body)
    h.wfile = io.BytesIO()
    h.path = path
    h.command = method
    h.request_version = "HTTP/1.0"
    h.requestline = f"{method} {path} HTTP/1.0"
    h.headers = {"Content-Length": str(len(body))}
    h.client_address = ("test", 0)
    getattr(h, "do_" + method)()
    raw = h.wfile.getvalue()
    head, _, payload = raw.partition(b"\r\n\r\n")
    code = int(head.split(b" ", 2)[1])
    return code, payload


def test_demo_route_served():
    code, body = _call("GET", "/demo")
    assert code == 200
    text = body.decode("utf-8")
    assert "/state" in text          # 상태 문구를 폴링하는 발표 페이지
    assert "비상정지" in text


def test_origin_route_served():
    code, body = _call("GET", "/origin")
    assert code == 200
    assert body.decode("utf-8") == g1_panel.PAGE


def test_auto_toggle_endpoint():
    g1_panel.AUTO["on"] = False
    code, body = _call("POST", "/auto", json.dumps({"on": True}).encode())
    assert code == 200
    assert json.loads(body)["auto_on"] is True
    assert g1_panel.AUTO["on"] is True
    _call("POST", "/auto", json.dumps({"on": False}).encode())
    assert g1_panel.AUTO["on"] is False


def test_state_includes_auto_flag():
    code, body = _call("GET", "/state")
    assert code == 200
    s = json.loads(body)
    assert "auto_on" in s and "auto_status" in s


def test_estop_and_stop_cmds_turn_auto_off(monkeypatch):
    monkeypatch.setattr(g1_panel, "ctl", lambda cmd: "ok")
    for cmd in ("damp", "stop", "release", "shutdown"):
        g1_panel.AUTO["on"] = True
        assert _call("POST", "/ctl/" + cmd)[0] == 200
        assert g1_panel.AUTO["on"] is False, cmd
    g1_panel.AUTO["on"] = True
    _call("POST", "/ctl/status")           # 무해한 명령은 유지
    assert g1_panel.AUTO["on"] is True
    g1_panel.AUTO["on"] = False


def test_bridge_no_tag_blocks_object_trigger(monkeypatch):
    calls = []
    monkeypatch.setattr(g1_panel.orc, "feed_part", lambda *a: None)
    monkeypatch.setattr(g1_panel.orc, "feed_object", lambda t, kind, b: calls.append((kind, b)))
    g1_panel._bridge_feed({"t": 1.0, "obj_kind": "cup", "bearing_deg": 10})              # 태그 없음
    g1_panel._bridge_feed({"t": 2.0, "obj_kind": "cup", "obj_bearing_deg": 30.4})        # 정상
    assert calls == [(None, 0), ("cup", 30)]
