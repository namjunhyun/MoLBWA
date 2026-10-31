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


def test_talk_toggle_endpoint_and_state():
    code, body = _call("POST", "/talk", json.dumps({"on": True}).encode())
    assert code == 200 and json.loads(body)["talk_only"] is True
    assert json.loads(_call("GET", "/state")[1])["talk_only"] is True
    _call("POST", "/talk", json.dumps({"on": False}).encode())
    assert g1_panel.TALK["on"] is False


def test_talk_mode_skips_auto_act(monkeypatch):
    sent = []
    monkeypatch.setattr(g1_panel, "send", lambda m: sent.append(m) or 1)
    g1_panel.AUTO["on"], g1_panel.TALK["on"] = True, True
    g1_panel._auto_send({"seq": 1, "cmd": "act", "part": "g1_hand", "bearing_deg": 30})
    assert sent == []                                   # 대화 모드 → 악수 동작 생략
    g1_panel.TALK["on"] = False
    g1_panel._auto_send({"seq": 2, "cmd": "act", "part": "g1_hand", "bearing_deg": 30})
    assert len(sent) == 1                               # 끄면 원래대로 동작
    g1_panel.AUTO["on"] = False


def test_talk_mode_dialog_without_attend(monkeypatch):
    calls = []
    monkeypatch.setattr(g1_panel.dialog, "start", lambda mode, with_attend, kind, after_act=False: calls.append(with_attend) or "ok")
    g1_panel.AUTO["on"], g1_panel.TALK["on"] = True, True
    g1_panel._auto_dialog("plant", 40)                 # 자동: 같이 보기 동작 없이
    g1_panel.AUTO["on"] = False
    _call("POST", "/dialog/start", json.dumps({"kind": "plant", "with_attend": True}).encode())   # 수동도 강제로 끔
    g1_panel.TALK["on"] = False
    g1_panel.AUTO["on"] = True
    g1_panel._auto_dialog("plant", 40)                 # 대화 모드 끄면 다시 같이 보기
    g1_panel.AUTO["on"] = False
    assert calls == [False, False, True]


def test_talk_mode_hand_greet_does_not_wait_for_skipped_wave(monkeypatch):
    calls = []
    monkeypatch.setattr(g1_panel.dialog, "start", lambda mode, with_attend, kind, after_act=False: calls.append((kind, after_act)) or "ok")
    g1_panel.AUTO["on"], g1_panel.TALK["on"] = True, True
    g1_panel._auto_dialog("greet", 0, True)            # 대화 모드: 손 흔들기 생략 → 기다리지 않고 바로 안부
    g1_panel.TALK["on"] = False
    g1_panel._auto_dialog("greet", 0, True)
    g1_panel.AUTO["on"] = False
    assert calls == [("greet", False), ("greet", True)]


def test_talk_mode_blocks_manual_motion(monkeypatch):
    sent = []
    monkeypatch.setattr(g1_panel, "send", lambda m: sent.append(m) or 1)
    monkeypatch.setattr(g1_panel, "send_attend", lambda *a: sent.append(a) or (1, [], None))
    g1_panel.TALK["on"] = True
    for path, req in (("/act", {"part": "g1_hand", "bearing_deg": 0}),
                      ("/attend", {"obj_bearing_deg": 60, "user_bearing_deg": -30})):
        code, body = _call("POST", path, json.dumps(req).encode())
        assert code == 200 and "대화 모드" in json.loads(body)["msg"]
    g1_panel.TALK["on"] = False
    assert sent == []


def test_bridge_no_tag_still_triggers_object_without_bearing(monkeypatch):
    calls = []
    monkeypatch.setattr(g1_panel.orc, "feed_part", lambda *a: None)
    monkeypatch.setattr(g1_panel.orc, "feed_object", lambda t, kind, b: calls.append((kind, b)))
    g1_panel._bridge_feed({"t": 1.0, "obj_kind": "plant", "bearing_deg": 10})            # 태그 없음 → 방위 None (돌아보기 없이 대화만)
    g1_panel._bridge_feed({"t": 2.0, "obj_kind": "plant", "obj_bearing_deg": 30.4})      # 정상
    assert calls == [("plant", None), ("plant", 30)]   # 2026-10-31: 태그가 없어도 꽃 대화는 시작


def test_reset_route_stops_dialog_and_resets_orchestrator(monkeypatch):
    calls = []
    monkeypatch.setattr(g1_panel.orc, "reset", lambda: calls.append("orc"))
    monkeypatch.setattr(g1_panel.dialog, "stop", lambda: calls.append("dlg") or "대화 중지 요청")
    g1_panel.AUTO["on"] = False
    code, body = _call("POST", "/reset")
    assert code == 200 and "초기화" in json.loads(body)["msg"]
    assert calls == ["dlg", "orc"]
    assert g1_panel.AUTO["on"] is True            # 2026-10-31: 초기화하면 자동 교감 시작 (로봇에 명령은 안 보낸다)
    g1_panel.AUTO["on"] = False


def test_dialog_end_resets_orchestrator(monkeypatch):
    calls = []
    monkeypatch.setattr(g1_panel.orc, "reset", lambda: calls.append("orc"))
    g1_panel.AUTO["on"], g1_panel.AUTO["held_off"] = False, False
    monkeypatch.setattr(g1_panel.dialog, "thanked", True)
    g1_panel._dialog_ended()
    assert calls == ["orc"] and g1_panel.AUTO["on"] is True and "시선" in g1_panel.AUTO["status"]   # 대화 끝 → 자동 교감 시작
    g1_panel.AUTO["on"] = False


def test_dialog_end_does_not_rearm_after_estop_or_manual_off(monkeypatch):
    # 비상정지·정지·직접 끔 뒤에 대화가 끝나도 자동 교감이 다시 켜지면 안 된다(시선만으로 로봇이 다시 움직임)
    monkeypatch.setattr(g1_panel, "ctl", lambda cmd: "ok")
    monkeypatch.setattr(g1_panel.orc, "reset", lambda: None)
    monkeypatch.setattr(g1_panel.dialog, "stop", lambda: "대화 중지 요청")
    monkeypatch.setattr(g1_panel.dialog, "thanked", True)
    g1_panel.AUTO["on"] = True
    _call("POST", "/ctl/damp")
    g1_panel._dialog_ended()
    assert g1_panel.AUTO["on"] is False
    _call("POST", "/auto", json.dumps({"on": True}).encode())
    _call("POST", "/auto", json.dumps({"on": False}).encode())
    g1_panel._dialog_ended()
    assert g1_panel.AUTO["on"] is False
    _call("POST", "/reset")                        # 사람이 초기화를 누르면 다시 켠다
    assert g1_panel.AUTO["on"] is True
    g1_panel.AUTO["on"] = False


def test_auto_dialog_without_bearing_talks_without_attend(monkeypatch):
    calls = []
    monkeypatch.setattr(g1_panel.dialog, "start", lambda mode, with_attend, kind, after_act=False: calls.append((kind, with_attend)) or "ok")
    g1_panel.AUTO["on"], g1_panel.TALK["on"] = True, False
    g1_panel._auto_dialog("plant", None)         # 태그 없음 → 꽃 쪽으로 못 돌아보니 대화만
    g1_panel._auto_dialog("plant", 40)           # 태그 있음 → 돌아보고 대화
    g1_panel.AUTO["on"] = False
    assert calls == [("plant", False), ("plant", True)]


def test_dialog_end_without_thanks_does_not_reset(monkeypatch):
    # "고마워" 없이 끝난 대화(대답 없음·오류·중지)는 다음 시나리오로 넘기지 않는다
    calls = []
    monkeypatch.setattr(g1_panel.orc, "reset", lambda: calls.append("orc"))
    monkeypatch.setattr(g1_panel.dialog, "thanked", False)
    g1_panel.AUTO["on"], g1_panel.AUTO["held_off"] = False, False
    g1_panel._dialog_ended()
    assert calls == [] and g1_panel.AUTO["on"] is False
