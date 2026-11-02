#!/usr/bin/env python3
"""로봇·YOLO·TTS 없이 자동 교감 관통 확인: 가짜 55057 이벤트 → 55070 명령 / 대화.

    /usr/bin/python3 g1/tests/manual_auto_integration.py

가짜 젯슨(55070 수신)을 띄우고 g1_panel.listen_bridge 스레드를 돌린 뒤, 55057 로
손 응시(g1_hand) dwell 로 악수 act, 머리 응시(g1_face) dwell 로 손 흔들기 act 가 55070 에 오고 안부 대화가 불리는지, 물체(plant) dwell 로
dialog.start(가짜로 교체)가 불리는지, auto OFF 면 명령이 끊기는지 본다."""
import os
import socket
import sys
import threading
import time

sys.argv = ["manual_auto_integration"]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import g1_panel  # noqa: E402
from g1_protocol import PORT_BRIDGE, PORT_CMD, decode, encode  # noqa: E402

# 1) 가짜 젯슨: 55070 명령 수집
got = []
jet = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
jet.bind(("127.0.0.1", PORT_CMD))
jet.settimeout(0.2)


def _collect():
    while True:
        try:
            m = decode(jet.recvfrom(4096)[0])
            if m:
                got.append(m)
        except socket.timeout:
            pass


threading.Thread(target=_collect, daemon=True).start()

# 2) 대화는 가짜(TTS·CLOVA 안 띄움)
dlg_calls = []
g1_panel.dialog.start = lambda *a, **k: (dlg_calls.append((a, k)), "가짜 대화 시작")[1]

# 3) 자동 ON + 브리지 수신
g1_panel.AUTO["on"] = True
threading.Thread(target=g1_panel.listen_bridge, daemon=True).start()
time.sleep(0.2)

tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def feed(label, obj_kind, sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        tx.sendto(encode({"t": time.time(), "label": label, "bearing_deg": -30,
                          "obj_kind": obj_kind, "obj_bearing_deg": 60, "valid": True}),
                  ("127.0.0.1", PORT_BRIDGE))
        time.sleep(0.1)


print("손(g1_hand) 응시 3.6초 → 악수…")
feed("g1_hand", None, 3.6)
time.sleep(0.4)
acts = [m for m in got if m.get("cmd") == "act"]
assert acts, "55070 에 act 명령이 안 왔다"
print("  → 55070 수신:", acts[-1])
assert acts[-1]["part"] == "g1_hand" and acts[-1]["bearing_deg"] == -30   # 젯슨에서 g1_hand = 회전 + handshake
assert dlg_calls and dlg_calls[-1][1].get("kind") == "greet" and dlg_calls[-1][1].get("after_act"), "악수 뒤 안부가 안 불렸다"
print("  → dialog.start:", dlg_calls[-1])
time.sleep(1.5)                                                          # 부위 우선 창 지나기

print("머리(g1_face) 응시 3.6초 → 손 흔들기…")
n_dlg = len(dlg_calls)
feed("g1_face", None, 3.6)
time.sleep(0.4)
acts = [m for m in got if m.get("cmd") == "act"]
assert acts[-1]["part"] == "g1_face", "머리 응시에 손 흔들기가 안 나갔다"
assert len(dlg_calls) > n_dlg and dlg_calls[-1][1].get("kind") == "greet", "손 흔들기 뒤 안부가 안 불렸다"
print("  → 55070 수신:", acts[-1])
time.sleep(1.5)

print("물체(plant) 응시 4초…")
feed(None, "plant", 4.0)
time.sleep(0.4)
assert dlg_calls, "dialog.start 가 안 불렸다"
print("  → dialog.start:", dlg_calls[-1])
a, k = dlg_calls[-1]
assert "plant" in a or k.get("kind") == "plant"

print("auto OFF 중 응시 — 명령이 끊기는지…")
g1_panel.AUTO["on"] = False
n = len(got)
feed("g1_hand", None, 3.6)
time.sleep(0.3)
assert len(got) == n, "auto OFF 인데 명령이 나갔다"

print("\n통합 확인 통과: 55057 → 오케스트레이터 → 55070 · 대화 · auto 게이트까지.")
