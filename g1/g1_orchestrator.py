"""시선 부위/물체 분기 오케스트레이터 (순수 상태기계, 소켓·ROS 비의존).

머리(얼굴) dwell 확정 -> 착용자 쪽으로 돌아 손 흔들기 인사(act), 손 dwell 확정 -> 악수(act). 둘 다 동작이 끝나면 안부 대화(greet).
물체 dwell 확정 -> 물체 대화. 부위가 우선.
"""
import os
import sys
import time

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "ros2_ws", "src", "gaze_hri", "gaze_hri"))
from g1_dwell import LabelDwell  # noqa: E402

PART_PRIORITY_WINDOW = 1.0  # 부위 확정 후 이 시간(초) 안의 물체 확정은 버린다
GREET_KIND = "greet"        # 인사·악수 뒤 이 대화 시나리오(g1_dialog.SCENARIOS)
GREET_PARTS = {"g1_face", "g1_hand"}   # 동작 뒤 안부 대화로 이어지는 부위 (젯슨: g1_face=wave, g1_hand=handshake)


class Orchestrator:
    def __init__(self, send_cmd, start_dialog, is_busy, dwell_time=3.0):
        self._send, self._dialog, self._busy = send_cmd, start_dialog, is_busy
        self._dwell_time = dwell_time
        self._seq = int(time.time() * 1000)   # ms 기반 — 노드 재시작 후에도 단조 증가(젯슨이 조용히 거부 안 하게)
        self.reset()

    def reset(self):
        """응시 누적·래치(같은 부위 재확정 금지)·부위 우선 창을 비운다 — 패널 '초기화'. seq 는 유지(단조 증가)."""
        self._part = LabelDwell(dwell_time=self._dwell_time)
        self._obj = LabelDwell(dwell_time=self._dwell_time)
        self._part_fire_t = -1e9

    def feed_part(self, t, label, bearing_deg):
        if self._busy():
            return  # 바쁠 땐 투입 안 함 -> dwell 창이 max_gap 으로 비워진다
        got = self._part.update(t, label)
        if got is None:
            return
        self._part_fire_t = t
        self._seq += 1  # 전송마다 단조 증가
        self._send({"seq": self._seq, "cmd": "act", "part": got, "bearing_deg": bearing_deg})
        if got in GREET_PARTS:
            self._dialog(GREET_KIND, bearing_deg, True)   # 인사·악수가 끝나면 안부 (대화가 기다린다)

    def feed_object(self, t, kind, bearing_deg):
        if self._busy():
            return
        got = self._obj.update(t, kind)
        if got is None or t - self._part_fire_t < PART_PRIORITY_WINDOW:
            return  # 부위 우선
        self._dialog(got, bearing_deg, False)
