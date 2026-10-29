"""시선 부위/물체 분기 오케스트레이터 (순수 상태기계, 소켓·ROS 비의존).

부위(얼굴/손) dwell 확정 -> act 명령, 물체 dwell 확정 -> 대화 시작. 부위가 우선.
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "ros2_ws", "src", "gaze_hri", "gaze_hri"))
from g1_dwell import LabelDwell  # noqa: E402

PART_PRIORITY_WINDOW = 1.0  # 부위 확정 후 이 시간(초) 안의 물체 확정은 버린다


class Orchestrator:
    def __init__(self, send_cmd, start_dialog, is_busy, dwell_time=3.0):
        self._send, self._dialog, self._busy = send_cmd, start_dialog, is_busy
        self._part = LabelDwell(dwell_time=dwell_time)
        self._obj = LabelDwell(dwell_time=dwell_time)
        self._seq = 0
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

    def feed_object(self, t, kind, bearing_deg):
        if self._busy():
            return
        got = self._obj.update(t, kind)
        if got is None or t - self._part_fire_t < PART_PRIORITY_WINDOW:
            return  # 부위 우선
        self._dialog(got, bearing_deg)
