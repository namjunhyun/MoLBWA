"""모션 라이브러리 재생 커서: idle 반복 -> act 명령 -> turn_<bin> -> 동작 -> idle.

라이브러리 npz 는 50 fps = 제어 50 Hz 라서 step 1회 = 1프레임. 세그먼트 경계는
library_meta.json 의 npz 프레임 인덱스 [start, end). Jetson(Python 3.8)에도 올라간다.
"""
from __future__ import annotations

import os
import sys

try:                                    # Jetson: 같은 폴더에 scp 됨
    from g1_protocol import PART_TO_SEGMENT, turn_segment
except ImportError:                     # 저장소: g1/deploy/ 의 부모가 g1/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from g1_protocol import PART_TO_SEGMENT, turn_segment


class SegmentPlayer:
    def __init__(self, segments: dict):
        self.segments = {n: (int(s), int(e)) for n, (s, e) in segments.items()}
        if "idle" not in self.segments:
            raise ValueError("라이브러리에 idle 세그먼트가 없다")
        for n, (s, e) in self.segments.items():
            if not 0 <= s < e:
                raise ValueError(f"세그먼트 {n} 경계 [{s}, {e}) 가 비었거나 음수")
        self.segment = "idle"           # 지금 재생 중인 세그먼트 이름
        self._i = 0                     # 세그먼트 안 다음 프레임 오프셋
        self._queue = []                # 수락됐지만 아직 시작 안 한 세그먼트들
        self._jump = False              # True 면 다음 step 에서 idle 을 끊고 큐 시작
        self.last_seq = -1

    @property
    def state(self) -> str:
        if self.segment == "idle":
            return "idle"
        return "turning" if self.segment.startswith("turn_") else "acting"

    def step(self):
        """(라이브러리 프레임 인덱스, 세그먼트 첫 프레임인가). True 면 호출측이 yaw_off 재정렬."""
        s, e = self.segments[self.segment]
        if self._jump or s + self._i >= e:
            self._jump = False
            self.segment = self._queue.pop(0) if self._queue else "idle"
            self._i = 0
            s, e = self.segments[self.segment]
        first = self._i == 0
        self._i += 1
        return s + self._i - 1, first

    def command(self, msg) -> bool:
        """act 이고, 쉬는 중이고, seq 가 새것일 때만 수락. 와이파이 중복·역순 패킷은 seq 로 거른다."""
        if not isinstance(msg, dict) or msg.get("cmd") != "act":
            return False
        seq = msg.get("seq")
        if type(seq) is not int or seq <= self.last_seq:
            return False
        if self.segment != "idle" or self._queue:
            return False
        gesture = PART_TO_SEGMENT.get(msg.get("part"))
        turn_bin = msg.get("turn_bin")
        if type(turn_bin) is not int:           # 0.0·False 가 0 으로 통과하지 않게
            return False
        try:
            turn = turn_segment(turn_bin)
        except ValueError:
            return False
        queue = [n for n in (turn, gesture) if n is not None]
        if gesture is None or any(n not in self.segments for n in queue):
            return False
        self._queue = queue
        self._jump = True
        self.last_seq = seq
        return True
