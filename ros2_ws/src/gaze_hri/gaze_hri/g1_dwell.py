"""데모 2(G1) 라벨 dwell: 같은 부위 라벨을 dwell_time 동안 min_ratio 이상 보면 1회 확정.

dwell_detector_node 의 3D 산포(I-DT) 대신 라벨 유지로 판정한다 — 브리지가 이미 부위를 골라 준다.
ROS 비의존(테스트용). 시각 t 는 단조 증가 초.

연타 방지 두 겹:
  * 확정 후 cooldown 동안은 어떤 라벨도 확정하지 않는다.
  * 확정한 라벨은 창 안 비율이 release_ratio 아래로 떨어질 때까지(=시선이 떠날 때까지) 재확정 없음.
    동작 중 같은 부위를 계속 보고 있어도 동작이 끝나자마자 재실행되지 않는다(Review Focus #2).
"""
from __future__ import annotations

from collections import deque


class LabelDwell:
    def __init__(self, dwell_time=3.0, min_ratio=0.7, cooldown=2.0,
                 release_ratio=0.3, max_gap=0.5):
        self.dwell_time = float(dwell_time)
        self.min_ratio = float(min_ratio)
        self.cooldown = float(cooldown)
        self.release_ratio = float(release_ratio)
        self.max_gap = float(max_gap)       # 샘플이 이만큼 끊기면 창을 비운다(브리지 정지)
        self._buf = deque()                 # (t, label | None)
        self._last_fire_t = -1e9
        self._latched = None                # 마지막 확정 라벨, 떠나기 전까지 재확정 금지

    def _ratio(self, label):
        return sum(1 for _, x in self._buf if x == label) / len(self._buf) if self._buf else 0.0

    def _top(self):
        labels = [x for _, x in self._buf if x is not None]
        return max(set(labels), key=labels.count) if labels else None

    def update(self, t: float, label: str | None) -> str | None:
        if self._buf and t - self._buf[-1][0] > self.max_gap:
            self._buf.clear()
        self._buf.append((t, label))
        # 창 경계 바깥 샘플을 하나 남겨 둔다 -> buf[0] 이 경계 이전이면 창이 꽉 찬 것.
        while len(self._buf) > 1 and self._buf[1][0] <= t - self.dwell_time:
            self._buf.popleft()

        if self._latched is not None and self._ratio(self._latched) < self.release_ratio:
            self._latched = None
        if t - self._last_fire_t < self.cooldown:
            return None
        if self._buf[0][0] > t - self.dwell_time:
            return None
        top = self._top()
        if top is None or top == self._latched or self._ratio(top) < self.min_ratio:
            return None
        self._last_fire_t = t
        self._latched = top
        return top

    def progress(self, t: float) -> float:
        """0~1 HUD 피드백. cooldown 중이거나 맨 위 라벨이 이미 확정된 것이면 0."""
        top = self._top()
        if top is None or top == self._latched or t - self._last_fire_t < self.cooldown:
            return 0.0
        fill = min(1.0, (t - self._buf[0][0]) / self.dwell_time)
        return fill * min(1.0, self._ratio(top) / self.min_ratio)


def next_seq(prev: int, now_s: float, jetson_seq: int = 0) -> int:
    """명령 seq: 밀리초 시각 기반 int, 항상 prev·Jetson 보고값보다 크다.
    Jetson(SegmentPlayer)은 seq <= 마지막 수락분을 거부하므로 노드 재시작 후에도 커져야 한다."""
    return int(max(prev + 1, jetson_seq + 1, int(now_s * 1000)))
