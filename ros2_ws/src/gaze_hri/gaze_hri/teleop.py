"""리더 -> 팔로워 텔레옵 공통 부분 (2026-10-14).

리더 틱 --(리더 관절 맵)--> 기구학 각도(rad) --(팔로워 관절 맵)--> 팔로워 틱.
두 팔 모두 같은 기구학 규약(kinematics.py)으로 캘리브했으므로 각도가 그대로 대응한다.

그리퍼만 따로: 리더 손잡이는 열리는 범위가 팔로워와 다르므로 리더 [닫힘, 최대 열림] 을
팔로워 [full_close, open] 으로 비례 대응시킨다.

두 가지 방식:
  absolute : 팔로워 = 리더 자세 (일반 텔레옵). 시작할 때 멀리 떨어져 있으면 천천히 붙는다.
  relative : 팔로워 = 기준 자세 + (리더 - 시작 때 리더). 따르기 보정처럼 계획한 자세에서
             조금만 고칠 때 쓴다 — 리더가 팔로워와 다른 자세여도 튀지 않는다.
"""

import math
import os

import yaml

from gaze_hri.so101_driver import JointMap, LeaderBus


class Leader:
    def __init__(self, port, map_file):
        if not port:
            raise RuntimeError("리더 포트가 지정되지 않았습니다 (tools/so101_leader.py ports / assign)")
        if not os.path.exists(map_file):
            raise RuntimeError(f"리더 관절 맵이 없습니다: {map_file} (tools/so101_leader.py capture/solve)")
        c = yaml.safe_load(open(map_file))
        self.map = JointMap(c["zero_ticks"], c["signs"])
        self.grip_max = float(c.get("gripper_open_rad_max", 1.5)) or 1.5
        self.bus = LeaderBus(port)
        self.last_ticks = None

    def read(self):
        """기구학 각도 6개 (rad). 그리퍼는 0(닫힘)~1(최대 열림) 비율로 바꿔서 [5] 에 둔다."""
        t = self.bus.ticks()
        self.last_ticks = list(t)
        q = self.map.to_rad(t)
        q[5] = min(1.0, max(0.0, q[5] / self.grip_max))
        return q

    def close(self):
        self.bus.close()


def grip_from_ratio(r, full_close, g_open):
    return full_close + (g_open - full_close) * r


def step_toward(cur, target, max_step):
    """관절마다 max_step(rad) 이상 한 번에 움직이지 않게 (리더를 휙 움직여도 팔로워는 제한 속도)."""
    return [c + max(-max_step, min(max_step, t - c)) for c, t in zip(cur, target)]


def max_gap_deg(a, b, n=5):
    return max(abs(math.degrees(x - y)) for x, y in zip(a[:n], b[:n]))
