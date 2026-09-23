"""so101_driver 안전장치 검증 — 실제 서보 없이 레지스터를 흉내 낸다.

실기에서 처음 시험하면 안 되는 것들:
  * 리더 팔(5V)에 연결하면 거부하는가
  * 토크를 켤 때 목표가 현재 위치로 먼저 맞춰지는가 (켜는 순간 튀지 않는가)
  * EEPROM(주소 < 40) 쓰기가 막혀 있는가
  * 목표가 위치 한계 밖이면 잘리는가
  * 막힘(목표를 못 따라감)이 지속되면 그 자리에 서는가 (SafetyStop)
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gaze_hri import so101_driver as d  # noqa: E402


class FakeServos:
    """주소별 레지스터 값을 가진 가짜 버스. 목표를 주면 follow=True 일 때 즉시 따라간다."""

    def __init__(self, volts=12.3, pos=2000, follow=True):
        self.reg = {sid: {9: 1000, 11: 3000, 40: 0, 42: 0, 56: pos + sid, 60: 0,
                          62: int(volts * 10), 63: 30} for sid in range(1, 7)}
        self.follow = follow
        self.writes = []

    def read(self, sid, addr):
        return self.reg[sid][addr]

    def write(self, sid, addr, val):
        self.writes.append((sid, addr, val))
        self.reg[sid][addr] = val
        if addr == 42 and self.follow:
            self.reg[sid][56] = val


def make_bus(fake, **safety):
    bus = d.So101Bus.__new__(d.So101Bus)          # 시리얼 없이 조립
    bus.log = lambda m: None
    bus.ids = [1, 2, 3, 4, 5, 6]
    bus.safety = d.SafetyConfig(**safety)
    bus._err_since = bus._load_since = None
    bus.last_goal = None
    bus._r = lambda sid, reg: fake.read(sid, reg[0])

    def _w(sid, reg, val):
        assert reg[0] >= 40, "EEPROM 쓰기 금지"
        fake.write(sid, reg[0], int(val))
    bus._w = _w
    bus.lo = [1000 + bus.safety.limit_margin] * 6
    bus.hi = [3000 - bus.safety.limit_margin] * 6
    return bus


class TestSo101Driver(unittest.TestCase):
    def test_jointmap_roundtrip(self):
        m = d.JointMap(zero=[2048, 900, 3100, 2600, 2000, 2030], sign=[1, -1, 1, -1, 1, 1])
        q = [0.3, 1.2, -1.5, -0.4, 0.0, 0.8]
        back = m.to_rad(m.to_ticks(q))
        for a, b in zip(q, back):
            self.assertAlmostEqual(a, b, delta=d.RAD_PER_TICK)

    def test_enable_sets_goal_to_present_before_torque(self):
        f = FakeServos()
        bus = make_bus(f)
        bus.enable()
        for sid in range(1, 7):
            goal_idx = max(i for i, w in enumerate(f.writes) if w[0] == sid and w[1] == 42)
            torque_idx = min(i for i, w in enumerate(f.writes) if w[0] == sid and w[1] == 40)
            self.assertLess(goal_idx, torque_idx)                 # 목표 먼저, 토크 나중
            self.assertEqual(f.reg[sid][42], 2000 + sid)          # 목표 = 현재
        self.assertTrue(all(w[1] >= 40 for w in f.writes))        # EEPROM 안 건드림

    def test_eeprom_write_blocked(self):
        bus = make_bus(FakeServos())
        with self.assertRaises(AssertionError):
            bus._w(1, d.R_MIN_POS, 0)

    def test_clamp_to_limits(self):
        f = FakeServos()
        bus = make_bus(f)
        bus.enable()
        out = bus.write_ticks([0, 5000, 2000, 2000, 2000, 2000])
        self.assertEqual(out[0], 1030)
        self.assertEqual(out[1], 2970)

    def test_stall_triggers_hold(self):
        f = FakeServos(follow=False)             # 서보가 목표를 못 따라감 (막힘)
        bus = make_bus(f, track_err_time=0.1)
        bus.enable()
        with self.assertRaises(d.SafetyStop):
            for _ in range(20):
                bus.write_ticks([2600] * 6)      # 약 53° 떨어진 목표
                time.sleep(0.02)
        for sid in range(1, 7):                  # 정지 후 목표 = 현재 (그 자리에 섬)
            self.assertEqual(f.reg[sid][42], f.reg[sid][56])
        self.assertTrue(all(f.reg[s][40] == 1 for s in range(1, 7)))   # 토크는 유지

    def test_leader_voltage_rejected_by_check(self):
        f = FakeServos(volts=5.2)
        bus = make_bus(f)
        bus.last_goal = [2000] * 6
        with self.assertRaises(d.SafetyStop):
            bus.check()


if __name__ == "__main__":
    unittest.main(verbosity=2)
