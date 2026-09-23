"""SO-101 팔로워 저수준 드라이버 — 안전장치 우선 (2026-09-23).

arm_server 의 예전 FeetechBackend 는 Feetech 공식 SDK 의 sms_sts 클래스를 가정했는데,
pip 의 feetech-servo-sdk(scservo_sdk)에는 그 클래스가 없다. 게다가
  * "2048틱 = 기구학 0도, 방향 전부 +" 를 가정했다 — 실제 팔은 lerobot 캘리브레이션
    (homing offset)이 들어가 있어 이 가정이 검증된 적이 없다
  * 기본 포트 /dev/ttyACM0 이 이 책상에서는 **리더 팔**(5V)이었다
그래서 레지스터를 직접 읽고 쓰는 드라이버로 다시 짰다. ROS 의존 없음.

원칙:
  * EEPROM 에는 절대 쓰지 않는다. RAM 레지스터(토크/가속/목표/속도/토크한계)만.
    lerobot 캘리브레이션(Homing_Offset, 위치 한계)이 그대로 보존된다.
  * 버스 전압이 min_voltage 미만이면 연결을 거부한다 (리더 팔 오인 방지).
  * 토크를 켜기 전에 목표 = 현재 위치로 맞춘다 (켜는 순간 튀지 않게).
  * 목표 틱은 서보 EEPROM 의 위치 한계 안쪽(margin)으로 자른다.
  * 매 쓰기마다 감시: 목표-현재 오차 지속(충돌/막힘), 부하, 온도, 전압.
    이상하면 즉시 hold(목표=현재) 하고 SafetyStop 을 던진다. 토크는 끄지 않는다 —
    끄면 팔이 중력으로 떨어진다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

TICKS_PER_REV = 4096
RAD_PER_TICK = 2 * math.pi / TICKS_PER_REV

# STS3215 레지스터 (주소, 바이트). lerobot feetech 테이블과 같다.
R_MIN_POS = (9, 2)
R_MAX_POS = (11, 2)
R_TORQUE_ENABLE = (40, 1)       # RAM
R_ACCEL = (41, 1)               # RAM
R_GOAL_POS = (42, 2)            # RAM
R_GOAL_SPEED = (46, 2)          # RAM (틱/s, 0 = 최대)
R_TORQUE_LIMIT = (48, 2)        # RAM (0~1000 = 0~100%)
R_PRESENT_POS = (56, 2)
R_PRESENT_LOAD = (60, 2)        # 부호-크기 (bit 10), 0~1000
R_VOLTAGE = (62, 1)             # 0.1V
R_TEMP = (63, 1)                # °C

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
               "wrist_roll", "gripper"]


class SafetyStop(RuntimeError):
    """감시가 이상을 잡아 팔을 그 자리에 세웠다."""


@dataclass
class JointMap:
    """서보 틱 <-> gaze_hri 기구학 각도(rad).

    rad = sign * (ticks - zero) * 2pi/4096
    zero/sign 은 tools/so101_map_calib.py 로 실측한다 (가정하지 않는다).
    """
    zero: list
    sign: list

    def to_rad(self, ticks):
        return [s * (t - z) * RAD_PER_TICK for t, z, s in zip(ticks, self.zero, self.sign)]

    def to_ticks(self, rad):
        return [int(round(z + s * r / RAD_PER_TICK)) for r, z, s in zip(rad, self.zero, self.sign)]


@dataclass
class SafetyConfig:
    min_voltage: float = 9.0          # 12V 팔로워만. 리더(5V)는 여기서 걸린다
    torque_limit: int = 500           # 50%. 부딪혀도 덜 아프게
    accel: int = 20                   # 서보 가속 (0~254, 작을수록 부드럽게)
    goal_speed: int = 600             # 틱/s ≈ 53°/s 상한 (서보 내부)
    limit_margin: int = 30            # EEPROM 위치 한계에서 이만큼 안쪽까지만
    max_track_err: int = 200          # 목표-현재 오차(틱, ≈17.6°) — 이게
    track_err_time: float = 0.6       #   이 시간 넘게 지속되면 충돌/막힘으로 본다
    max_load: int = 800               # 부하 80% 지속 시 정지 (그리퍼는 제외)
    load_time: float = 0.5
    max_temp: int = 60                # °C
    check_ids: list = field(default_factory=lambda: [1, 2, 3, 4, 5])  # 그리퍼는 물면 부하가 정상


class So101Bus:
    def __init__(self, port: str, ids=(1, 2, 3, 4, 5, 6), safety: SafetyConfig | None = None,
                 log=print):
        from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler
        self._ok = COMM_SUCCESS
        self.log = log
        self.ids = list(ids)
        self.safety = safety or SafetyConfig()
        self.port = PortHandler(port)
        if not self.port.openPort() or not self.port.setBaudRate(1_000_000):
            raise RuntimeError(f"시리얼 포트 열기 실패: {port}")
        self.ph = PacketHandler(0)             # STS 계열 protocol_end = 0
        self.port_name = port

        for sid in self.ids:
            _, res, _ = self.ph.ping(self.port, sid)
            if res != self._ok:
                self.close()
                raise RuntimeError(f"서보 ID{sid} 응답 없음 ({port})")
        v = self.voltage()
        if v < self.safety.min_voltage:
            self.close()
            raise RuntimeError(
                f"{port} 버스 전압 {v:.1f}V < {self.safety.min_voltage}V — 리더 팔(5V)이거나 "
                "팔로워 전원이 꺼져 있다. 연결 거부.")
        self.lo = [self._r(sid, R_MIN_POS) + self.safety.limit_margin for sid in self.ids]
        self.hi = [self._r(sid, R_MAX_POS) - self.safety.limit_margin for sid in self.ids]
        self._err_since = None
        self._load_since = None
        self.last_goal = None
        log(f"[so101] {port} 연결, {v:.1f}V, 한계 {list(zip(self.lo, self.hi))}")

    # ---------- 저수준 ----------
    def _r(self, sid, reg):
        addr, n = reg
        fn = self.ph.read1ByteTxRx if n == 1 else self.ph.read2ByteTxRx
        val, res, _ = fn(self.port, sid, addr)
        if res != self._ok:
            raise RuntimeError(f"ID{sid} 읽기 실패 (addr {addr})")
        return val

    def _w(self, sid, reg, val):
        addr, n = reg
        assert addr >= 40, "EEPROM 쓰기 금지"          # 40 미만은 EEPROM 영역
        fn = self.ph.write1ByteTxRx if n == 1 else self.ph.write2ByteTxRx
        _, res, _ = fn(self.port, sid, addr, int(val))
        if res != self._ok:
            raise RuntimeError(f"ID{sid} 쓰기 실패 (addr {addr})")

    # ---------- 읽기 ----------
    def ticks(self):
        return [self._r(sid, R_PRESENT_POS) for sid in self.ids]

    def loads(self):
        out = []
        for sid in self.ids:
            v = self._r(sid, R_PRESENT_LOAD)
            out.append(-(v & 0x3FF) if v & 0x400 else v)
        return out

    def voltage(self):
        return self._r(self.ids[0], R_VOLTAGE) / 10.0

    def temps(self):
        return [self._r(sid, R_TEMP) for sid in self.ids]

    # ---------- 토크 ----------
    def enable(self):
        """목표=현재로 맞춘 뒤 토크를 켠다 -> 켜는 순간 움직이지 않는다."""
        now = self.ticks()
        for sid, t in zip(self.ids, now):
            self._w(sid, R_TORQUE_LIMIT, self.safety.torque_limit)
            self._w(sid, R_ACCEL, self.safety.accel)
            self._w(sid, R_GOAL_SPEED, self.safety.goal_speed)
            self._w(sid, R_GOAL_POS, t)
        for sid in self.ids:
            self._w(sid, R_TORQUE_ENABLE, 1)
        self.last_goal = now
        self.log(f"[so101] 토크 ON (한계 {self.safety.torque_limit / 10:.0f}%), 현재 자세 유지")

    def hold(self):
        """지금 있는 자리에 세운다. 토크는 유지 (끄면 팔이 떨어진다)."""
        try:
            now = self.ticks()
            for sid, t in zip(self.ids, now):
                self._w(sid, R_GOAL_POS, t)
            self.last_goal = now
        except Exception as e:                  # 통신이 죽었으면 할 수 있는 게 없다
            self.log(f"[so101] hold 실패: {e}")

    def relax(self, ids=None):
        """토크를 끈다. ★ 팔이 중력으로 떨어진다 — 사람이 받치고 있을 때만."""
        for sid in (ids or self.ids):
            try:
                self._w(sid, R_TORQUE_ENABLE, 0)
            except Exception as e:
                self.log(f"[so101] ID{sid} 토크 OFF 실패: {e}")

    # ---------- 쓰기 + 감시 ----------
    def write_ticks(self, goal):
        """목표 틱 쓰기. 한계로 자르고, 쓴 직후 상태를 감시한다."""
        clamped = [min(max(int(g), lo), hi) for g, lo, hi in zip(goal, self.lo, self.hi)]
        for sid, g in zip(self.ids, clamped):
            self._w(sid, R_GOAL_POS, g)
        self.last_goal = clamped
        self.check()
        return clamped

    def check(self):
        s = self.safety
        now_t = time.monotonic()
        pos = self.ticks()
        if self.last_goal is not None:
            errs = [abs(p - g) for p, g, sid in zip(pos, self.last_goal, self.ids)
                    if sid in s.check_ids]
            if max(errs) > s.max_track_err:
                self._err_since = self._err_since or now_t
                if now_t - self._err_since > s.track_err_time:
                    self._stop(f"목표를 못 따라감 (최대 오차 {max(errs)}틱 ≈ "
                               f"{max(errs) * 360 / TICKS_PER_REV:.0f}°) — 충돌/막힘?")
            else:
                self._err_since = None
        loads = self.loads()
        big = [abs(l) for l, sid in zip(loads, self.ids) if sid in s.check_ids]
        if max(big) > s.max_load:
            self._load_since = self._load_since or now_t
            if now_t - self._load_since > s.load_time:
                self._stop(f"부하 {max(big) / 10:.0f}% 지속")
        else:
            self._load_since = None
        t = max(self.temps())
        if t > s.max_temp:
            self._stop(f"서보 온도 {t}°C")
        v = self.voltage()
        if v < s.min_voltage:
            self._stop(f"전압 강하 {v:.1f}V")
        return pos

    def _stop(self, why):
        self.hold()
        self.log(f"[so101] ★ 안전 정지: {why}")
        raise SafetyStop(why)

    def close(self):
        try:
            self.port.closePort()
        except Exception:
            pass


def move_smooth(bus: So101Bus, target_ticks, max_deg_s=20.0, rate_hz=25.0,
                should_stop=lambda: False):
    """현재 목표에서 target 까지 관절 공간 코사인 보간. 가장 많이 움직이는 관절이
    max_deg_s 를 넘지 않게 시간을 잡는다. 매 스텝 감시(SafetyStop)."""
    start = bus.last_goal or bus.ticks()
    span = max(abs(t - s) for t, s in zip(target_ticks, start))
    dur = max(0.3, span * 360 / TICKS_PER_REV / max_deg_s)
    n = max(1, int(dur * rate_hz))
    for i in range(1, n + 1):
        if should_stop():
            bus.hold()
            return False
        a = 0.5 - 0.5 * math.cos(math.pi * i / n)
        bus.write_ticks([s + (t - s) * a for s, t in zip(start, target_ticks)])
        time.sleep(1.0 / rate_hz)
    return True
