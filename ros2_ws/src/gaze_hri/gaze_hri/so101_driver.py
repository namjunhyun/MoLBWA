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
    # 2026-10-12: 50% -> 90% (사용자 결정, 전 관절). 그리퍼 20cm + 컵을 앞 40cm 수평으로 들면
    # 50% 에서 어깨/팔꿈치가 11° 처지고 옮기다 추종 오차로 안전 정지했다.
    # ★ 그리퍼(6번)도 90% — 컵을 쥔 채 오래 두면 서보가 뜨거워진다 (lerobot 은 그리퍼 50% 로 둔다).
    #    온도 감시(max_temp 60°C)가 마지막 방어선.
    torque_limit: int = 900
    torque_limit_by_id: dict = field(default_factory=dict)   # 관절별 예외 {서보 ID: 한계}
    # 2026-10-12: 서보 내부 가감속/속도 상한을 끈다 (lerobot 텔레옵과 같은 0/0).
    # 예전 accel=20, goal_speed=600 은 PC 가 8mm 마다 새 목표를 줄 때 서보가 구간마다
    # 가속->감속을 다시 해서 덜컥거리는 주원인이었다. 이제 궤적(속도/가속)은 PC 가 촘촘한
    # 스트림으로 만들고, 서보는 그 목표를 그대로 따라간다. 대신 한 번에 크게 튀는 목표는
    # max_step_deg 로 막는다 (서보가 최고속으로 휘두르지 않게).
    accel: int = 0                    # 0 = 서보 내부 가감속 없음
    goal_speed: int = 0               # 0 = 서보 내부 속도 상한 없음
    max_step_deg: float = 8.0         # 직전 목표 대비 한 번 쓰기에 허용하는 최대 변화 [deg]
    limit_margin: int = 30            # EEPROM 위치 한계에서 이만큼 안쪽까지만
    max_track_err: int = 200          # 목표-현재 오차(틱, ≈17.6°) — 이게
    track_err_time: float = 0.6       #   이 시간 넘게 지속되면 충돌/막힘으로 본다
    max_load: int = 950               # 부하 95% 지속 시 정지 (그리퍼는 제외). 토크 90% 에 맞춰 (옛 800)
    max_load_by_id: dict = field(default_factory=dict)
    load_time: float = 0.5
    max_temp: int = 60                # °C
    slow_check_every: int = 10        # 온도/전압은 N 번에 한 번만 읽는다 (버스 부하 감소)
    confirm_n: int = 3                # 온도/전압 이상은 연속 N 번이어야 정지 (한 번 튄 값 무시)
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
        self.raw_lo = [self._r(sid, R_MIN_POS) for sid in self.ids]
        self.raw_hi = [self._r(sid, R_MAX_POS) for sid in self.ids]
        self.lo = [v + self.safety.limit_margin for v in self.raw_lo]
        self.hi = [v - self.safety.limit_margin for v in self.raw_hi]
        self._err_since = None
        self._load_since = None
        self._n_check = 0
        self._bad = {"temp": 0, "volt": 0, "glitch": 0}
        self.last_goal = None
        # 6축 목표를 패킷 하나로 (sync write), 위치+속도+부하(56~61)를 한 번에 (sync read).
        # 예전엔 매 스텝 쓰기 6번 + 읽기 12번(위치/부하) = 왕복 18번이라 루프가 ~70ms 로 늘어지고
        # 주기도 들쭉날쭉했다.
        from scservo_sdk import GroupSyncRead, GroupSyncWrite
        self._gsw = GroupSyncWrite(self.port, self.ph, R_GOAL_POS[0], 2)
        for sid in self.ids:
            self._gsw.addParam(sid, [0, 0])
        self._gsr = GroupSyncRead(self.port, self.ph, R_PRESENT_POS[0], 6)
        for sid in self.ids:
            self._gsr.addParam(sid)
        log(f"[so101] {port} 연결, {v:.1f}V, 한계 {list(zip(self.lo, self.hi))}")

    # ---------- 저수준 ----------
    def _r(self, sid, reg):
        """★ 한 번 실패해도 바로 예외를 던지지 않는다 — 2026-09-27: load 레지스터
        읽기가 한 번 실패해서 arm_server 전체가 죽었다(다단계 캘리브 도중). 접촉
        불량성 순간 오류는 흔하니, flush 후 한 번 더 시도하고 그래도 안 되면 예외."""
        addr, n = reg
        fn = self.ph.read1ByteTxRx if n == 1 else self.ph.read2ByteTxRx
        val, res, _ = fn(self.port, sid, addr)
        if res != self._ok:
            self._flush()
            val, res, _ = fn(self.port, sid, addr)
            if res != self._ok:
                raise RuntimeError(f"ID{sid} 읽기 실패 (addr {addr})")
        return val

    def _w(self, sid, reg, val):
        """_r()과 같은 이유로 한 번 재시도한다 (2026-09-27, 5번 모터 쓰기 실패로 노드 죽음)."""
        addr, n = reg
        assert addr >= 40, "EEPROM 쓰기 금지"          # 40 미만은 EEPROM 영역
        fn = self.ph.write1ByteTxRx if n == 1 else self.ph.write2ByteTxRx
        res, _ = fn(self.port, sid, addr, int(val))     # 쓰기는 (result, error) 2개다
        if res != self._ok:
            self._flush()
            res, _ = fn(self.port, sid, addr, int(val))
        if res != self._ok:
            raise RuntimeError(f"ID{sid} 쓰기 실패 (addr {addr})")

    # ---------- 읽기 ----------
    def ticks(self):
        """현재 위치. ★ 말이 안 되는 값(한계에서 150틱 넘게 벗어남)은 통신 오류로 보고
        한 번 다시 읽는다. 그래도 이상하면 예외 — 엉터리 위치를 hold 목표로 쓰면 멈추려는
        순간 팔이 튄다 (2026-09-23: 이동 중 온도 레지스터가 140°C 로 한 번 잘못 읽혔다)."""
        return self._pos_load()[0]

    def _sync_read_raw(self):
        """sync read 한 번 -> ([위치], [부하]). 실패하면 None (호출자가 개별 읽기로 물러선다)."""
        gsr = getattr(self, "_gsr", None)
        if gsr is None:
            return None
        if gsr.txRxPacket() != self._ok:
            self._flush()
            if gsr.txRxPacket() != self._ok:
                return None
        pos = [gsr.getData(sid, R_PRESENT_POS[0], 2) for sid in self.ids]
        raw = [gsr.getData(sid, R_PRESENT_LOAD[0], 2) for sid in self.ids]
        return pos, [-(v & 0x3FF) if v & 0x400 else v for v in raw]

    def _pos_load(self, need_load=False):
        """위치(+부하). sync read 우선, 값이 이상하면 개별 읽기로 다시 확인한다."""
        got = self._sync_read_raw()
        if got is not None and all(self._plausible_pos(i, v) for i, v in enumerate(got[0])):
            return got
        pos = self._ticks_individual()
        return pos, (self.loads() if need_load else None)

    def _ticks_individual(self):
        out = []
        for i, sid in enumerate(self.ids):
            v = self._r(sid, R_PRESENT_POS)
            if not self._plausible_pos(i, v):
                self._flush()
                v = self._r(sid, R_PRESENT_POS)
                if not self._plausible_pos(i, v):
                    raise RuntimeError(f"통신 이상: ID{sid} 위치 {v} (한계 "
                                       f"{self.raw_lo[i]}~{self.raw_hi[i]})")
            out.append(v)
        return out

    def _plausible_pos(self, i, v):
        lo = getattr(self, "raw_lo", None)
        if lo is None:                               # 한계를 읽기 전 (초기화 중)
            return 0 <= v < TICKS_PER_REV
        return self.raw_lo[i] - 150 <= v <= self.raw_hi[i] + 150

    def _flush(self):
        try:
            self.port.clearPort()
        except Exception:
            pass

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
        """목표=현재로 맞춘 뒤 토크를 켠다 -> 켜는 순간 움직이지 않는다.

        ★ 단, 현재 위치가 EEPROM 위치 한계 **밖**이면 서보 펌웨어가 목표를 한계 안으로
        잘라서 켜는 순간 그쪽으로 민다 (2026-09-23 실기: wrist_roll 38 < 최소 78 에서
        토크를 켜자 롤이 한계 쪽으로 50% 힘으로 버티며 막혔다). 그런 관절이 있으면 켜지 않는다.
        """
        now = self.ticks()
        out = [(JOINT_NAMES[sid - 1] if 1 <= sid <= 6 else sid, t, lo, hi)
               for sid, t, lo, hi in zip(self.ids, now, self.raw_lo, self.raw_hi)
               if not lo <= t <= hi]
        if out:
            raise RuntimeError(
                "토크를 켜지 않는다 — 가동 범위 밖 관절: "
                + ", ".join(f"{n} 위치 {t} (범위 {lo}~{hi})" for n, t, lo, hi in out)
                + ". 손으로 범위 안쪽으로 옮긴 뒤 다시.")
        for sid, t in zip(self.ids, now):
            self._w(sid, R_TORQUE_LIMIT,
                    self.safety.torque_limit_by_id.get(sid, self.safety.torque_limit))
            self._w(sid, R_ACCEL, self.safety.accel)
            self._w(sid, R_GOAL_SPEED, self.safety.goal_speed)
            self._w(sid, R_GOAL_POS, t)
        for sid in self.ids:
            self._w(sid, R_TORQUE_ENABLE, 1)
        self.last_goal = now
        extra = "".join(f", ID{k} {v / 10:.0f}%" for k, v in self.safety.torque_limit_by_id.items())
        self.log(f"[so101] 토크 ON (한계 {self.safety.torque_limit / 10:.0f}%{extra}), 현재 자세 유지")

    def hold(self):
        """지금 있는 자리에 세운다. 토크는 유지 (끄면 팔이 떨어진다).

        현재 위치를 못 믿겠으면(통신 이상) 마지막으로 보낸 목표를 다시 쓴다 — 엉터리
        값을 목표로 쓰는 것보다 방금까지 가던 곳에 서는 게 낫다."""
        try:
            now = self.ticks()
        except Exception as e:
            self.log(f"[so101] hold: 위치를 못 믿음({e}) -> 마지막 목표 유지")
            now = self.last_goal
        if now is None:
            return
        try:
            for sid, t in zip(self.ids, now):
                self._w(sid, R_GOAL_POS, t)
            self.last_goal = list(now)
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
        clamped = [min(max(int(round(g)), lo), hi) for g, lo, hi in zip(goal, self.lo, self.hi)]
        if self.last_goal is not None:
            jump = max(abs(a - b) for a, b in zip(clamped, self.last_goal))
            if jump * 360 / TICKS_PER_REV > self.safety.max_step_deg:
                self._stop(f"한 번에 {jump * 360 / TICKS_PER_REV:.1f}° 점프하는 목표 거부 "
                           f"(상한 {self.safety.max_step_deg}°) — 보간 없이 쓴 명령")
        self._write_goal(clamped)
        self.last_goal = clamped
        self.check()
        return clamped

    def _write_goal(self, ticks):
        gsw = getattr(self, "_gsw", None)
        if gsw is None:                           # 테스트용 조립(시리얼 없음)
            for sid, g in zip(self.ids, ticks):
                self._w(sid, R_GOAL_POS, g)
            return
        for sid, g in zip(self.ids, ticks):
            gsw.changeParam(sid, [g & 0xFF, (g >> 8) & 0xFF])
        if gsw.txPacket() != self._ok:            # 응답 없는 패킷이라 실패 = 포트 문제
            self._flush()
            if gsw.txPacket() != self._ok:
                self._stop("sync write 실패 (포트)")

    def check(self):
        s = self.safety
        now_t = time.monotonic()
        try:
            pos, loads = self._pos_load(need_load=True)
        except RuntimeError as e:                 # 통신 이상 -> 일반 예외로 새지 않게 정지
            self._stop(str(e))
        self.last_pos = list(pos)                 # 궤적 기록용 (arm_server traj_log_dir)
        self.last_load = list(loads) if loads else None
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
        over = [(abs(l), sid) for l, sid in zip(loads, self.ids)
                if sid in s.check_ids and abs(l) > s.max_load_by_id.get(sid, s.max_load)]
        if over:
            self._load_since = self._load_since or now_t
            if now_t - self._load_since > s.load_time:
                l, sid = max(over)
                self._stop(f"ID{sid} 부하 {l / 10:.0f}% 지속")
        else:
            self._load_since = None
        self._n_check += 1
        if self._n_check % s.slow_check_every == 0:
            self._slow_checks()
        return pos

    def _slow_checks(self):
        """온도/전압. 물리적으로 불가능한 값은 통신 오류로 따로 세고, 진짜 이상은 연속
        confirm_n 번일 때만 정지한다. 통신 오류가 계속되면 그것 자체로 정지."""
        s = self.safety
        temps, v = self.temps(), self.voltage()
        glitch = any(t > 100 or t <= 0 for t in temps) or not (3.0 < v < 20.0)
        if glitch:
            self._bad["glitch"] += 1
            self.log(f"[so101] 이상한 값 무시 (통신 오류?): 온도 {temps}, {v:.1f}V")
            if self._bad["glitch"] >= 5:
                self._stop("온도/전압 읽기가 계속 비정상 — 통신 이상")
            return
        self._bad["glitch"] = 0
        for key, bad, msg in (("temp", max(temps) > s.max_temp, f"서보 온도 {max(temps)}°C"),
                              ("volt", v < s.min_voltage, f"전압 강하 {v:.1f}V")):
            self._bad[key] = self._bad[key] + 1 if bad else 0
            if self._bad[key] >= s.confirm_n:
                self._stop(msg + f" (연속 {s.confirm_n}회)")

    def _stop(self, why):
        self.hold()
        self.log(f"[so101] ★ 안전 정지: {why}")
        raise SafetyStop(why)

    def close(self):
        try:
            self.port.closePort()
        except Exception:
            pass


def joint_limits_rad(bus: So101Bus, jmap: JointMap, names=JOINT_NAMES[:4]):
    """서보 EEPROM 위치 한계(여유 limit_margin 포함)를 기구학 각도 한계로 환산.

    2026-10-12: 팔로워가 2대라 kinematics.ArmGeometry 의 고정 한계(a 팔 기준 환산값)를
    그대로 쓸 수 없다. 팔마다 서보 한계와 zero_ticks 가 달라서 실제 범위도 다르다."""
    lo = jmap.to_rad(bus.lo)
    hi = jmap.to_rad(bus.hi)
    return {n: (min(a, b), max(a, b)) for n, a, b in zip(names, lo, hi)}


def move_smooth(bus: So101Bus, target_ticks, max_deg_s=20.0, rate_hz=50.0,
                should_stop=lambda: False):
    """현재 목표에서 target 까지 관절 공간 코사인 보간. 가장 많이 움직이는 관절이
    max_deg_s 를 넘지 않게 시간을 잡는다. 매 스텝 감시(SafetyStop)."""
    start = bus.last_goal or bus.ticks()
    span = max(abs(t - s) for t, s in zip(target_ticks, start))
    dur = max(0.3, span * 360 / TICKS_PER_REV / max_deg_s)
    n = max(1, int(dur * rate_hz))
    t0 = time.monotonic()
    for i in range(1, n + 1):
        if should_stop():
            bus.hold()
            return False
        a = min_jerk(i / n)
        bus.write_ticks([s + (t - s) * a for s, t in zip(start, target_ticks)])
        # 고정 sleep 이 아니라 마감 시각까지 대기 -> 읽기/쓰기 시간이 흔들려도 주기 일정
        time.sleep(max(0.0, t0 + i / rate_hz - time.monotonic()))
    return True


def settle_ticks(bus: So101Bus, goal, joints=(1, 2, 3), max_deg=6.0, iters=2, wait=0.25):
    """처짐 보정 (틱 단위, ROS 없는 도구용). arm_server 의 FeetechBackend.settle 과 같은 원리:
    도착 후 실제 위치를 읽어 모자란 만큼 목표에 덧대 다시 쓴다. 덧댄 목표를 돌려준다."""
    lim = max_deg * TICKS_PER_REV / 360.0
    cmd = list(bus.last_goal)
    for _ in range(iters):
        time.sleep(wait)
        pos = bus.ticks()
        new = list(cmd)
        for j in joints:
            new[j] = int(round(goal[j] + max(-lim, min(lim, cmd[j] - goal[j] + goal[j] - pos[j]))))
        if max(abs(goal[j] - pos[j]) for j in joints) < 4:      # 0.35° 안이면 끝
            break
        for k in range(1, 11):
            bus.write_ticks([c + (n - c) * k / 10 for c, n in zip(cmd, new)])
            time.sleep(0.02)
        cmd = new
    return cmd


def min_jerk(tau):
    """최소 저크 보간 0->1. 시작/끝에서 속도·가속도가 모두 0 이라 사람 손 동작에 가깝다
    (코사인 보간은 끝에서 가속도가 0 이 아니라 멈출 때 '툭' 하고 선다)."""
    tau = min(max(tau, 0.0), 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau * tau)
