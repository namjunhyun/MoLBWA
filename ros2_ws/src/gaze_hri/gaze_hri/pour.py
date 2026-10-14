"""물 따르기 경로 계획 (2026-10-14) — ROS 없는 순수 함수.

병은 수평 옆 집기로 쥔다 (그리퍼 수평, 접근 방향 = 받침대에서 바깥쪽 반경 방향).
이 자세에서 손목 roll(5번)을 돌리면 병이 반경 축을 중심으로 옆(접선 방향)으로 기운다.
그래서 컵은 병의 **옆**(접선 방향)에 두고, 병 입구가 컵 위에 오도록 손목을 굴린다.

roll 규약 (kinematics 와 같은 방향 규칙으로 정함): 받침대에서 그리퍼 쪽을 바라볼 때
**반시계 = +**. 이때 병 축(쥔 곳 -> 입구) a(θ) = cosθ·z + sinθ·t, t = 팔 기준 왼쪽 접선.
roll = s·θ (s = +1 이면 왼쪽으로, -1 이면 오른쪽으로 기운다).

사람이 따르는 방식을 흉내 낸다:
  1) 병을 세운 채 컵 옆으로 가져가서, 물이 안 나오는 각도(pre)까지만 기울인 채
  2) 입구를 컵 테두리 안쪽 위(M1)로 옮기고 (멈춰서 처짐을 맞춘다)
  3) '물이 입구에 닿는 각도'(onset)까지는 빨리, 그 뒤로는 입구를 M1 에 고정한 채 천천히 더
     기울인다 (물이 줄수록 더 기울여야 나온다)
  4) 잠깐 멈췄다가 **컵 위에서 먼저 pre 까지 세우고** 옆으로 빼서 똑바로 세운 뒤 제자리에 둔다
     (기운 채로 옮기면 물이 흐르는 중에 컵 밖으로 나간다).
입구를 고정점으로 돌리기 때문에 손목만 돌리는 게 아니라 그리퍼(쥔 곳)가 원호를 그린다.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PourConfig:
    bottle_height: float = 0.20      # 병 바닥 ~ 입구 [m]
    grasp_height: float = 0.06       # 병 바닥에서 쥔 곳까지 (= arm_server grasp_height)
    bottle_radius: float = 0.035
    neck_radius: float = 0.015       # 위쪽 neck_frac 구간 (어깨~입구) 반지름
    neck_frac: float = 0.3
    cup_height: float = 0.145
    cup_rim_radius: float = 0.045
    side_offset: float = 0.11        # 시작 자세: 병 축 ~ 컵 축 수평 거리
    lip_inset: float = 0.015         # 입구를 컵 테두리에서 이만큼 안쪽에
    lip_clear: float = 0.025         # 입구 높이 = 컵 높이 + 이것
    body_clear: float = 0.02         # 병 바닥이 언제나 컵 높이보다 이만큼 위
    table_clear: float = 0.01        # 병 바닥이 책상보다 이만큼 위
    pre_deg: float = 30.0            # 컵 옆에서 이 각도까지만 (물이 안 나오는 각도) 기울인 뒤 컵 위로
    onset_deg: float = 50.0          # 컵 위에서 여기까지 빠르게 (물이 입구에 닿는 각도 근처)
    max_deg: float = 105.0           # 여기까지 천천히
    fast_deg_s: float = 20.0
    slow_deg_s: float = 6.0
    hold_s: float = 1.5
    step_deg: float = 2.0            # 경로 점 간격
    move_speed: float = 0.03         # 기울인 채 입구를 옮기는 속도 [m/s]


def tangent(p):
    """p 위치에서 팔 기준 왼쪽 접선 (단위, z=0)."""
    phi = math.atan2(p[1], p[0])
    return np.array([-math.sin(phi), math.cos(phi), 0.0])


def axis(theta, s, grip, pitch=0.0):
    """병 축(쥔 곳 -> 입구). theta = 손목 roll 크기, s = 기우는 쪽.

    2026-10-14: 병을 그리퍼를 숙여(pitch<0) 쥐었으면 병은 수직이지만 roll 축(그리퍼 축)이
    숙어 있다. roll 은 그 축을 중심으로 z 를 돌린다 (Rodrigues). pitch=0 이면
    cosθ·z + sinθ·s·t 와 같다 (오른손 회전 -sθ: r̂ × z = -t)."""
    t = tangent(grip)
    r_hat = np.array([t[1], -t[0], 0.0])                  # 반경 방향 (팔 바깥)
    g = math.cos(pitch) * r_hat + np.array([0.0, 0.0, math.sin(pitch)])
    z = np.array([0.0, 0.0, 1.0])
    ang = -s * theta
    return (z * math.cos(ang) + np.cross(g, z) * math.sin(ang)
            + g * float(np.dot(g, z)) * (1.0 - math.cos(ang)))


def tilt_of_roll(theta, pitch):
    """roll theta 일 때 병이 수직에서 기운 각 (cosφ = sin²p + cos²p·cosθ)."""
    c = math.sin(pitch) ** 2 + math.cos(pitch) ** 2 * math.cos(theta)
    return math.acos(max(-1.0, min(1.0, c)))


def roll_for_tilt(phi, pitch):
    """병을 phi 만큼 기울이려면 필요한 roll. 숙여 쥐면 같은 기울기에 roll 이 더 든다."""
    c2 = math.cos(pitch) ** 2
    v = (math.cos(phi) - math.sin(pitch) ** 2) / max(c2, 1e-9)
    if v < -1.0 - 1e-9:
        raise PourError(f"그리퍼를 {abs(math.degrees(pitch)):.0f}° 숙여 쥐면 병을 "
                        f"{math.degrees(phi):.0f}° 까지 못 기울입니다")
    return math.acos(max(-1.0, min(1.0, v)))


def grip_for_mouth(mouth, theta, s, L, iters=8, pitch=0.0):
    """입구가 mouth 에 오도록 하는 쥔 곳 G. 접선 방향이 G 의 위치에 따라 바뀌므로 고정점 반복."""
    g = np.asarray(mouth, float) - L * np.array([0.0, 0.0, 1.0])
    for _ in range(iters):
        g = np.asarray(mouth, float) - L * axis(theta, s, g, pitch)
    return g


class PourError(ValueError):
    pass


def check_clearance(g, theta, s, cfg, cup, pitch=0.0):
    """병(선분)이 책상/컵과 겹치지 않는지. 문제 있으면 사유 문자열."""
    a = axis(theta, s, g, pitch)
    bottom = g - cfg.grasp_height * a
    mouth = g + (cfg.bottle_height - cfg.grasp_height) * a
    if bottom[2] < cfg.table_clear * 0.5:
        return f"병 바닥이 책상 아래로 ({bottom[2] * 100:.1f}cm, θ={math.degrees(theta):.0f}°)"
    for k in np.linspace(0.0, 1.0, 21):
        p = bottom + (mouth - bottom) * k
        if p[2] < cfg.cup_height + 0.005:
            d = math.hypot(p[0] - cup[0], p[1] - cup[1])
            r = cfg.neck_radius if k > 1.0 - cfg.neck_frac else cfg.bottle_radius
            if d < cfg.cup_rim_radius + r - 0.005:
                return (f"병 몸통이 컵에 닿음 (높이 {p[2] * 100:.1f}cm, 컵 축에서 {d * 100:.1f}cm, "
                        f"θ={math.degrees(theta):.0f}°)")
    return None


def plan_pour(bottle, cup, cfg: PourConfig, end_shift=None, pitch=0.0):
    """bottle: 병을 쥔 곳 (x,y,z) — 옆 집기 파지점. cup: 컵 중심 (x,y), 책상 기준.

    돌려주는 것: dict
      s          : 기우는 방향 (+1 왼쪽 / -1 오른쪽)
      carry      : 세운 채 옮겨 갈 쥔 곳 (컵 옆, 높이 = 컵 위를 지나는 높이)
      start      : 세운 채 내려앉은 시작 쥔 곳
      segments   : {"pre","move","onset","pour"} 각각 [(G, 손목 roll θ, 구간 시간 dt)]
    pitch: 병을 쥔 그리퍼 각도 (숙여 쥐었으면 음수). 옮기고 따르는 내내 이 각도를 유지해 병을
           수직(roll 0)으로 둔다.
    end_shift: 따르는 동안 입구를 시작 위치에서 이만큼(xy, m) 옮겨 끝낸다 (보정: 시작/종료
               자세에서 따로 맞춘 위치. 기울수록 처짐이 달라 한 점으로는 둘 다 못 맞춘다).
    """
    cup = np.array([cup[0], cup[1], 0.0], float)
    bottle = np.asarray(bottle, float)
    L = cfg.bottle_height - cfg.grasp_height
    tc = tangent(cup)
    # 병이 오는 쪽에 세운다 -> 기우는 방향은 병 쪽에서 컵 쪽 = s·t
    s = 1.0 if float(np.dot(bottle[:2] - cup[:2], tc[:2])) < 0 else -1.0

    side = cup - cfg.side_offset * s * tc
    # 2026-10-14(3): 병 몸통 전체를 컵 테두리 + body_clear 위로 유지한다 (사용자: "계속 컵을 쳐서
    # 컵이 밀린다"). 실제 위치 오차(~7cm)가 있어도 몸통이 컵 높이 아래로 내려가지 않으면 안 친다.
    rim_clear = cfg.cup_height + cfg.body_clear           # 병 바닥이 이보다 위
    mouth_hi = rim_clear + cfg.bottle_height              # 세운 병의 입구 높이
    start = np.array([side[0], side[1], mouth_hi - L])
    carry = start.copy()

    m0 = np.array([side[0], side[1], mouth_hi])
    m1 = cup - (cfg.cup_rim_radius - cfg.lip_inset) * s * tc
    m1[2] = cfg.cup_height + cfg.lip_clear

    pre, on, mx = (math.radians(cfg.pre_deg), math.radians(cfg.onset_deg),
                   math.radians(cfg.max_deg))
    st = math.radians(cfg.step_deg)

    # 각도(pre/onset/max)는 병이 수직에서 기운 각. 경로 점에는 손목 roll 로 바꿔 넣는다
    # (그리퍼를 숙여 쥐었으면 같은 기울기에 roll 이 더 든다).
    def rl(phi):
        return roll_for_tilt(phi, pitch)

    def gm(m, phi):
        return grip_for_mouth(m, rl(phi), s, L, pitch=pitch)

    def tilt_at(m, th0, th1, deg_s):
        n = max(1, int(math.ceil(abs(th1 - th0) / st)))
        return [(gm(m, th0 + (th1 - th0) * i / n), rl(th0 + (th1 - th0) * i / n),
                 (abs(th1 - th0) / n) / math.radians(deg_s)) for i in range(1, n + 1)]

    # 2026-10-14(2): 순서를 바꿨다. 예전엔 컵 옆(M0)에서 60° 까지 기울인 뒤 입구를 컵 위로
    # 옮겨서, 물이 있으면 컵 밖에서 이미 흘렀다 (사용자: "처음엔 밖으로 따르다 컵 쪽으로 온다").
    # (1) 컵 옆에서 물이 안 나오는 각도(pre)까지만 기울이고
    seg_pre = tilt_at(m0, 0.0, pre, cfg.fast_deg_s)
    # (2) 그 각도로 **높이 그대로** 입구를 컵 위로 옮긴 다음 — 여기서 멈춰 처짐을 맞춘다
    m1_hi = np.array([m1[0], m1[1], mouth_hi])
    d = float(np.linalg.norm(m1_hi - m0))
    n2 = max(2, int(math.ceil(d / 0.01)))
    seg_move = [(gm(m0 + (m1_hi - m0) * (i / n2), pre), rl(pre),
                 (d / n2) / cfg.move_speed) for i in range(1, n2 + 1)]

    # (3) 컵 위에서 onset 까지 기울이며 입구를 내린다. 병 바닥(입구 - H cosθ)이 rim_clear 아래로
    #     안 가는 만큼만 내린다 -> 입구는 onset 근처에서야 따를 높이(M1)에 닿는다.
    def mouth_z(th):
        return max(m1[2], rim_clear + cfg.bottle_height * math.cos(th))
    n3 = max(1, int(math.ceil((on - pre) / st)))
    seg_onset = []
    for i in range(1, n3 + 1):
        th = pre + (on - pre) * i / n3
        m = np.array([m1[0], m1[1], mouth_z(th)])
        seg_onset.append((gm(m, th), rl(th),
                          ((on - pre) / n3) / math.radians(cfg.fast_deg_s)))
    # onset 이 작으면(보정으로 정한 시작 각도) 그 각도에서는 입구가 M1 보다 조금 높다 — 병 바닥을
    # 컵 위에 두기 위해서. 따르는 구간에서 기울수록 M1 높이로 내려온다.
    sh = np.zeros(3) if end_shift is None else np.array([end_shift[0], end_shift[1], 0.0])
    n4 = max(1, int(math.ceil((mx - on) / st)))
    seg_pour = []
    for i in range(1, n4 + 1):
        th = on + (mx - on) * i / n4
        m = m1 + sh * (i / n4)
        m[2] = mouth_z(th)
        seg_pour.append((gm(m, th), rl(th),
                         ((mx - on) / n4) / math.radians(cfg.slow_deg_s)))
    # 되돌릴 때는 (4)(3) 을 거꾸로 = 컵 위에서 먼저 세우고(pre), 그 다음에 옆으로 뺀다.

    for g, th, _ in [(start, 0.0, 0)] + seg_pre + seg_move + seg_onset + seg_pour:
        why = check_clearance(g, th, s, cfg, cup, pitch)
        if why:
            raise PourError(why)
    return {"s": s, "carry": carry, "start": start, "mouth": m1, "pitch": pitch,
            "segments": {"pre": seg_pre, "move": seg_move, "onset": seg_onset, "pour": seg_pour}}
