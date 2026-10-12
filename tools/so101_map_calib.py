#!/usr/bin/env python3
"""SO-101 팔로워: 서보 틱 <-> gaze_hri 기구학 각도 대응을 **실측**한다.

gaze_hri 기구학 규약 (kinematics.forward_kinematics):
  j1 pan        : 0 = 정면(+x), + = 팔의 왼쪽(+y)으로 회전
  j2 shoulder   : 위팔의 수평 기준 절대각. 0 = 위팔이 앞으로 수평, + = 위로
  j3 elbow      : 위팔 대비 상대각. 0 = 아래팔이 위팔과 일직선
  j4 wrist      : 아래팔 대비 상대각. 0 = 손목이 아래팔과 일직선
토크를 끈 팔을 손으로 아래 자세로 잡고, 자세마다 capture 한다.

  B  쭉 편 자세 : 팔 전체를 정면으로 **수평으로 쭉 편다** (위팔·아래팔·손목 일직선)
                  -> j = [0, 0, 0, 0]. 그리퍼 손가락 방향(롤)도 이때 기준이 된다
  A  ㄱ 자세    : 위팔 **수직 위**, 아래팔 **앞으로 수평**, 손목은 아래팔과 일직선
                  -> j2 = +90°, j3 = -90°, j4 = 0
  C  손목 꺾기  : B 에서 손목만 **아래로 90°** (그리퍼가 바닥을 향함)  -> j4 = -90°
  D  좌회전     : **팔을 마주 본 자리(사용자 자리)에서 본인의 오른쪽**으로 약 45° 돌린다.
                  (= 팔 기준 왼쪽, +y). 2026-09-23 첫 측정에서 반대로 잡아 pan 부호가 뒤집혔다 —
                  "팔의 왼쪽" 같은 표현은 보는 위치에 따라 헷갈리니 이 한 가지 기준만 쓸 것.
                  맵을 만든 뒤 반드시 tools/so101_jog.py 1 --deg 10 으로 방향을 눈으로 확인.
  GC / GO       : 그리퍼 완전히 닫음 / 완전히 엶

손으로 잡은 자세라 몇 도 오차는 생긴다. 탑다운 호모그래피 캘리브레이션이 "로봇이
IK 로 믿는 위치"를 기준으로 화면을 맞추므로, 일정한 0점 오차는 상당 부분 상쇄된다.

    python3 tools/so101_map_calib.py --arm a relax --yes   # 토크 OFF (팔을 받친 상태에서)
    python3 tools/so101_map_calib.py --arm a capture B     # 자세마다
    python3 tools/so101_map_calib.py --arm a solve         # -> config/so101_follower_a.yaml

2026-10-12: 팔로워 2대(a/b, gaze_hri/arms.py) — 캡처/결과 파일이 팔마다 따로다.
"""
import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ros2_ws", "src", "gaze_hri"))
from gaze_hri import arms  # noqa: E402
from gaze_hri.so101_driver import JOINT_NAMES, So101Bus  # noqa: E402

Q90 = 1024          # 90° = 1024틱
URDF_UPPER_DEG = 13.97   # 위팔 축선 vs 몸통 (atan2(0.028, 0.11257))
URDF_LOWER_DEG = 2.21    # 아래팔 축선 vs 몸통 (atan2(0.0052, 0.1349))


def load_caps(arm):
    path = arms.captures_file(arm)
    return json.load(open(path)) if os.path.exists(path) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["relax", "capture", "solve", "show"])
    ap.add_argument("pose", nargs="?")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--arm", required=True, choices=sorted(arms.ARMS),
                    help="a = …5667 (기존 팔), b = …5906")
    a = ap.parse_args()

    if a.cmd == "solve":
        return solve(load_caps(a.arm), a.arm)
    if a.cmd == "show":
        print(json.dumps(load_caps(a.arm), indent=1))
        return 0

    bus = So101Bus(arms.port(a.arm))
    try:
        if a.cmd == "relax":
            if not a.yes:
                sys.exit("토크를 끄면 팔이 떨어진다. 팔을 받치고 --yes 로 다시 실행.")
            bus.relax()
            print("토크 OFF — 이제 손으로 자세를 잡을 수 있다")
        elif a.cmd == "capture":
            if a.pose not in ("A", "B", "C", "D", "GC", "GO"):
                sys.exit("pose 는 A B C D GC GO 중 하나")
            t = bus.ticks()
            caps = load_caps(a.arm)
            caps[a.pose] = t
            path = arms.captures_file(a.arm)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(caps, open(path, "w"))
            print(f"{a.pose}: " + ", ".join(f"{n}={v}" for n, v in zip(JOINT_NAMES, t)))
    finally:
        bus.close()
    return 0


def solve(c, arm):
    need = ["A", "B", "C", "D", "GC", "GO"]
    miss = [p for p in need if p not in c]
    if miss:
        sys.exit(f"아직 안 잡은 자세: {miss}")
    A, B, C, D, GC, GO = (c[p] for p in need)
    sgn = lambda v: 1 if v > 0 else -1                          # noqa: E731
    problems = []

    def expect(name, delta, want):
        """delta 틱이 want 틱(부호 무시) 근처인지. 손 자세 오차 ±35% 허용."""
        if not (0.65 * abs(want) < abs(delta) < 1.35 * abs(want)):
            problems.append(f"{name}: {delta}틱 (기대 약 ±{abs(want)}틱 = "
                            f"{abs(want) * 360 / 4096:.0f}°) — 자세를 다시 잡을 것")

    s2 = sgn(A[1] - B[1]);  expect("A-B 어깨(+90°)", A[1] - B[1], Q90)
    s3 = -sgn(A[2] - B[2]); expect("A-B 팔꿈치(-90°)", A[2] - B[2], Q90)
    s4 = -sgn(C[3] - B[3]); expect("C-B 손목(-90°)", C[3] - B[3], Q90)
    s1 = sgn(D[0] - B[0]);  expect("D-B 좌회전(~45°)", D[0] - B[0], Q90 // 2)
    if abs(A[3] - B[3]) > 250:
        problems.append(f"A 와 B 의 손목이 {abs(A[3] - B[3])}틱 다르다 — 둘 다 일직선이어야 함")
    s6 = sgn(GO[5] - GC[5])

    # 0점: B 하나로 정하지 않고 A 에서 역산한 값과 평균낸다. 손으로 잡은 자세라 둘 다
    # 몇 도씩 틀리는데, 2026-09-23 실측에서 A-B 가 어깨/팔꿈치 모두 81°(기대 90°)로
    # 같은 방향으로 어긋났다 = B 에서 위팔이 약 9° 들려 있었다. 평균이면 오차가 반으로 준다.
    #   A: j2=+90° -> tA = z2 + s2*1024,  j3=-90° -> tA = z3 - s3*1024,  j4=0 -> tA = z4
    est = {
        "pan":      (B[0], A[0]),
        "shoulder": (B[1], A[1] - s2 * Q90),
        "elbow":    (B[2], A[2] + s3 * Q90),
        "wrist":    (B[3], A[3]),
    }
    zero4 = [round((b + a2) / 2) for b, a2 in est.values()]

    # 2026-10-12: 손 자세는 **몸통**을 수평/수직으로 맞추지만, 기구학 각도는 **축과 축을 잇는 선**
    # 기준이다. 공식 URDF(so101_new_calib.urdf)에서 위팔은 축선이 몸통보다 14.0°
    # (elbow_flex origin -0.11257, -0.028), 아래팔은 2.2° (wrist_flex origin -0.1349, 0.0052)
    # 기울어 있다. 팔 a 실측(home 자세에서 축 중심 높이/거리 자로 측정)도 어깨 -18°, 팔꿈치 +15°,
    # 아래팔 -3° 로 같은 방향·비슷한 크기였다. 이 차이를 무시해서 그리퍼가 몇 cm 씩 어긋났다.
    #   a1(축선) = U(몸통) - 14.0°,  a2 = F - 2.2°  ->  j2 -= 14.0°, j3 += 11.8°, j4 += 2.2°
    # 각도에 d 를 더하려면 zero 를 sign*d 만큼 뺀다 (rad = sign*(tick - zero)).
    for i, d in ((1, -URDF_UPPER_DEG), (2, URDF_UPPER_DEG - URDF_LOWER_DEG), (3, URDF_LOWER_DEG)):
        zero4[i] = round(zero4[i] - (s2, s3, s4)[i - 1] * d * Q90 / 90.0)
    spread = {k: abs(b - a2) * 360 / 4096 / 2 for k, (b, a2) in est.items()}
    for k, v in spread.items():
        print(f"  0점 {k:8s}: B 추정 {est[k][0]}, A 추정 {est[k][1]} -> 평균 (불확실성 ±{v:.1f}°)")
    zero = zero4 + [B[4], GC[5]]
    sign = [s1, s2, s3, s4, 1, s6]
    grip_open_rad = abs(GO[5] - GC[5]) * 2 * 3.14159265 / 4096

    for p in problems:
        print("[경고]", p)
    cfg = {
        "port": arms.port(arm),
        "servo_ids": [1, 2, 3, 4, 5, 6],
        "zero_ticks": [int(z) for z in zero],
        "signs": sign,
        # 그리퍼: 0 rad = 완전히 닫힘, + = 열림. arm_server 의 gripper_open/closed 에 쓴다
        "gripper_open_rad_max": round(grip_open_rad, 3),
        "zero_uncertainty_deg": {k: round(v, 1) for k, v in spread.items()},
        "captured": {k: c[k] for k in need},
        "warnings": problems,
    }
    OUT = arms.map_file(arm)
    if os.path.exists(OUT):                      # 이전 캘리브 보존
        import time
        os.rename(OUT, OUT.replace(".yaml", time.strftime("_bak_%Y%m%d_%H%M%S.yaml")))
    with open(OUT, "w") as f:
        f.write("# tools/so101_map_calib.py solve 결과. 손으로 다시 잡으면 덮어써진다.\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"저장: {os.path.abspath(OUT)}")
    print(f"  zero={zero}\n  sign={sign}\n  그리퍼 최대 열림 {grip_open_rad:.2f} rad")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
