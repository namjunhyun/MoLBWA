"""데모 2(G1 교감) 프로세스 사이 UDP JSON 규약. 브리지·ROS 노드·Jetson 서버가 같이 쓴다.

    g1_gaze_bridge ─55057─> g1_interaction ─55070─> g1_motion_server ─55071─> g1_interaction

bridge : {"t", "label": g1_face|g1_hand|g1_torso|None, "bearing_deg": float|None, "obj_bearing_deg": float|None, "valid"}
cmd    : {"seq", "cmd": "act"|"ping", "part", "bearing_deg": int|None}
         {"seq", "cmd": "attend", "obj_bearing_deg": int, "user_bearing_deg": int, "gesture": "celebrate"|...}
           = 같이 보기: 물체 쪽으로 돌아 가리키고 → 착용자 쪽으로 돌아와 동작 (plan_attend)
state  : {"state": "idle"|"turning"|"acting", "seq"}

bearing_deg 는 G1 torso 기준 사용자 방위(+ = 왼쪽, 도). 회전 클립 turn_l*/turn_r* 도 l = 왼쪽(+).
Jetson 에도 올라가므로 Python 3.8 호환.
"""
from __future__ import annotations

import json

PORT_BRIDGE = 55057
PORT_CMD = 55070
PORT_STATE = 55071

PART_TO_SEGMENT = {"g1_face": "wave", "g1_hand": "handshake", "g1_torso": "open_arms"}
TURN_STEP = 15   # 2026-10-09: 45 → 15 (사용자 좌표에 따라 여러 각도로). 잔차 최대 ±7.5°
TURN_BINS = tuple(range(0, 181, TURN_STEP)) + tuple(-a for a in range(TURN_STEP, 180, TURN_STEP))   # + = 왼쪽, 180 은 왼쪽 하나


def _wrap(deg: float) -> float:
    """(-180, 180] 로 감는다."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def _quant(bearing_deg: float) -> int:
    """방위를 원형 거리로 가장 가까운 TURN_BINS 로 (동률이면 TURN_BINS 앞쪽 = 작은 |각|, 왼쪽)."""
    b = _wrap(bearing_deg)
    return min(TURN_BINS, key=lambda x: abs(_wrap(b - x)))


def _turn(tb: int) -> list:
    return [] if tb == 0 else [f"turn_l{tb}" if tb > 0 else f"turn_r{-tb}"]


ATTEND_GESTURES = ("celebrate", "wave")


def plan_attend(obj_bearing_deg: float, user_bearing_deg: float, gesture: str = "celebrate") -> list:
    """같이 보기: 물체 방위·착용자 방위(둘 다 지금 torso 기준, + = 왼쪽) -> 세그먼트 목록.
    [물체 쪽 회전?, point, 착용자 쪽 회전?, 동작]. 회전 클립은 상대 회전이라 두 번째 회전은
    첫 회전 뒤 실제로 향한 bin 에서 착용자까지의 각이다. 예: (60, -30) -> [turn_l60, point, turn_r90, celebrate]"""
    if gesture not in ATTEND_GESTURES:
        raise ValueError(f"모르는 동작 {gesture!r}")
    to = _quant(obj_bearing_deg)
    return _turn(to) + ["point"] + _turn(_quant(user_bearing_deg - to)) + [gesture]


def plan_msg(msg: dict) -> list:
    """명령 dict -> 세그먼트 목록 (act·attend). 잘못된 값은 ValueError/TypeError."""
    if msg.get("cmd") == "act":
        return plan(msg.get("part"), msg.get("bearing_deg"))
    if msg.get("cmd") == "attend":
        o, u = msg.get("obj_bearing_deg"), msg.get("user_bearing_deg")
        if type(o) is not int or type(u) is not int:   # float·bool·None 거부
            raise TypeError("obj/user_bearing_deg 는 int")
        return plan_attend(o, u, msg.get("gesture", "celebrate"))
    raise ValueError(f"모르는 명령 {msg.get('cmd')!r}")


def plan(part: str, bearing_deg: float | None) -> list:
    """부위 + 사용자 방위(+ = 왼쪽) -> 재생할 세그먼트 목록. 예: ("g1_face", 152) -> ["turn_l150", "wave"].
    방위를 원형 거리로 가장 가까운 TURN_BINS 로 양자화해 전신 제자리 회전 클립을 먼저 틀고,
    사용자를 마주본 채 동작을 허리 변형 없이 재생한다. 0 bin 이거나 방위가 없으면 동작만."""
    g = PART_TO_SEGMENT.get(part)
    if g is None:
        raise ValueError(f"모르는 부위 {part!r}")
    if bearing_deg is None:
        return [g]
    return _turn(_quant(bearing_deg)) + [g]


def encode(msg: dict) -> bytes:
    return json.dumps(msg).encode("utf-8")


def decode(data: bytes) -> dict | None:
    """깨진 패킷이나 dict 가 아닌 JSON 은 None (와이파이 너머라 믿지 않는다)."""
    try:
        m = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return m if isinstance(m, dict) else None
