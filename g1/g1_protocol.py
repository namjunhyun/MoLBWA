"""데모 2(G1 교감) 프로세스 사이 UDP JSON 규약. 브리지·ROS 노드·Jetson 서버가 같이 쓴다.

    g1_gaze_bridge ─55057─> g1_interaction ─55070─> g1_motion_server ─55071─> g1_interaction

bridge : {"t", "label": g1_face|g1_hand|g1_torso|None, "bearing_deg": float|None, "valid"}
cmd    : {"seq", "cmd": "act"|"ping", "part", "bearing_deg": int|None}
state  : {"state": "idle"|"turning"|"acting", "seq"}

bearing_deg 는 G1 torso 기준 사용자 방위(+ = 왼쪽, 도). 허리 yaw 변형도 + = 왼쪽.
Jetson 에도 올라가므로 Python 3.8 호환.
"""
from __future__ import annotations

import json

PORT_BRIDGE = 55057
PORT_CMD = 55070
PORT_STATE = 55071

PART_TO_SEGMENT = {"g1_face": "wave", "g1_hand": "handshake", "g1_torso": "open_arms"}
WAIST_LIMIT_DEG = 75        # |bearing| 이하면 허리 yaw 변형만, 넘으면 turn_l180 + 잔차를 허리로


def _wrap(deg: float) -> float:
    """(-180, 180] 로 감는다."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def _waist(deg: float) -> int:
    return max(-60, min(60, 30 * int(round(deg / 30.0))))


def plan(part: str, bearing_deg: float | None) -> list:
    """부위 + 사용자 방위 -> 재생할 세그먼트 목록. 예: ("g1_face", 150) -> ["turn_l180", "wave_y-30"].
    |b| <= 75 는 허리 yaw 변형 하나(잔차 최대 ±15°), 그 밖은 turn_l180 뒤 잔차를 허리로(±60° 로 잘림)."""
    g = PART_TO_SEGMENT.get(part)
    if g is None:
        raise ValueError(f"모르는 부위 {part!r}")
    if bearing_deg is None:
        return [f"{g}_y0"]
    b = _wrap(bearing_deg)
    if abs(b) <= WAIST_LIMIT_DEG:
        return [f"{g}_y{_waist(b)}"]
    return ["turn_l180", f"{g}_y{_waist(_wrap(b - 180.0))}"]


def encode(msg: dict) -> bytes:
    return json.dumps(msg).encode("utf-8")


def decode(data: bytes) -> dict | None:
    """깨진 패킷이나 dict 가 아닌 JSON 은 None (와이파이 너머라 믿지 않는다)."""
    try:
        m = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return m if isinstance(m, dict) else None
