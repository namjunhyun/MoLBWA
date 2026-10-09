"""데모 2(G1 교감) 프로세스 사이 UDP JSON 규약. 브리지·ROS 노드·Jetson 서버가 같이 쓴다.

    g1_gaze_bridge ─55057─> g1_interaction ─55070─> g1_motion_server ─55071─> g1_interaction

bridge : {"t", "label": g1_face|g1_hand|g1_torso|None, "bearing_deg": float|None, "valid"}
cmd    : {"seq", "cmd": "act"|"ping", "part", "bearing_deg": int|None}
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
TURN_BINS = (0, 45, 90, 135, 180, -45, -90, -135)   # 제자리 회전 클립 각도(+ = 왼쪽). 잔차 최대 ±22.5°


def _wrap(deg: float) -> float:
    """(-180, 180] 로 감는다."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def plan(part: str, bearing_deg: float | None) -> list:
    """부위 + 사용자 방위(+ = 왼쪽) -> 재생할 세그먼트 목록. 예: ("g1_face", 150) -> ["turn_l135", "wave"].
    방위를 원형 거리로 가장 가까운 TURN_BINS 로 양자화해 전신 제자리 회전 클립을 먼저 틀고,
    사용자를 마주본 채 동작을 허리 변형 없이 재생한다. 0 bin 이거나 방위가 없으면 동작만."""
    g = PART_TO_SEGMENT.get(part)
    if g is None:
        raise ValueError(f"모르는 부위 {part!r}")
    if bearing_deg is None:
        return [g]
    b = _wrap(bearing_deg)
    tb = min(TURN_BINS, key=lambda x: abs(_wrap(b - x)))   # 동률이면 TURN_BINS 앞쪽(작은 |각|, 왼쪽)
    if tb == 0:
        return [g]
    return [f"turn_l{tb}" if tb > 0 else f"turn_r{-tb}", g]


def encode(msg: dict) -> bytes:
    return json.dumps(msg).encode("utf-8")


def decode(data: bytes) -> dict | None:
    """깨진 패킷이나 dict 가 아닌 JSON 은 None (와이파이 너머라 믿지 않는다)."""
    try:
        m = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return m if isinstance(m, dict) else None
