"""데모 2(G1 교감) 프로세스 사이 UDP JSON 규약. 브리지·ROS 노드·Jetson 서버가 같이 쓴다.

    g1_gaze_bridge ─55057─> g1_interaction ─55070─> g1_motion_server ─55071─> g1_interaction

bridge : {"t", "label": g1_face|g1_hand|g1_torso|None, "bearing_deg": float|None, "valid"}
cmd    : {"seq", "cmd": "act"|"ping", "part", "turn_bin"}
state  : {"state": "idle"|"turning"|"acting", "seq"}

turn_bin 은 G1 torso 기준 사용자 방위(+ = 왼쪽, 도). Jetson 에도 올라가므로 Python 3.8 호환.
"""
from __future__ import annotations

import json

PORT_BRIDGE = 55057
PORT_CMD = 55070
PORT_STATE = 55071

PART_TO_SEGMENT = {"g1_face": "bow", "g1_hand": "handshake", "g1_torso": "open_arms"}
TURN_BINS = (0, 45, 90, 135, 180, -45, -90, -135)


def turn_segment(turn_bin: int) -> str | None:
    """회전 bin -> 라이브러리 세그먼트 이름. 0 은 회전 없음, 180 은 왼쪽 클립 하나뿐."""
    if turn_bin == 0:
        return None
    if turn_bin not in TURN_BINS:
        raise ValueError(f"turn_bin {turn_bin} 은 {TURN_BINS} 밖")
    return f"turn_l{turn_bin}" if turn_bin > 0 else f"turn_r{-turn_bin}"


def nearest_bin(deg: float) -> int:
    """원형 거리로 가장 가까운 bin. 뒤쪽(±180 근방)은 부호와 무관하게 180."""
    def dist(b):
        return abs((deg - b + 180.0) % 360.0 - 180.0)
    return min(TURN_BINS, key=dist)


def encode(msg: dict) -> bytes:
    return json.dumps(msg).encode("utf-8")


def decode(data: bytes) -> dict | None:
    """깨진 패킷이나 dict 가 아닌 JSON 은 None (와이파이 너머라 믿지 않는다)."""
    try:
        m = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return m if isinstance(m, dict) else None
