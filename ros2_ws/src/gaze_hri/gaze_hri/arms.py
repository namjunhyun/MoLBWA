"""팔로워 팔 2대 구분 (2026-10-12).

책상에 12V 팔로워가 2대 있다. 관절 대응(zero_ticks/signs), 탑다운 호모그래피, 휴식 자세는
팔마다 다르므로 파일을 팔 이름(a/b)으로 나눈다. 포트는 /dev/ttyACM* 이 아니라 by-id 로
고정한다 — ACM 번호는 꽂는 순서에 따라 바뀐다.

    a : …5B8E115667  (2026-09 까지 픽앤플레이스에 쓴 팔)
    b : …5B8E115906
"""

import os

ARMS = {
    "a": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115667-if00",
    "b": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115906-if00",
}

# realpath: colcon --symlink-install 의 build/ 경로가 아니라 src/ 의 config 를 가리키게
_CONFIG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "config")


def _check(arm):
    if arm not in ARMS:
        raise ValueError(f"arm 은 {sorted(ARMS)} 중 하나 (받은 값 {arm!r})")
    return arm


def port(arm):
    return ARMS[_check(arm)]


def map_file(arm):
    """tools/so101_map_calib.py 결과 (틱 <-> 기구학 각도)."""
    return os.path.abspath(os.path.join(_CONFIG, f"so101_follower_{_check(arm)}.yaml"))


def homography_file(arm):
    """tools/topdown_calib.py 결과 (탑다운 화면 -> 그 팔의 base_link)."""
    return os.path.expanduser(f"~/.ros/topdown_homography_{_check(arm)}.yaml")


def captures_file(arm):
    return os.path.expanduser(f"~/.ros/so101_map_captures_{_check(arm)}.json")


def rest_file(arm):
    return os.path.expanduser(f"~/.ros/so101_rest_ticks_{_check(arm)}.json")


# ---------------------------------------------------------------------------
# 리더 팔 (2026-10-14) — 텔레옵/따르기 보정용. 아직 연결 전이라 포트는 파일로 지정한다:
#   python3 tools/so101_leader.py ports              # 새로 보이는 포트 확인
#   python3 tools/so101_leader.py assign --arm b --port /dev/serial/by-id/...
# -> ~/.ros/so101_leaders.yaml  {a: <by-id>, b: <by-id>}
# ---------------------------------------------------------------------------
LEADERS_FILE = os.path.expanduser("~/.ros/so101_leaders.yaml")


def leader_port(arm):
    """팔 arm 을 조종할 리더의 포트. 지정 안 됐으면 None."""
    _check(arm)
    if not os.path.exists(LEADERS_FILE):
        return None
    import yaml
    with open(LEADERS_FILE) as f:
        return (yaml.safe_load(f) or {}).get(arm)


def leader_map_file(arm):
    """tools/so101_leader.py solve 결과 (리더 틱 <-> 기구학 각도)."""
    return os.path.abspath(os.path.join(_CONFIG, f"so101_leader_{_check(arm)}.yaml"))


def leader_captures_file(arm):
    return os.path.expanduser(f"~/.ros/so101_leader_captures_{_check(arm)}.json")

