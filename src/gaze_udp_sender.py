#!/usr/bin/env python3
"""
융합 파이프라인 -> 로봇팔(ROS2) UDP 브리지 송신 측.

`ros2_ws/src/gaze_hri` 의 `gaze_bridge` 노드(`source:=udp`)가 받는 JSON을 쏜다.

왜 UDP인가
----------
ORB-SLAM3 + 동공검출 + OpenCV 파이프라인은 자기만의 파이썬 환경/의존성을 갖는다.
여기에 rclpy 를 같은 프로세스로 묶으면 의존성 충돌이 잦다. 프로세스를 분리하고
UDP 한 줄로 던지면 양쪽 다 편해진다. 손실이 나도 30Hz 로 계속 보내므로
다음 샘플이 금방 온다(신뢰성보다 최신성이 중요한 데이터다).

쓰는 법 — 기존 융합 루프에 세 줄만 추가한다
-------------------------------------------
    from gaze_udp_sender import GazeUdpSender
    sender = GazeUdpSender()                      # 루프 전에 한 번

    # ... 루프 안, gaze_point_world() 호출 직후 ...
    p_W, origin, ray_dir = gaze_point_world(u, v, D, K, T_WS)
    sender.send(p_W, valid=pupil_detected and depth_ok, T_WS=T_WS)

`valid` 를 성실하게 채우는 게 중요하다. 깜빡일 때 좌표가 튀면 dwell 판정이
계속 깨진다. 동공 미검출/깊이 실패/SLAM 트래킹 상실은 전부 False 로 보낸다.

받는 쪽 확인:
    ros2 run gaze_hri gaze_bridge --ros-args -p source:=udp
    ros2 topic echo /gaze/point_raw

이 파일을 직접 실행하면 합성 데이터를 쏜다(수신 확인용):
    python src/gaze_udp_sender.py
"""

import json
import socket

import numpy as np

from fusion import rotation_matrix_to_quat

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 55055


class GazeUdpSender:
    """시선 3D점 + 헤드 pose 를 UDP JSON 한 줄로 보낸다."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0
        self.dropped = 0

    def send(self, p_W, valid=True, T_WS=None):
        """한 샘플을 보낸다.

        p_W   : (3,) 시선이 향한 세계 좌표 3D점. gaze_point_world() 의 첫 반환값.
        valid : 이 샘플을 믿어도 되는지. 깜빡임/추적실패/깊이실패면 False.
        T_WS  : (4,4) SLAM pose. 주면 헤드 pose 도 같이 실어 보낸다.

        valid=False 면 point 는 보내되 수신 측이 버린다. 좌표가 NaN/inf 면
        아예 보내지 않는다(수신 측에서 걸러도 되지만 여기서 막는 게 싸다).
        """
        p = np.asarray(p_W, dtype=float).reshape(-1)[:3]
        if p.shape[0] != 3 or not np.all(np.isfinite(p)):
            self.dropped += 1
            return False

        payload = {
            "point": [float(p[0]), float(p[1]), float(p[2])],
            "valid": bool(valid),
        }
        if T_WS is not None:
            T = np.asarray(T_WS, dtype=float)
            if T.shape == (4, 4) and np.all(np.isfinite(T)):
                payload["head"] = {
                    "position": [float(v) for v in T[:3, 3]],
                    "orientation": rotation_matrix_to_quat(T[:3, :3]),
                }

        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
        except OSError:
            self.dropped += 1
            return False
        self.sent += 1
        return True

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def _selftest():
    """합성 데이터를 30Hz 로 쏜다. 수신 측 배선 확인용."""
    import math
    import time

    sender = GazeUdpSender()
    print(f"UDP {sender.addr} 로 합성 시선 데이터 송신 중. Ctrl+C 로 중단.")
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            # 두 지점을 4초 주기로 번갈아 응시하는 시나리오
            cup = (0.30, -0.12, 0.05) if int(t / 4) % 2 == 0 else (0.42, 0.0, 0.0)
            jitter = 0.004
            p = [c + jitter * math.sin(t * 7 + i) for i, c in enumerate(cup)]
            T = np.eye(4)
            T[:3, 3] = [0.0, 0.0, 0.45]
            sender.send(p, valid=True, T_WS=T)
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print(f"\n보냄 {sender.sent}, 버림 {sender.dropped}")
        sender.close()


if __name__ == "__main__":
    _selftest()
