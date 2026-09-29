#!/usr/bin/env python3
"""시선 픽셀 (u,v) UDP — gaze_on_scene.py -> arm/run_demo.py --tag-direct.

gaze_udp_sender.py(포트 55055)는 SLAM 세계좌표 3D점을 보낸다. 태그 직결에서는
3D점이 필요 없다: 시선은 "어느 컵이냐"만 고르고, 컵의 3D 위치는 팔 쪽이 태그 +
테이블 평면으로 따로 구한다. 그래서 픽셀만 보낸다. 포트를 분리한 이유는 gaze_bridge
(55055 수신)가 모르는 형식을 받지 않게 하려는 것.

좌표: rectified 좌영상의 **카메라 좌표**(--scene-flip 을 되돌린 값). 팔 쪽은
/camera/left 를 뒤집지 않고 받으므로 같은 좌표계다.

무효일 때도 보낸다(valid=False). 안 보내면 받는 쪽이 "아직 안 옴"과
"지금 못 믿음"을 구분할 수 없다.
"""

import json
import socket
import threading
import time

import numpy as np

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 55056


class GazePixelSender:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0

    def send(self, u, v, valid=True):
        payload = {"type": "gaze_px", "t": time.time(), "valid": bool(valid)}
        if valid:
            payload["uv"] = [float(u), float(v)]
        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
            self.sent += 1
        except OSError:
            pass

    def close(self):
        self.sock.close()


class GazePixelReceiver:
    """백그라운드 스레드로 받아 최신 한 개만 들고 있는다."""

    def __init__(self, host="0.0.0.0", port=DEFAULT_PORT, max_age_s=0.15):
        self.max_age_s = max_age_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._latest = None          # (uv or None, 수신시각)
        self.received = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except ValueError:
                continue
            if msg.get("type") != "gaze_px":
                continue
            uv = tuple(msg["uv"]) if msg.get("valid") and "uv" in msg else None
            with self._lock:
                self._latest = (uv, time.time())
                self.received += 1

    def latest(self):
        """-> (u, v) 또는 None. 무효 패킷이거나 max_age_s 보다 오래됐으면 None.

        수신 시각으로 판단한다(송신 측 시계와 맞출 필요 없음). 같은 머신/LAN 이라
        전송 지연은 무시할 수준이다.
        """
        with self._lock:
            got = self._latest
        if got is None:
            return None
        uv, t = got
        if time.time() - t > self.max_age_s:
            return None
        return uv

    def close(self):
        self._stop = True
        self.sock.close()


class GazeRaySender:
    """gaze_on_scene.py -> arm/gaze_tag_bridge.py, docs/12 (R,p_eye) 실전 연결용 (기본 포트 55059).

    GazePixelSender(픽셀)와 달리 씬카메라 좌표계의 3D 광선(origin=p_eye, direction)을
    그대로 보낸다. gaze_tag_bridge 쪽에서 픽셀로 역투영하면(K 사용) 단일 시점 가정이
    되살아나 버리므로, (R,p_eye) 캘리브의 이득을 보려면 광선 자체를 넘겨야 한다."""

    DEFAULT_PORT = 55059

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0

    def send(self, origin, direction, valid=True):
        payload = {"type": "gaze_ray", "t": time.time(), "valid": bool(valid)}
        if valid:
            payload["origin"] = [float(x) for x in origin]
            payload["direction"] = [float(x) for x in direction]
        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
            self.sent += 1
        except OSError:
            pass

    def close(self):
        self.sock.close()


class GazeRayReceiver:
    """arm/gaze_tag_bridge.py 쪽 수신단. GazePixelReceiver와 같은 구조."""

    DEFAULT_PORT = 55059

    def __init__(self, host="0.0.0.0", port=DEFAULT_PORT, max_age_s=0.15):
        self.max_age_s = max_age_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._latest = None          # ((origin,direction) or None, 수신시각)
        self.received = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except ValueError:
                continue
            if msg.get("type") != "gaze_ray":
                continue
            ray = None
            if msg.get("valid") and "origin" in msg and "direction" in msg:
                ray = (tuple(msg["origin"]), tuple(msg["direction"]))
            with self._lock:
                self._latest = (ray, time.time())
                self.received += 1

    def latest(self):
        """-> (origin_xyz, direction_xyz) 또는 None (무효/오래됨)."""
        with self._lock:
            got = self._latest
        if got is None:
            return None
        ray, t = got
        if time.time() - t > self.max_age_s:
            return None
        return ray

    def close(self):
        self._stop = True
        self.sock.close()


class ArmPointReceiver:
    """arm/eye_scene_calib_helper.py -> gaze_on_scene.py (기본 포트 55058).

    docs/12용: 스테레오 깊이(검증 실패, docs/12 참고) 대신 태그+로봇팔 FK로 구한
    3D점(씬카메라 기준, m)을 받는다. GazePixelReceiver와 같은 구조."""

    DEFAULT_PORT = 55058

    def __init__(self, host="0.0.0.0", port=DEFAULT_PORT, max_age_s=0.3):
        self.max_age_s = max_age_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._latest = None          # (point or None, 수신시각)
        self._last_valid = []        # [(point, 수신시각)] — 태그가 깜빡여도 최근 유효값을 쓰려고
        self.received = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except ValueError:
                continue
            point = tuple(msg["point"]) if msg.get("valid") and "point" in msg else None
            with self._lock:
                self._latest = (point, time.time())
                if point is not None:
                    self._last_valid.append((point, time.time()))
                    if len(self._last_valid) > 200:
                        del self._last_valid[:100]
                self.received += 1

    def latest(self):
        """-> (x,y,z) 또는 None (무효거나 max_age_s 보다 오래됨)."""
        with self._lock:
            got = self._latest
        if got is None:
            return None
        point, t = got
        if time.time() - t > self.max_age_s:
            return None
        return point

    def latest_valid(self, window_s=0.7):
        """-> 최근 window_s 안의 유효 점들의 중앙값 (x,y,z) 또는 None.

        latest() 는 가장 최근 패킷이 무효면 None — 태그 유효 프레임이 50~70% 일 때
        (2026-09-27 실측) 키를 누른 순간 운 나쁘면 계속 실패한다. 그리퍼는 멈춰 있으니
        짧은 창의 중앙값이면 충분하고, 튀는 PnP 해도 걸러진다."""
        now = time.time()
        with self._lock:
            pts = [p for p, t in self._last_valid if now - t <= window_s]
        if not pts:
            return None
        return tuple(float(v) for v in np.median(np.asarray(pts, dtype=float), axis=0))

    def close(self):
        self._stop = True
        self.sock.close()
