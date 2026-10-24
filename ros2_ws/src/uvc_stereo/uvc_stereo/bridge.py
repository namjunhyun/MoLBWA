"""PC 쪽: UNO Q 의 stereo_relay.py(TCP) → 정렬된 스테레오 ROS 토픽.

    ros2 run uvc_stereo uvc_stereo_bridge --ros-args -p host:=<UNO Q IP>

내는 토픽 (기존 oCamS 경로와 같은 계약 — 소비 코드는 그대로 둔다):
    {prefix}/left/compressed, {prefix}/right/compressed   sensor_msgs/CompressedImage (jpeg, mono)
    {prefix}/left, {prefix}/right                          sensor_msgs/Image mono8   (publish_raw:=true 일 때)
  prefix 기본 /pc/camera → gaze_on_scene·gaze_tag_bridge·run_slam 이 쓰는 /pc/camera/... 로 바로 나간다(cam_relay 불필요).
  rectify:=true(기본)면 left/right_opencv.yaml 로 왜곡 보정+정렬한 영상, 모든 소비 코드가 이걸 기대한다.
  **캘리브레이션 녹화 때는** rectify:=false prefix:=/camera publish_raw:=true → Kalibr 스크립트가 쓰는 /camera/left·right 원본.

시각: 보드와 PC 시계가 다르다. (PC 수신 시각 − 보드 송신 시각)의 최근 최솟값을 오프셋으로 써서
보드 촬영 시각을 PC 시각으로 옮긴다(최소값 = 네트워크 지연이 가장 작았던 순간 ≈ 시계 차).
왼쪽·오른쪽은 **같은 스탬프**(왼쪽 촬영 시각)로 낸다 — SLAM·ApproximateTime 이 쌍으로 묶도록.
"""
import collections
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image

HDR = struct.Struct("<4sIdddII")
MAGIC = b"STR1"


def _default_calib_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("uvc_stereo"), "config")
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")


def load_maps(calib_dir, size):
    """left/right_opencv.yaml (oCamS 와 같은 OpenCV FileStorage 형식) → remap 맵. 없으면 None."""
    maps = []
    for side in ("left", "right"):
        p = os.path.join(calib_dir, f"{side}_opencv.yaml")
        fs = cv2.FileStorage(p, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            return None, p
        K, D = fs.getNode("camera_matrix").mat(), fs.getNode("distortion_coefficients").mat()
        Rm, P = fs.getNode("rectification_matrix").mat(), fs.getNode("projection_matrix").mat()
        w, h = int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real())
        fs.release()
        if (w, h) != size:
            raise SystemExit(f"{p} 은 {w}x{h} 용인데 영상은 {size[0]}x{size[1]} — 같은 해상도로 캘리브레이션하세요")
        maps.append(cv2.initUndistortRectifyMap(K, D, Rm, P, size, cv2.CV_16SC2))
    return maps, calib_dir


class Bridge(Node):
    def __init__(self):
        super().__init__("uvc_stereo_bridge")
        g = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.host, self.port = g("host", "192.168.50.120"), g("port", 55100)
        prefix = g("prefix", "/pc/camera").rstrip("/")
        self.rectify, self.calib_dir = g("rectify", True), g("calib_dir", "") or _default_calib_dir()
        self.publish_raw, self.jq = g("publish_raw", False), g("jpeg_quality", 90)
        self.fid = (g("left_frame_id", "left_camera"), g("right_frame_id", "right_camera"))
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.pc = [self.create_publisher(CompressedImage, f"{prefix}/{s}/compressed", qos) for s in ("left", "right")]
        self.pr = [self.create_publisher(Image, f"{prefix}/{s}", 10) for s in ("left", "right")] if self.publish_raw else None
        self.maps, self.size = None, None
        self.offs = collections.deque(maxlen=300)
        self.stat = {"n": 0, "skew": [], "lat": [], "t": time.monotonic()}
        self.get_logger().info(f"{self.host}:{self.port} → {prefix}/{{left,right}}"
                               f"{'/compressed' + (' + raw' if self.publish_raw else '')} · 정렬 {'켬' if self.rectify else '끔(원본)'}")
        threading.Thread(target=self.loop, daemon=True).start()

    # ------------------------------------------------------------- 수신
    def loop(self):
        while rclpy.ok():
            try:
                s = socket.create_connection((self.host, self.port), timeout=5)
                s.settimeout(5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.get_logger().info("UNO Q 연결됨")
                self.offs.clear()
                while rclpy.ok():
                    h = self.recv(s, HDR.size)
                    magic, seq, tl, tr, ts, nl, nr = HDR.unpack(h)
                    if magic != MAGIC:
                        raise IOError("헤더가 깨졌다")
                    jl, jr = self.recv(s, nl), self.recv(s, nr)
                    self.on_pair(tl, tr, ts, jl, jr)
            except (OSError, IOError) as e:
                self.get_logger().warn(f"연결 실패/끊김 ({type(e).__name__}: {e}) — 2초 뒤 다시")
                time.sleep(2)

    @staticmethod
    def recv(s, n):
        b = bytearray()
        while len(b) < n:
            c = s.recv(n - len(b))
            if not c:
                raise IOError("연결 닫힘")
            b += c
        return bytes(b)

    # ------------------------------------------------------------- 처리
    def on_pair(self, tl, tr, ts, jl, jr):
        now = time.time()
        self.offs.append(now - ts)
        off = min(self.offs)                              # 보드 단조 시각 → PC 벽시계
        stamp_s = tl + off
        imgs = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_GRAYSCALE) for j in (jl, jr)]
        if imgs[0] is None or imgs[1] is None:
            return
        size = (imgs[0].shape[1], imgs[0].shape[0])
        if self.rectify and self.size != size:
            self.size = size
            self.maps, where = load_maps(self.calib_dir, size)
            if self.maps is None:
                self.get_logger().error(f"캘리브레이션 없음({where}) — 정렬 안 된 원본을 낸다. 소비 코드는 정렬된 영상을 기대하니 "
                                        "캘리브레이션 전에는 rectify:=false 로 쓰세요")
        if self.rectify and self.maps is not None:
            imgs = [cv2.remap(im, m[0], m[1], cv2.INTER_LINEAR) for im, m in zip(imgs, self.maps)]
        sec = int(stamp_s); nsec = int((stamp_s - sec) * 1e9)
        for i, im in enumerate(imgs):
            m = CompressedImage()
            m.header.stamp.sec, m.header.stamp.nanosec, m.header.frame_id = sec, nsec, self.fid[i]
            m.format = "mono8; jpeg compressed mono8"
            m.data = (jl, jr)[i] if not (self.rectify and self.maps is not None) else \
                cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, self.jq])[1].tobytes()
            self.pc[i].publish(m)
            if self.pr:
                r = Image()
                r.header = m.header
                r.height, r.width, r.encoding, r.step = im.shape[0], im.shape[1], "mono8", im.shape[1]
                r.data = im.tobytes()
                self.pr[i].publish(r)
        st = self.stat
        st["n"] += 1; st["skew"].append(abs(tl - tr) * 1000); st["lat"].append((now - stamp_s) * 1000)
        if time.monotonic() - st["t"] > 5:
            sk, la = np.array(st["skew"]), np.array(st["lat"])
            self.get_logger().info(f"5초: 쌍 {st['n'] / 5:.1f}/s · 좌우 어긋남 중앙 {np.median(sk):.1f} 최대 {sk.max():.1f} ms · "
                                   f"촬영→발행 지연 중앙 {np.median(la):.0f} ms (최소지연 기준이라 실제보다 작게 보임)")
            self.stat = {"n": 0, "skew": [], "lat": [], "t": time.monotonic()}


def main():
    rclpy.init()
    n = Bridge()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
