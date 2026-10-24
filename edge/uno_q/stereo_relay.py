#!/usr/bin/env python3
"""UNO Q(또는 아무 리눅스 보드)에서 OV9281 USB 카메라 두 대 → 짝 맞춘 JPEG 쌍을 TCP 로 PC 에 보낸다.

    python3 stereo_relay.py --list                      # 꽂힌 카메라 목록 + 각 카메라 사진 한 장(/tmp/cam_*.jpg)
    python3 stereo_relay.py --left <장치> --right <장치> # 중계 시작 (기본 포트 55100)

ROS 가 없는 보드용이라 파이썬 + OpenCV(+numpy)만 쓴다 (Debian: sudo apt install python3-opencv v4l-utils).
PC 쪽은 ros2_ws/src/uvc_stereo 의 uvc_stereo_bridge 가 받아 ROS 토픽으로 낸다.

왜 이렇게 짰나
- **MJPG 를 풀지 않고 그대로 보낸다.** 카메라가 이미 JPEG 로 압축해 주므로(CAP_PROP_CONVERT_RGB=0),
  보드는 디코드·인코드를 안 한다 — A53 코어 부담 최소. 카메라가 MJPG 를 못 주면 GREY/YUYV 로 받아 직접 인코드.
- **두 카메라는 하드웨어 동기가 없다**(Waveshare OV9281 USB 에 트리거 핀 없음). 각자 찍은 프레임의 커널 시각
  (V4L2 버퍼 시각)을 읽어 가장 가까운 것끼리 짝을 짓고, 어긋남이 --max_skew_ms 를 넘으면 버린다.
  어긋남 통계(중앙·95%·최대)를 5초마다 찍는다 — 머리를 빨리 돌리면 이 값만큼 깊이가 틀어진다.
- **같은 모델 두 대는 /dev/v4l/by-id 이름이 같을 수 있다** → 장치는 꽂은 USB 포트(/dev/v4l/by-path)로 고른다.
  허브의 같은 포트에 같은 카메라를 꽂으면 왼쪽·오른쪽이 바뀌지 않는다.

전송 형식 (TCP, 한 쌍마다):  헤더 struct '<4sIdddII'  = b'STR1', seq, t_left, t_right, t_send, len_left, len_right
                               + left JPEG + right JPEG     (시각은 보드의 CLOCK_MONOTONIC 초)
"""
import argparse
import collections
import glob
import os
import socket
import struct
import subprocess
import threading
import time

import cv2
import numpy as np

HDR = struct.Struct("<4sIdddII")
MAGIC = b"STR1"


def list_cams():
    """비디오 장치마다 by-path·by-id 이름과 지원 형식, 사진 한 장."""
    paths = {}
    for d in ("/dev/v4l/by-path", "/dev/v4l/by-id"):
        for p in sorted(glob.glob(d + "/*video-index0")):
            paths.setdefault(os.path.realpath(p), []).append(p)
    if not paths:
        print("비디오 장치 없음 — 허브 전원(PD)·케이블 확인. `ls /dev/video*` 도 보세요.")
        return
    for i, (dev, names) in enumerate(sorted(paths.items())):
        print(f"\n[{i}] {dev}")
        for n in names:
            print("    ", n)
        try:
            fm = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats-ext"], capture_output=True, text=True, timeout=5).stdout
            fmts = sorted({ln.split("'")[1] for ln in fm.splitlines() if "'" in ln and "[" in ln})
            sizes = sorted({ln.split()[-1] for ln in fm.splitlines() if "Size:" in ln})
            print("     형식", fmts, "크기", sizes[:8])
        except Exception as e:
            print("     (v4l2-ctl 없음 — sudo apt install v4l-utils)", type(e).__name__)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        ok, f = cap.read() if cap.isOpened() else (False, None)
        for _ in range(5):                   # 노출이 자리 잡도록 몇 장 버린다
            ok, f = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            out = f"/tmp/cam_{i}.jpg"
            cv2.imwrite(out, f)
            print(f"     사진 {out}  밝기 평균 {float(np.mean(f)):.0f}  ← 한쪽 렌즈를 손으로 가리고 다시 돌리면 어느 게 어느 쪽인지 안다")


def v4l2_set(dev, controls):
    """'auto_exposure=1,exposure_time_absolute=100' 처럼 v4l2 컨트롤을 그대로 넣는다(카메라마다 이름이 다름 —
    `v4l2-ctl -d <장치> -l` 로 확인). 두 카메라 노출이 같아야 스테레오 매칭이 잘 된다."""
    if controls:
        r = subprocess.run(["v4l2-ctl", "-d", dev, "-c", controls], capture_output=True, text=True)
        if r.returncode:
            print(f"[경고] {dev} 컨트롤 설정 실패: {r.stderr.strip()}")


class Cam(threading.Thread):
    """한 카메라를 계속 읽어 (보드 단조 시각 초, JPEG 바이트) 를 큐에 쌓는다."""

    def __init__(self, name, dev, a):
        super().__init__(daemon=True)
        self.name, self.dev = name, dev
        self.q = collections.deque(maxlen=8)
        self.n = 0
        self.passthrough = False
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise SystemExit(f"{name} 카메라를 못 엶: {dev}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*a.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        cap.set(cv2.CAP_PROP_FPS, a.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if a.fourcc == "MJPG":
            self.passthrough = bool(cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))
        self.cap, self.jq = cap, a.jpeg_quality
        v4l2_set(dev, a.controls)
        fcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
        print(f"{name}: {dev} → {int(cap.get(3))}x{int(cap.get(4))} @{cap.get(5):.0f}fps {fcc} "
              f"({'MJPG 그대로 전달' if self.passthrough else '보드에서 JPEG 인코드'})", flush=True)
        self.t0_wall_mono = (time.time(), time.monotonic())

    def run(self):
        while True:
            ok, f = self.cap.read()
            t_cam = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0    # V4L2 버퍼 시각(단조 시계)
            t = t_cam if t_cam > 0 else time.monotonic()            # 못 주는 드라이버면 읽은 시각
            if not ok or f is None:
                time.sleep(0.005)
                continue
            if self.passthrough and f.ndim == 2 and f.shape[0] == 1:   # 압축 그대로 (1×N 바이트)
                jpg = f.tobytes()
            else:
                if f.ndim == 3:
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, self.jq])[1].tobytes()
            self.q.append((t, jpg))
            self.n += 1


def pair(L, R, max_skew):
    """두 큐의 머리끼리 비교해 어긋남 ≤ max_skew 면 짝, 아니면 더 오래된 쪽을 버린다."""
    out = []
    while L.q and R.q:
        tl, tr = L.q[0][0], R.q[0][0]
        if abs(tl - tr) <= max_skew:
            out.append((L.q.popleft(), R.q.popleft()))
        elif tl < tr:
            L.q.popleft()
        else:
            R.q.popleft()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="카메라 목록·형식·사진 한 장씩")
    ap.add_argument("--left", help="왼쪽 카메라 장치 (/dev/v4l/by-path/...video-index0 권장)")
    ap.add_argument("--right", help="오른쪽 카메라 장치")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--fourcc", default="MJPG", help="MJPG(권장, USB2 두 대) / GREY / YUYV")
    ap.add_argument("--jpeg_quality", type=int, default=90, help="보드에서 인코드할 때만")
    ap.add_argument("--controls", default="", help="두 카메라에 똑같이 넣을 v4l2 컨트롤 (예: auto_exposure=1,exposure_time_absolute=150)")
    ap.add_argument("--max_skew_ms", type=float, default=None, help="짝으로 인정할 최대 어긋남 (기본: 프레임 간격의 절반)")
    ap.add_argument("--port", type=int, default=55100)
    a = ap.parse_args()
    if a.list:
        list_cams()
        return
    if not (a.left and a.right):
        ap.error("--left 와 --right 를 주세요 (먼저 --list 로 확인)")
    max_skew = (a.max_skew_ms if a.max_skew_ms is not None else 500.0 / a.fps) / 1000.0
    L, R = Cam("왼쪽", a.left, a), Cam("오른쪽", a.right, a)
    L.start(); R.start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", a.port)); srv.listen(1)
    print(f"PC 접속 대기 :{a.port}  (짝 기준 어긋남 ≤ {max_skew * 1000:.1f} ms)", flush=True)
    seq = 0
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"PC 접속: {addr}", flush=True)
        skews, sent, t_rep = [], 0, time.monotonic()
        L.q.clear(); R.q.clear()
        try:
            while True:
                for (tl, jl), (tr, jr) in pair(L, R, max_skew):
                    conn.sendall(HDR.pack(MAGIC, seq, tl, tr, time.monotonic(), len(jl), len(jr)) + jl + jr)
                    seq += 1; sent += 1
                    skews.append(abs(tl - tr) * 1000)
                if time.monotonic() - t_rep > 5.0:
                    s = np.array(skews) if skews else np.zeros(1)
                    print(f"5초: 받은 프레임 왼 {L.n} 오 {R.n} · 보낸 쌍 {sent} ({sent / 5:.1f}/s) · 어긋남 ms "
                          f"중앙 {np.median(s):.1f} 95% {np.percentile(s, 95):.1f} 최대 {s.max():.1f}", flush=True)
                    L.n = R.n = sent = 0; skews = []; t_rep = time.monotonic()
                time.sleep(0.002)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"PC 연결 끊김 ({type(e).__name__}) — 다시 대기", flush=True)
            conn.close()


if __name__ == "__main__":
    main()
