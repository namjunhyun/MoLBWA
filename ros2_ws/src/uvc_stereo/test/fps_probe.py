"""카메라 실제 fps: 한 대씩, 그다음 두 대 동시에 (MJPG 그대로, 디코드 없음).
    python3 fps_probe.py <장치1> [<장치2>] [노출(0.1ms 단위), 0=자동]
"""
import subprocess
import sys
import threading
import time

import cv2


def opencap(d):
    c = cv2.VideoCapture(d, cv2.CAP_V4L2)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(3, 640); c.set(4, 480); c.set(5, 60); c.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    return c


def run(devs, sec=4.0):
    caps = [opencap(d) for d in devs]
    n = [0] * len(caps)
    ts = [[] for _ in caps]

    def rd(i):
        t_end = time.monotonic() + sec
        while time.monotonic() < t_end:
            ok, _ = caps[i].read()
            if ok:
                n[i] += 1
                ts[i].append(caps[i].get(cv2.CAP_PROP_POS_MSEC))
    th = [threading.Thread(target=rd, args=(i,)) for i in range(len(caps))]
    [t.start() for t in th]; [t.join() for t in th]
    [c.release() for c in caps]
    out = []
    for i in range(len(caps)):
        d = sorted(b - a for a, b in zip(ts[i], ts[i][1:]) if b > a)
        out.append(f"{n[i] / sec:.1f} fps (버퍼 시각 간격 중앙 {d[len(d) // 2] if d else 0:.1f} ms)")
    return out


devs = [a for a in sys.argv[1:] if a.startswith("/dev")]
exp = next((int(a) for a in sys.argv[1:] if not a.startswith("/dev")), 0)
for d in devs:
    subprocess.run(["v4l2-ctl", "-d", d, "-c", "auto_exposure=3" if exp == 0 else f"auto_exposure=1,exposure_time_absolute={exp}"])
for d in devs:
    print(f"한 대 {d[-30:]}: {run([d])[0]}")
if len(devs) == 2:
    print("두 대 동시:", " | ".join(run(devs)))
