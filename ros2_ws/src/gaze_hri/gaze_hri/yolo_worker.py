#!/usr/bin/env python3
"""YOLO 검출 워커. ~/yolo-env 의 파이썬으로 띄운다 (시스템 numpy 1.x 와 torch 충돌을 피하려고 프로세스를 분리).

    yolo_worker.py MODEL CONF CLS1,CLS2,...

프로토콜: stdin 으로 [uint32 길이][JPEG], stdout 으로 JSON 한 줄
{"dets": [[클래스, conf, x1, y1, x2, y2], ...]}. 시작하면 {"ready": true} 를 한 번 낸다.
"""
import json
import struct
import sys

import cv2
import numpy as np


def read_exact(f, n):
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    model_path, conf, classes = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(",")
    out = sys.stdout
    sys.stdout = sys.stderr            # 라이브러리 출력이 프로토콜을 오염시키지 않게

    from ultralytics import YOLO
    model = YOLO(model_path)
    names = model.names
    ids = [i for i, n in names.items() if n in classes]
    model.predict(np.zeros((480, 640, 3), np.uint8), verbose=False)    # 워밍업
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()

    inp = sys.stdin.buffer
    while True:
        hdr = read_exact(inp, 4)
        if hdr is None:
            break
        data = read_exact(inp, struct.unpack("<I", hdr)[0])
        if data is None:
            break
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        r = model.predict(img, conf=conf, classes=ids, imgsz=640, verbose=False)[0]
        dets = [[names[int(b.cls)], float(b.conf)] + [float(x) for x in b.xyxy[0]] for b in r.boxes]
        out.write(json.dumps({"dets": dets}) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
