#!/usr/bin/env python3
"""SO-101 서보 상태 점검 — **읽기 전용**. 어떤 레지스터에도 쓰지 않는다 (팔이 안 움직인다).

    python3 tools/so101_probe.py                  # /dev/ttyACM* 전부
    python3 tools/so101_probe.py /dev/ttyACM1
    python3 tools/so101_probe.py --watch          # 위치를 계속 찍는다 (리더/팔로워 구분용:
                                                  #  손으로 움직이는 쪽 포트의 값이 변한다)

STS3215 레지스터 주소는 lerobot 의 feetech 테이블과 같다.
"""
import argparse
import glob
import sys
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

BAUD = 1_000_000
IDS = [1, 2, 3, 4, 5, 6]
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# (주소, 바이트수)
REG = {
    "Min_Position_Limit": (9, 2), "Max_Position_Limit": (11, 2),
    "Max_Torque_Limit": (16, 2), "Homing_Offset": (31, 2), "Operating_Mode": (33, 1),
    "Torque_Enable": (40, 1), "Acceleration": (41, 1), "Goal_Position": (42, 2),
    "Torque_Limit": (48, 2), "Present_Position": (56, 2), "Present_Load": (60, 2),
    "Present_Voltage": (62, 1), "Present_Temperature": (63, 1), "Moving": (66, 1),
}


def sign_mag(v, bit):
    """Feetech 부호 표현(최상위 비트가 부호)."""
    return -(v & ((1 << bit) - 1)) if v & (1 << bit) else v


def read(ph, port, sid, name):
    addr, n = REG[name]
    fn = {1: ph.read1ByteTxRx, 2: ph.read2ByteTxRx}[n]
    val, res, err = fn(port, sid, addr)
    return val if res == COMM_SUCCESS else None


def open_port(dev):
    port = PortHandler(dev)
    if not port.openPort() or not port.setBaudRate(BAUD):
        raise RuntimeError(f"{dev} 열기 실패")
    return port, PacketHandler(0)          # STS 계열은 protocol_end=0


def probe(dev):
    port, ph = open_port(dev)
    print(f"\n=== {dev}")
    found = []
    for sid, nm in zip(IDS, NAMES):
        model, res, _ = ph.ping(port, sid)
        if res != COMM_SUCCESS:
            print(f"  ID{sid} {nm:14s} 응답 없음")
            continue
        found.append(sid)
        r = {k: read(ph, port, sid, k) for k in REG}
        ho = r["Homing_Offset"]
        print(f"  ID{sid} {nm:14s} model={model} pos={r['Present_Position']:4d} "
              f"범위=[{r['Min_Position_Limit']},{r['Max_Position_Limit']}] "
              f"homing={sign_mag(ho, 11) if ho is not None else None} "
              f"torque={r['Torque_Enable']} mode={r['Operating_Mode']} "
              f"accel={r['Acceleration']} tlim={r['Torque_Limit']}/{r['Max_Torque_Limit']} "
              f"{(r['Present_Voltage'] or 0) / 10:.1f}V {r['Present_Temperature']}°C")
    port.closePort()
    return found


def watch(devs):
    ports = [(d, *open_port(d)) for d in devs]
    print("위치 감시 (Ctrl+C 종료). 손으로 움직이는 팔의 포트 값이 변한다.")
    try:
        while True:
            line = []
            for d, port, ph in ports:
                pos = [read(ph, port, s, "Present_Position") for s in IDS]
                line.append(f"{d.split('/')[-1]}: " + " ".join(f"{p if p is not None else '----':>4}"
                                                            for p in pos))
            print("  |  ".join(line), flush=True)
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        for _, port, _ in ports:
            port.closePort()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ports", nargs="*")
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    devs = a.ports or sorted(glob.glob("/dev/ttyACM*"))
    if not devs:
        sys.exit("시리얼 장치가 없다")
    if a.watch:
        watch(devs)
    else:
        for d in devs:
            probe(d)
