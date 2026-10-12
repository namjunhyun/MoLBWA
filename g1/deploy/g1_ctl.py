"""Jetson 쪽 조작 명령 (조작 패널이 SSH 로 부른다). Python 3.8.

    python3 g1_ctl.py status|release|highlevel|damp|stop|start|start_dry

status    : 모드 이름('ai' = 고수준, '' = 디벨롭/R&D) + 정책 서버 실행 여부
release   : 디벨롭 모드 (MotionSwitcher.ReleaseMode) — 이후 로봇은 스스로 못 선다(하네스)
highlevel : 고수준 모드 — SelectMode 를 ai → normal → advanced 로 시도(각 2초 뒤 확인), 성공하면 곧바로 Damp. 정책 서버가 돌면 거부
            (console.py 와 같은 순서. 셋 다 실패하면 로봇 재부팅)
damp      : 비상정지 — 정책 서버가 돌면 SIGTERM(kp0·kd2 댐핑 3초), 아니면 고수준 SetFsmId(1)(R&D 상태에선 안 먹는다)
stop      : 정지(기본자세) — 정책 서버가 돌면 SIGINT(기본자세 유지 후 끝), 아니면 고수준 StopMove()
start     : 정책 서버 발행(--arm). 디벨롭 모드이고 서버가 없을 때만
"""
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
PIDF = "/tmp/g1_deploy.pid"
LOG = os.path.join(HERE, "server.log")
SERVER_ARGS = ["--policy", "policy.npz", "--motion", "library.npz", "--library_meta", "library_meta.json",
               "--state_host", os.environ.get("DESK", "192.168.50.211")]


def server_pid():
    try:
        pid = int(open(PIDF).read())
        if b"g1_motion_server" not in open(f"/proc/{pid}/cmdline", "rb").read():   # 재사용된 PID 에 신호 보내지 않기
            return None
        return pid
    except (OSError, ValueError):
        return None


def dds():
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    out = subprocess.run(["ip", "-4", "-o", "addr"], stdout=subprocess.PIPE, universal_newlines=True).stdout
    ChannelFactoryInitialize(0, next(f.split()[1] for f in out.splitlines() if " 192.168.123." in f))


def switcher():
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    m = MotionSwitcherClient(); m.SetTimeout(5.0); m.Init()
    return m


def loco():
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    c = LocoClient(); c.SetTimeout(5.0); c.Init()
    return c


def mode_name(m):
    r = m.CheckMode()
    return r[1].get("name", "?") if isinstance(r, tuple) and isinstance(r[1], dict) else "?"


def main(cmd):
    pid = server_pid()
    if cmd in ("damp", "stop") and pid:            # 정책 서버가 돌면 DDS 를 열지 않고 신호만 (가장 빠른 경로)
        os.kill(pid, signal.SIGTERM if cmd == "damp" else signal.SIGINT)
        return f"정책 서버 {pid} 에 {'SIGTERM(댐핑)' if cmd == 'damp' else 'SIGINT(기본자세 유지 후 종료)'}"
    dds(); m = switcher(); name = mode_name(m)
    if cmd == "status":
        return f"mode={name or '디벨롭'} server={pid or '-'}"
    if cmd == "release":
        m.ReleaseMode()
        time.sleep(1.0)
        after = mode_name(m)
        return f"디벨롭 모드 {'완료' if after == '' else '실패'}: {name!r} → {after!r}"
    if cmd == "highlevel":
        if pid:
            return "거부: 정책 서버가 돌고 있다. 먼저 정지/비상정지"
        if name not in ("", "?"):
            return f"이미 고수준 ({name!r})"
        for nm in ("ai", "normal", "advanced"):
            m.SelectMode(nm)
            time.sleep(2.0)
            after = mode_name(m)
            if after not in ("", "?"):
                loco().SetFsmId(1)          # 올라오자마자 댐핑 (console.py 와 같음)
                return f"고수준 모드 완료 ({after!r}) → 댐핑 상태. 일으키려면 리모컨/기본 앱"
        return "고수준 복귀 실패 (ai/normal/advanced) — 로봇 재부팅 필요"
    if cmd in ("damp", "stop"):
        if name == "":
            return "정책 서버도 고수준도 없다 — 이미 힘이 빠진 상태"
        c = loco()
        if cmd == "damp":
            c.SetFsmId(1)
            return "고수준 댐핑 SetFsmId(1)"
        c.StopMove()
        return "고수준 StopMove (제자리 정지)"
    if cmd in ("start", "start_dry"):
        if pid:
            return f"이미 실행 중 ({pid})"
        if name != "":
            return f"거부: 모드가 {name!r} — 먼저 디벨롭 모드"
        args = ["python3", os.path.join(HERE, "g1_motion_server.py")] + SERVER_ARGS + (["--arm"] if cmd == "start" else ["--run_sec", "20"])
        subprocess.Popen(args, cwd=HERE, stdout=open(LOG, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        return f"정책 서버 시작({'발행' if cmd == 'start' else 'dry-run 20초'}) — 로그 {LOG}"
    return f"모르는 명령 {cmd}"


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else "status"))
