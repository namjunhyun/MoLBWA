#!/usr/bin/env python3
"""
gaze_hri 정합성 검사 — colcon build 없이 돌릴 수 있는 사전 점검.

빌드하고 노드를 띄워야만 드러나는 실수를 미리 잡는다:

  1. yaml 에 있는데 코드에 declare 되지 않은 파라미터
     -> ROS2 는 선언되지 않은 파라미터가 params 파일에 있으면 **노드 기동에 실패**한다.
        오타 하나로 "왜 안 뜨지"를 한참 헤매게 되는 대표적인 함정이다.
  2. 발행자만 있고 구독자가 없는(또는 그 반대) 토픽
  3. launch 가 부르는 executable 이 setup.py entry_points 에 없는 경우
  4. 액션/메시지 필드와 실제 사용의 불일치
  5. .msg / .action 이 실제 rosidl 파서로 파싱되는지 (ROS 환경이 있을 때만)

사용법:

    python scripts/check_gaze_hri.py

종료 코드 0 이면 이상 없음.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PKG = os.path.join(REPO, "ros2_ws", "src", "gaze_hri")
MSGS = os.path.join(REPO, "ros2_ws", "src", "gaze_hri_msgs")
SRC = os.path.join(PKG, "gaze_hri")

# 노드 이름 -> 소스 파일 (yaml 섹션 이름과 맞춰야 한다)
NODE_FILES = {
    "gaze_bridge": "gaze_bridge_node.py",
    "dwell_detector": "dwell_detector_node.py",
    "target_resolver": "target_resolver_node.py",
    "task_manager": "task_manager_node.py",
    "arm_server": "arm_server_node.py",
    "topdown_click": "topdown_click_node.py",
    "control_panel": "control_panel_node.py",
    "calibrate_world_to_base": "calibrate_world_to_base.py",
    "calib_tf_publisher": "calib_tf_publisher.py",
}


def read(*parts):
    with open(os.path.join(*parts)) as f:
        return f.read()


def yaml_sections():
    """들여쓰기 두 단계짜리 단순 구조라 파서 없이 읽는다."""
    out, cur = {}, None
    for line in read(PKG, "config", "gaze_hri.yaml").splitlines():
        if re.match(r"^[a-z_]+:\s*$", line):
            cur = line.strip().rstrip(":")
            out[cur] = []
        elif cur and re.match(r"^\s{4}[a-z_0-9]+:", line):
            out[cur].append(line.strip().split(":")[0])
    return out


def check_parameters(problems):
    print("[1] yaml 파라미터가 코드에 선언돼 있는가")
    sections = yaml_sections()
    for node, keys in sections.items():
        if node not in NODE_FILES:
            problems.append(f"yaml 섹션 '{node}' 에 대응하는 노드를 모르겠다")
            print(f"    {node:26s} ** 대응 노드 없음")
            continue
        declared = set(re.findall(r'declare_parameter\(\s*"([a-z_0-9]+)"',
                                  read(SRC, NODE_FILES[node])))
        missing = [k for k in keys if k not in declared]
        if missing:
            problems.append(f"{node}: 미선언 파라미터 {missing} (노드가 기동에 실패한다)")
        print(f"    {node:26s} yaml {len(keys):2d}개  "
              f"{'OK' if not missing else '** 미선언 ' + str(missing)}")


def check_topics(problems):
    print("\n[2] 토픽 발행/구독 짝")
    pubs, subs = {}, {}
    for f in sorted(os.listdir(SRC)):
        if not f.endswith(".py"):
            continue
        src = read(SRC, f)
        for t in re.findall(r'create_publisher\(\s*\w+,\s*"([^"]+)"', src):
            pubs.setdefault(t, []).append(f)
        for t in re.findall(r'create_subscription\(\s*\w+,\s*"([^"]+)"', src):
            subs.setdefault(t, []).append(f)

    def short(fs):
        return ", ".join(x.replace("_node.py", "").replace(".py", "") for x in fs)

    for t in sorted(set(pubs) | set(subs)):
        note = ""
        if t not in pubs:
            note = "  <- 외부에서 들어오는 입력인지 확인"
        elif t not in subs:
            note = "  <- 시각화 전용인지 확인"
        print(f"    {t:24s} 발행[{short(pubs.get(t, [])) or '-'}]  "
              f"구독[{short(subs.get(t, [])) or '-'}]{note}")


def check_launch(problems):
    print("\n[3] launch 의 executable 이 setup.py 에 있는가")
    entries = set(re.findall(r'"([a-z_0-9]+) = gaze_hri', read(PKG, "setup.py")))
    for lf in sorted(os.listdir(os.path.join(PKG, "launch"))):
        execs = re.findall(r'package="gaze_hri",\s*executable="([a-z_0-9]+)"',
                           read(PKG, "launch", lf))
        bad = sorted({e for e in execs if e not in entries})
        if bad:
            problems.append(f"{lf}: setup.py 에 없는 executable {bad}")
        print(f"    {lf:38s} {'OK' if not bad else '** ' + str(bad)}")


def _fields(text):
    out = set()
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line and " " in line:
            out.add(line.split()[-1])
    return out


def check_action(problems):
    print("\n[4] 액션 목표 필드와 실제 사용")
    goal = _fields(read(MSGS, "action", "PickPlace.action").split("---")[0])
    used = set(re.findall(r"goal\.([a-z_]+)\s*=", read(SRC, "task_manager_node.py")))
    got = set(re.findall(r"\bg\.([a-z_]+)\b", read(SRC, "arm_server_node.py")))
    for label, s in (("task_manager 가 채우는", used), ("arm_server 가 읽는", got)):
        bad = sorted(s - goal)
        if bad:
            problems.append(f"PickPlace.Goal 에 없는 필드 사용: {bad}")
        print(f"    {label:22s} {sorted(s)}  {'OK' if not bad else '** ' + str(bad)}")


def check_msg_parses(problems):
    """실제 ROS 파서로 검증. ROS 환경이 없으면 건너뛴다."""
    print("\n[5] .msg / .action 파싱 (실제 rosidl 파서)")
    try:
        from rosidl_adapter.parser import parse_message_string
    except ImportError:
        print("    ROS 환경이 없어 건너뜀 (colcon build 때 검증된다)")
        return
    for name in ("Fixation", "GazeTarget"):
        try:
            spec = parse_message_string("gaze_hri_msgs", name,
                                        read(MSGS, "msg", f"{name}.msg"))
            print(f"    {name + '.msg':20s} 필드 {len(spec.fields)}개  OK")
        except Exception as exc:                        # noqa: BLE001
            problems.append(f"{name}.msg 파싱 실패: {exc}")
            print(f"    {name}.msg  ** {exc}")
    parts = read(MSGS, "action", "PickPlace.action").split("---")
    for label, body in zip(("Goal", "Result", "Feedback"), parts):
        try:
            parse_message_string("gaze_hri_msgs", f"PickPlace_{label}", body)
            print(f"    {'PickPlace.' + label:20s} OK")
        except Exception as exc:                        # noqa: BLE001
            problems.append(f"PickPlace.{label} 파싱 실패: {exc}")
            print(f"    PickPlace.{label}  ** {exc}")


def main():
    print("=" * 74)
    print(" gaze_hri 정합성 검사 (빌드 없이)")
    print("=" * 74)
    problems = []
    check_parameters(problems)
    check_topics(problems)
    check_launch(problems)
    check_action(problems)
    check_msg_parses(problems)

    print("\n" + "=" * 74)
    if problems:
        print(f" 문제 {len(problems)}건")
        for p in problems:
            print(f"   - {p}")
        return 1
    print(" 정합성 문제 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
