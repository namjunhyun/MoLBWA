#!/bin/bash
# 작업 중이면 끄지 않는다. 강제로 끄려면 FORCE=1
# 2026-10-26: 이미 꺼져 있으면 옛 로그로 busy 판정하지 말 것 (그래서 restart 가 조용히 실패했다)
pgrep -f "install/gaze_hri/lib/gaze_hri/" >/dev/null || { echo stopped; exit 0; }
L=$(cat ~/.ros/molbwa_run/gui_log_path 2>/dev/null)
if [ -z "$FORCE" ] && [ -f "$L" ]; then
  sent=$(grep -c "목표 전달" "$L"); done_=$(grep -E "task_manager\].*(결과\[|사용자 취소|완료:|실패:|어느 팔로도 안 됩니다|로 다시|못 잡습니다|닿는 팔이 없습니다)" "$L" | grep -vc "집기와 놓을 곳 둘 다")
  # 2026-10-22: "집기와 놓을 곳 둘 다 닿는 팔이 없습니다"(선택 단계 경고, 작업 아님)를 끝으로 세서 팔 B 들어 올리는 중에 꺼 버렸다.
  # (팔 서버가 거부한 "놓을 곳(컵)에 닿는 팔이 없습니다 → 다시 고르세요"는 작업 끝이 맞다)
  if [ "$sent" -gt "$done_" ]; then echo "busy — 작업 중이라 끄지 않음"; exit 2; fi
fi
# 2026-10-22: launch 에 INT 를 보내도 노드에 안 전달돼 GUI 두 개가 겹쳐 뜬 일 -> 노드에 직접 INT, 안 꺼지면 TERM
N='install/gaze_hri/lib/gaze_hri/'
for p in $(pgrep -f "$N"); do kill -INT $p; done
for i in $(seq 1 30); do pgrep -f "$N" >/dev/null || break; sleep 0.5; done
for p in $(pgrep -f "$N"); do kill -TERM $p; done
sleep 1
for p in $(pgrep -f "^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 (launch|run) gaze_hri"); do kill -TERM $p; done
sleep 1
pgrep -f "$N|bin/ros2 launch gaze_hri" >/dev/null && { echo still-running; exit 1; }
echo stopped
