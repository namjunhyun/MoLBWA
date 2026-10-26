#!/bin/bash
# 끄고(작업 중이면 거부) 다 꺼진 걸 확인한 뒤에만 새로 띄운다. 로그 경로는 gui_log_path
D=$(dirname $0)
bash $D/stop_gui.sh || exit 1
L=~/MoLBWA-gaze-hri/notes/gui_dual_$(date +%m%d_%H%M%S).log
echo $L > $D/gui_log_path
setsid bash $D/run_dual.sh > $L 2>&1 < /dev/null &
disown
echo "$L"
