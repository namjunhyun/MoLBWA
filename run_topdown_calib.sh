#!/bin/bash
# 탑다운 호모그래피 캘리브 (conda 회피). ★ 팔이 움직임. 데모(run_b_demo.sh)를 먼저 끌 것 — 포트/카메라를 잡고 있다.
#   bash run_topdown_calib.sh --check            # 경로 검사만 (팔 안 움직임)
#   bash run_topdown_calib.sh                    # 캘리브 (창에서 두 손가락 끝의 가운데 클릭)
# 2026-09-29: 2번 모터(어깨)가 헐거워 무게로 ~4~5cm 처진다 -> 손끝 명령 높이를 6cm 로(실제 ~1~3cm).
cd ~/MoLBWA-gaze-hri
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
source /opt/ros/jazzy/setup.bash
exec python3 -u tools/topdown_calib.py --cam 0 --tip-z 0.06 "$@"
