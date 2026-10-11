#!/bin/bash
# 시선 안경 없이 시연: MuJoCo 지원이(뷰어) + 웹 조작 패널. 브라우저 http://localhost:8080 에서 각도·동작 선택.
#   사용: bash g1/tools/sim_panel.sh            (VIEWER= 로 비우면 창 없이)
REPO=$(cd "$(dirname "$0")/../.." && pwd); ART=${ART:-$HOME/molbwa_g1}
POLICY=${POLICY:-$ART/policy_v5b_15998.npz}
MOTION=${MOTION:-$HOME/whole_body_tracking/motions/molbwa_library5.npz}
META=${META:-$ART/lib5/library_meta.json}
cd "$REPO/g1/deploy"
env -u PYTHONPATH -u LD_LIBRARY_PATH $HOME/miniconda3/envs/g1deploy/bin/python g1_motion_server.py \
    --policy "$POLICY" --motion "$MOTION" --library_meta "$META" --backend mujoco ${VIEWER---viewer} --sim_realtime --run_sec ${RUN_SEC:-0} ${EXTRA:-} &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
python3 "$REPO/g1/g1_panel.py" --port ${PORT:-8080}
