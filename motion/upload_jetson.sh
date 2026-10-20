#!/bin/bash
# 지원이 젯슨 ~/molbwa_g1 에 데모2 최종본을 올리고 md5 를 대조한다. 로봇 명령은 보내지 않는다.
#   사용: bash motion/upload_jetson.sh [젯슨 IP]        (기본 192.168.50.107)
# 정책이 돌고 있으면(모션 서버 실행 중) 올리지 않는다 — 패널 '정지' 후 다시.
set -e
J=unitree@${1:-192.168.50.107}
R=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf $T' EXIT
cp "$R"/g1/deploy/{g1_motion_server.py,segment_player.py,g1_ctl.py,g1_audio.py,g1_tracking_policy_meta.json} "$R"/g1/g1_protocol.py $T/
cp "$R"/motion/policy_v16_22900.npz $T/policy.npz
cp "$R"/motion/molbwa_library12.npz $T/library.npz
cp "$R"/motion/library_meta.json $T/library_meta.json
if ssh -o BatchMode=yes -o ConnectTimeout=5 $J 'pgrep -f "[g]1_motion_server.py" >/dev/null'; then
  echo "거부: 젯슨에서 모션 서버(정책)가 돌고 있다. 패널에서 정지한 뒤 다시 실행."; exit 1
fi
ssh -o BatchMode=yes $J 'mkdir -p ~/molbwa_g1'
scp -o BatchMode=yes -q $T/* $J:~/molbwa_g1/
F="g1_motion_server.py segment_player.py g1_ctl.py g1_audio.py g1_tracking_policy_meta.json g1_protocol.py policy.npz library.npz library_meta.json"
diff <(cd $T && md5sum $F) <(ssh -o BatchMode=yes $J "cd ~/molbwa_g1 && md5sum $F") && echo "md5 일치 — 업로드 완료"
