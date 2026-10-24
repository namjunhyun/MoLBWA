#!/usr/bin/env bash
# 도커 29 의 이미지 저장소(containerd snapshotter)를 /var/lib/containerd → /mnt/data/containerd 로.
# move_docker_to_mnt_data.sh 만으로는 이미지가 안 옮겨진다(도커 기본 폴더엔 236 KB 뿐, 이미지는 containerd 쪽).
#   사용: sudo bash scripts/move_containerd_to_mnt_data.sh
set -euo pipefail
NEW=/mnt/data/containerd
CFG=/etc/containerd/config.toml
[[ $EUID -eq 0 ]] || { echo "sudo 로 실행하세요"; exit 1; }
mountpoint -q /mnt/data || { echo "/mnt/data 가 마운트돼 있지 않다 — 중단"; exit 1; }
if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then echo "돌고 있는 컨테이너가 있다 — 끝낸 뒤 다시"; docker ps; exit 1; fi
echo "== 지금 크기"; du -sh /var/lib/containerd
echo "== 1/5 도커·containerd 정지"
systemctl stop docker.socket docker.service containerd.service
echo "== 2/5 복사 → $NEW"
mkdir -p "$NEW"
rsync -aHAX --info=progress2 /var/lib/containerd/ "$NEW/"
echo "== 3/5 containerd 설정 root"
if [[ -f "$CFG" ]]; then
  cp "$CFG" "$CFG.bak"
  if grep -qE '^\s*root\s*=' "$CFG"; then
    sed -i -E "s|^\s*root\s*=.*|root = \"$NEW\"|" "$CFG"
  else
    sed -i "1i root = \"$NEW\"" "$CFG"
  fi
else
  mkdir -p /etc/containerd
  printf 'version = 2\nroot = "%s"\n' "$NEW" > "$CFG"
fi
grep -nE '^\s*(version|root)\s*=' "$CFG"
echo "== 4/5 /mnt/data 마운트 뒤에만 시작"
mkdir -p /etc/systemd/system/containerd.service.d
printf '[Unit]\nRequiresMountsFor=/mnt/data\n' > /etc/systemd/system/containerd.service.d/mnt-data.conf
mv /var/lib/containerd /var/lib/containerd.bak
systemctl daemon-reload
echo "== 5/5 시작·확인"
systemctl start containerd.service docker.service
docker info | grep -E "Docker Root Dir|driver-type"
docker images
df -h / | tail -1
echo
echo "정상이면: sudo rm -rf /var/lib/containerd.bak /var/lib/docker.bak   (루트가 약 $(du -sh /var/lib/containerd.bak | cut -f1) 빈다)"
echo "되돌리기: sudo systemctl stop docker containerd; sudo rm -rf /var/lib/containerd; sudo mv /var/lib/containerd.bak /var/lib/containerd;"
echo "          sudo mv $CFG.bak $CFG (원래 없었으면 sudo rm $CFG); sudo systemctl start containerd docker"
