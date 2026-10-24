#!/usr/bin/env bash
# 도커 저장 위치를 /var/lib/docker(루트, 99% 참) → /mnt/data/docker(1 TB 보조 디스크)로 옮긴다.
#   사용: sudo bash scripts/move_docker_to_mnt_data.sh
# - 원본은 /var/lib/docker.bak 으로 남긴다. `docker images` 가 정상인지 본 뒤에
#   sudo rm -rf /var/lib/docker.bak 으로 지우면 루트가 그만큼 빈다.
# - /mnt/data 가 마운트되기 전에는 도커가 뜨지 않게 한다(nofail 마운트라 디스크가 빠진 채
#   부팅하면 도커가 루트의 빈 /mnt/data/docker 에 새로 쌓는 사고를 막는다).
set -euo pipefail
NEW=/mnt/data/docker
[[ $EUID -eq 0 ]] || { echo "sudo 로 실행하세요: sudo bash $0"; exit 1; }
mountpoint -q /mnt/data || { echo "/mnt/data 가 마운트돼 있지 않다 — 중단"; exit 1; }
if docker info 2>/dev/null | grep -q "Docker Root Dir: $NEW"; then echo "이미 $NEW 를 쓰고 있다"; exit 0; fi
if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then
  echo "돌고 있는 컨테이너가 있다 — 끝낸 뒤 다시:"; docker ps; exit 1
fi

echo "== 1/5 도커 정지"
systemctl stop docker.socket docker.service
echo "== 2/5 복사 /var/lib/docker → $NEW (수십 GB, 몇 분 걸림)"
mkdir -p "$NEW"
rsync -aHAX --info=progress2 /var/lib/docker/ "$NEW/"
echo "== 3/5 daemon.json 에 data-root (기존 설정 유지)"
python3 - "$NEW" <<'EOF'
import json, os, sys
p = "/etc/docker/daemon.json"
d = json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else {}
if os.path.exists(p):
    os.replace(p, p + ".bak")
d["data-root"] = sys.argv[1]
json.dump(d, open(p, "w"), indent=4)
print(open(p).read())
EOF
echo "== 4/5 /mnt/data 마운트 뒤에만 도커 시작"
mkdir -p /etc/systemd/system/docker.service.d
printf '[Unit]\nRequiresMountsFor=/mnt/data\n' > /etc/systemd/system/docker.service.d/mnt-data.conf
mv /var/lib/docker /var/lib/docker.bak
systemctl daemon-reload
echo "== 5/5 도커 시작·확인"
systemctl start docker.service
docker info | grep "Docker Root Dir"
docker images
echo
echo "정상이면 원본 지우기:  sudo rm -rf /var/lib/docker.bak   (루트가 약 $(du -sh /var/lib/docker.bak | cut -f1) 빈다)"
echo "되돌리기: sudo systemctl stop docker; sudo mv /var/lib/docker.bak /var/lib/docker; sudo mv /etc/docker/daemon.json.bak /etc/docker/daemon.json; sudo systemctl start docker"
