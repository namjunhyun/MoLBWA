#!/usr/bin/env bash
# move_docker_to_mnt_data.sh · move_containerd_to_mnt_data.sh 뒤 확인이 끝나면 루트의 원본 사본을 지운다.
#   사용: sudo bash scripts/remove_docker_bak.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "sudo 로 실행하세요"; exit 1; }
docker info 2>/dev/null | grep -q "Docker Root Dir: /mnt/data/docker" || { echo "도커가 /mnt/data 를 안 쓰고 있다 — 중단"; exit 1; }
grep -qE '^root = "/mnt/data/containerd"' /etc/containerd/config.toml || { echo "containerd 가 /mnt/data 를 안 쓰고 있다 — 중단"; exit 1; }
df -h / | tail -1
rm -rf /var/lib/containerd.bak /var/lib/docker.bak
echo "지움"
df -h / | tail -1
