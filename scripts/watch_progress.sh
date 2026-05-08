#!/usr/bin/env bash
# Theo dõi tiến độ deploy từ máy local. Chạy ở terminal khác trong khi deploy.
# Hiển thị: dung lượng venv, số file, GPU mem, process còn sống.

set -uo pipefail

cd "$(dirname "$0")/.."

REMOTE_HOST="${REMOTE_HOST:-80.59.54.98}"
REMOTE_PORT="${REMOTE_PORT:-12832}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/vbk-ai-server}"
KEY="${KEY:-$(pwd)/server/deploy_key}"
KNOWN="${KNOWN:-$(pwd)/server/known_hosts}"

SSH_OPTS=(-p "$REMOTE_PORT" -i "$KEY" -o IdentitiesOnly=yes -o UserKnownHostsFile="$KNOWN" -o BatchMode=yes)

echo "Theo dõi tiến độ deploy mỗi 5s. Ctrl+C để dừng."
echo ""

while true; do
    clear
    echo "=== $(date '+%H:%M:%S') — Theo dõi deploy $REMOTE_DIR ==="
    echo ""

    ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" "
        echo '--- Disk usage venv ---'
        for d in $REMOTE_DIR/.venv-api $REMOTE_DIR/.venv-vllm $REMOTE_DIR/data/hf-cache; do
            if [ -d \$d ]; then
                size=\$(du -sh \$d 2>/dev/null | awk '{print \$1}')
                count=\$(find \$d -type f 2>/dev/null | wc -l)
                printf '  %-50s %10s  %d files\n' \$d \$size \$count
            else
                printf '  %-50s (chưa tồn tại)\n' \$d
            fi
        done

        echo ''
        echo '--- Supervisord vbk-ai ---'
        supervisorctl -c /etc/supervisor/supervisord.conf status vbk-ai:* 2>/dev/null | sed 's/^/  /' || echo '  (vbk-ai chưa load)'

        echo ''
        echo '--- Process AI server ---'
        pgrep -af 'vllm.entrypoints|uvicorn main:app' | head -10 || echo '  (chưa có process nào)'

        echo ''
        echo '--- Process deploy đang chạy ---'
        pgrep -af 'pip install|uv pip|python3.12.*venv|huggingface' | head -10 || echo '  (không có process deploy)'

        echo ''
        echo '--- GPU ---'
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null

        echo ''
        echo '--- Network ---'
        cat /proc/net/dev | awk 'NR>2 && \$1 ~ /eth|en/ {print \"  \" \$1 \" rx=\" \$2 \" tx=\" \$10}' | head -3
    " 2>/dev/null

    sleep 5
done
