#!/usr/bin/env bash
# Sync source code lên vast.ai server — chạy TRÊN máy local
set -euo pipefail

cd "$(dirname "$0")/.."
LOCAL_ROOT=$(pwd)

REMOTE_HOST="${REMOTE_HOST:-80.59.54.98}"
REMOTE_PORT="${REMOTE_PORT:-12832}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/vbk-ai-server}"
KEY="${KEY:-$LOCAL_ROOT/server/deploy_key}"
KNOWN="${KNOWN:-$LOCAL_ROOT/server/known_hosts}"

SSH_OPTS=(-p "$REMOTE_PORT" -i "$KEY" -o IdentitiesOnly=yes -o UserKnownHostsFile="$KNOWN")

echo "Sync $LOCAL_ROOT/ → $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
echo ""

# Tạo thư mục đích nếu chưa có
ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

# Rsync — loại trừ những thứ không cần / nhạy cảm
rsync -avz --delete --progress \
    --exclude='.env' \
    --exclude='.venv-api' \
    --exclude='.venv-vllm' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='.pytest_cache' \
    --exclude='logs/' \
    --exclude='data/' \
    --exclude='server/deploy_key' \
    --exclude='*.log' \
    -e "ssh ${SSH_OPTS[*]}" \
    "$LOCAL_ROOT/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

echo ""
echo "Đã sync. Bước tiếp theo trên server:"
echo "  ssh ${SSH_OPTS[*]} $REMOTE_USER@$REMOTE_HOST"
echo "  cd $REMOTE_DIR"
echo "  cp .env.example .env && \$EDITOR .env       # sửa HF_TOKEN"
echo "  ./scripts/deploy_phase1_native.sh"
