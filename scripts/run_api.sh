#!/usr/bin/env bash
# Wrapper khởi động FastAPI gateway (port 9002).
# Gateway KHÔNG nhúng PaddleOCR — proxy /v1/ocr sang ocr_service (port 9003).
set -euo pipefail

ROOT=/workspace/vbk-ai-server
cd "$ROOT/api"

if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

export VLLM_TRANSLATOR_URL="${VLLM_TRANSLATOR_URL:-http://127.0.0.1:9001}"
export VLLM_TRANSLATOR_MODEL="${VLLM_TRANSLATOR_MODEL:-translator}"
export OCR_SERVICE_URL="${OCR_SERVICE_URL:-http://127.0.0.1:9003}"

# Số uvicorn worker = process độc lập. Gateway không có CUDA → fork an toàn.
# Mỗi worker chỉ httpx + vllm_client → ~150MB RAM/worker.
# 10 worker đủ cho translate + dict + proxy ocr (vLLM tự xử lý concurrent batching).
WORKERS="${API_WORKERS:-10}"

exec "$ROOT/.venv-api/bin/uvicorn" main:app \
    --host 0.0.0.0 \
    --port 9002 \
    --workers "$WORKERS" \
    --loop uvloop \
    --http httptools \
    --access-log
