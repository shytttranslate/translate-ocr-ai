#!/usr/bin/env bash
# Wrapper khởi động translate service (port 9002).
# Endpoints: /v1/translate, /v1/json, /v1/dict — chung vLLM Qwen3-14B-AWQ.
# OCR là service RIÊNG port 9003, không liên quan service này.
set -euo pipefail

ROOT=/workspace/vbk-ai-server
cd "$ROOT/translate_service"

if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

export VLLM_TRANSLATOR_URL="${VLLM_TRANSLATOR_URL:-http://127.0.0.1:9001}"
export VLLM_TRANSLATOR_MODEL="${VLLM_TRANSLATOR_MODEL:-translator}"

# Số uvicorn worker = process độc lập. Service không nhúng CUDA → fork an toàn.
# Mỗi worker chỉ httpx → ~150MB RAM/worker. 10 worker đủ cho 100 req/s.
WORKERS="${TRANSLATE_WORKERS:-10}"

exec "$ROOT/.venv-api/bin/uvicorn" main:app \
    --host 0.0.0.0 \
    --port 9002 \
    --workers "$WORKERS" \
    --loop uvloop \
    --http httptools \
    --access-log
