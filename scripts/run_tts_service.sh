#!/usr/bin/env bash
# Wrapper khởi động TTS service standalone (port 9004).
# Chatterbox Multilingual 0.5B (ResembleAI, MIT) — 23 ngôn ngữ, KHÔNG có VI.
# GPU Blackwell SM 12.0, torch cu130, FP16, ~6-9GB VRAM.
set -euo pipefail

ROOT=/workspace/vbk-ai-server
cd "$ROOT/tts_service"

if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

# Cache HF tránh download lại Chatterbox weights mỗi restart container.
export HF_HOME="${HF_HOME:-$ROOT/data/hf-cache}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# 1 worker — model state share GPU, fork không an toàn (giống OCR).
# Concurrency dựa vào asyncio.Semaphore trong engine (TTS_CONCURRENCY=1 mặc định).
WORKERS="${TTS_WORKERS:-1}"

exec "$ROOT/.venv-tts/bin/uvicorn" main:app \
    --host 0.0.0.0 \
    --port 9004 \
    --workers "$WORKERS" \
    --loop uvloop \
    --http httptools \
    --access-log
