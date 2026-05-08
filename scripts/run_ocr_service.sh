#!/usr/bin/env bash
# Wrapper khởi động OCR service standalone (port 9003).
# Service riêng để OCR không block API gateway.
set -euo pipefail

ROOT=/workspace/vbk-ai-server
cd "$ROOT/ocr_service"

if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

# PaddlePaddle 3.x: tắt mkldnn + PIR executor — workaround bug.
export FLAGS_use_mkldnn="${FLAGS_use_mkldnn:-0}"
export FLAGS_enable_pir_in_executor="${FLAGS_enable_pir_in_executor:-0}"

# Engine pool: số instance per lang. Pool lớn → parallel cao, RAM tăng.
# CPU mode init ~2s/engine. 8 instance × 4 lang = ~32 engine, RAM ~10-15GB OK trong 377GB.
export PADDLEOCR_POOL_SIZE="${PADDLEOCR_POOL_SIZE:-8}"

# Device: PHẢI là 'cpu' trên Blackwell SM 12.0!
# paddlepaddle-gpu cu126 wheel build cho SM 7.0-9.0. Blackwell SM 12.0 không trong target list
# → detection model GPU inference silent fail (dt_polys=0, không crash).
# CPU mode work bình thường, RAM 377GB dư cho pool size lớn.
export PADDLEOCR_DEVICE="${PADDLEOCR_DEVICE:-cpu}"
# Mobile detection model: nhanh 3-5x trên CPU, accuracy giảm ~5%.
# Default 1 vì pipeline nhận diện text rõ ràng (chụp screen/document scan) là chính.
export PADDLEOCR_USE_MOBILE="${PADDLEOCR_USE_MOBILE:-1}"

# 1 worker để CUDA context init 1 lần. Pool 8 lo concurrency. KHÔNG tăng workers > 1
# nếu paddle GPU mode, vì uvicorn fork() làm CUDA context die ở child.
exec "$ROOT/.venv-api/bin/uvicorn" main:app \
    --host 0.0.0.0 \
    --port 9003 \
    --workers 1 \
    --loop uvloop \
    --http httptools \
    --access-log
