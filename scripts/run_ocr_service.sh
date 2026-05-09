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

# PaddlePaddle 3.x: tắt mkldnn + PIR executor — workaround bug oneDNN PIR attribute.
export FLAGS_use_mkldnn="${FLAGS_use_mkldnn:-0}"
export FLAGS_enable_pir_in_executor="${FLAGS_enable_pir_in_executor:-0}"

# Engine pool — 1 worker × pool 8 = 8 parallel inference per lang.
# CPU release GIL trong predict() nên thread pool 8 effective parallelism.
export PADDLEOCR_POOL_SIZE="${PADDLEOCR_POOL_SIZE:-8}"

# Device: BẮT BUỘC 'cpu' trên Blackwell SM 12.0!
# paddlepaddle-gpu cu126 wheel không có kernel SM 12.0 → detection silent fail.
export PADDLEOCR_DEVICE="${PADDLEOCR_DEVICE:-cpu}"
# Server detection model: accuracy cao hơn mobile ~5%, chậm hơn 3-5x trên CPU.
export PADDLEOCR_USE_MOBILE="${PADDLEOCR_USE_MOBILE:-0}"

# Resize ảnh > MAX_DIMENSION (default 1600px) → giảm latency 2-3x với ảnh lớn.
export PADDLEOCR_MAX_DIMENSION="${PADDLEOCR_MAX_DIMENSION:-1600}"
# Textline orientation classification — tắt mặc định (nhanh ~10-15%). Bật nếu text xoay 90/180/270°.
export PADDLEOCR_USE_TEXTLINE_ORI="${PADDLEOCR_USE_TEXTLINE_ORI:-0}"
# LRU cache size — repeat request cùng ảnh + lang trả ngay (ms).
export PADDLEOCR_CACHE_SIZE="${PADDLEOCR_CACHE_SIZE:-256}"

# Workers: paddle CPU + fork CÓ ISSUE — đôi khi worker crash khi xử lý ảnh lớn
# (PIL/numpy/paddle Conv kernel race condition). Giảm xuống 1 cho stable.
# Concurrency dựa vào engine pool size + asyncio. Trade-off throughput cho stability.
WORKERS="${OCR_WORKERS:-1}"

exec "$ROOT/.venv-api/bin/uvicorn" main:app \
    --host 0.0.0.0 \
    --port 9003 \
    --workers "$WORKERS" \
    --loop uvloop \
    --http httptools \
    --access-log
