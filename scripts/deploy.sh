#!/usr/bin/env bash
# Deploy native cho server vast.ai (không Docker, không Redis, không supervisord).
# Chạy trên server đích, dưới root user.
#
# Stack:
#   - vLLM translator (Qwen3-14B-AWQ) port 9001 — venv riêng .venv-vllm
#   - OCR service port 9003 — venv .venv-api (PaddleOCR v5)
#   - FastAPI gateway port 9002 — venv .venv-api (proxy ocr → 9003, gọi vLLM 9001)
#
# 2 process chạy bằng nohup, pid file ở $ROOT/run/.

set -euo pipefail

ROOT="${VBK_ROOT:-/workspace/vbk-ai-server}"
VENV_API="$ROOT/.venv-api"
VENV_VLLM="$ROOT/.venv-vllm"
LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/run"
DATA_DIR="$ROOT/data"
HF_CACHE="$DATA_DIR/hf-cache"

cd "$ROOT" || { echo "Không tìm thấy $ROOT. Rsync code trước."; exit 1; }

echo "================================================"
echo " VietByte AI API — deploy native"
echo " Root: $ROOT"
echo "================================================"

# 0. Check .env
if [[ ! -f .env ]]; then
    echo "[ERR] Chưa có .env. Sao chép .env.example và sửa HF_TOKEN."
    exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ "${HF_TOKEN:-}" == "hf_xxxxxxxxxxxxxxxxxxxxxxxxx" || -z "${HF_TOKEN:-}" ]]; then
    echo "[ERR] HF_TOKEN trong .env chưa đặt thật."
    exit 1
fi

mkdir -p "$LOG_DIR" "$RUN_DIR" "$HF_CACHE"

timer_start() { TIMER_START=$(date +%s); }
timer_end() {
    local label="$1"
    local elapsed=$(($(date +%s) - TIMER_START))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    echo "    [TIMING] $label: ${mins}m${secs}s"
}

# 1. Cài apt packages
echo ""
echo "[1/7] Cài apt packages..."
timer_start
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    build-essential curl ca-certificates \
    libgl1 libglib2.0-0 libgomp1 \
    git jq
timer_end "apt-get"

if ! command -v uv >/dev/null 2>&1; then
    echo "    Cài uv (modern pip thay thế)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "    uv version: $(uv --version)"

# 2. Setup venv API
echo ""
echo "[2/7] Setup venv API + PaddleOCR v5..."
timer_start
if [[ ! -d "$VENV_API" ]]; then
    uv venv --python 3.12 "$VENV_API"
fi
VIRTUAL_ENV="$VENV_API" uv pip install --upgrade pip wheel setuptools
VIRTUAL_ENV="$VENV_API" uv pip install -r translate_service/requirements.txt
# PaddleOCR v3 (PP-OCRv5). paddlepaddle-gpu cu130 wheel có SM 12.0 (Blackwell)
# kernel — KHÔNG dùng PyPI default (chỉ có cu126, không support SM 12.0).
# Index Paddle: https://www.paddlepaddle.org.cn/packages/stable/cu130/
if ! VIRTUAL_ENV="$VENV_API" uv pip install "paddlepaddle-gpu>=3.3.1,<4.0" \
        --index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/ 2>/dev/null; then
    echo "    paddlepaddle-gpu cu130 fail → fallback CPU"
    VIRTUAL_ENV="$VENV_API" uv pip install "paddlepaddle>=3.0.0,<4.0"
fi
VIRTUAL_ENV="$VENV_API" uv pip install "paddleocr>=3.0.0,<4.0"
timer_end "venv-api"
echo "    venv-api size: $(du -sh "$VENV_API" | awk '{print $1}')"

# Warm-up PaddleOCR engines
echo ""
echo "[2.5/7] Warm-up PaddleOCR (en/ch/japan/korean)..."
timer_start
"$VENV_API/bin/python" scripts/warmup_paddleocr.py 2>&1 \
    | grep -E "^\[|===" \
    || echo "    warm-up có cảnh báo nhưng không fail deploy"
timer_end "paddleocr-warmup"

# 3. Setup venv vLLM
echo ""
echo "[3/7] Setup venv vLLM..."
timer_start
if [[ ! -d "$VENV_VLLM" ]]; then
    python3.12 -m venv "$VENV_VLLM"
fi
# vLLM 0.10.x yêu cầu setuptools>=77,<80 — pin để pip không tự nâng lên 80+/82+.
"$VENV_VLLM/bin/pip" install --upgrade --progress-bar on pip wheel "setuptools>=77,<80"
"$VENV_VLLM/bin/pip" install --progress-bar on -r requirements-vllm.txt
# Đảm bảo setuptools không bị bump bởi resolver
"$VENV_VLLM/bin/pip" install --progress-bar on "setuptools>=77,<80"
timer_end "venv-vllm"
echo "    venv-vllm size: $(du -sh "$VENV_VLLM" | awk '{print $1}')"

# 4. Cấu hình supervisord + stop process cũ
echo ""
echo "[4/7] Cấu hình supervisord + stop process cũ..."
SC="supervisorctl -c /etc/supervisor/supervisord.conf"

# Vast.ai container đã chạy supervisord daemon sẵn (quản jupyter/caddy/portal/...).
# Chỉ cần copy program config và reread.
if ! pgrep -f "supervisord.*-c" >/dev/null; then
    echo "    [WARN] supervisord daemon chưa chạy — bất thường trên Vast.ai. Khởi động..."
    /usr/local/bin/supervisord -c /etc/supervisor/supervisord.conf
    sleep 3
fi

# Stop group cũ (nếu đã có) trước khi update config
$SC stop vbk-ai:* 2>/dev/null || true
# Cleanup process zombie nếu có
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 2

mkdir -p /etc/supervisor/conf.d
cp -f vbk-supervisord.conf /etc/supervisor/conf.d/vbk-ai.conf
echo "    Copy: vbk-supervisord.conf → /etc/supervisor/conf.d/vbk-ai.conf"

$SC reread
$SC update

# 5. Start vLLM translator qua supervisord
echo ""
echo "[5/7] Start vLLM translator (supervisord)..."
: > "$LOG_DIR/vllm-translator.log"
: > "$LOG_DIR/vllm-translator-err.log"
$SC start vbk-ai:vbk-vllm-translator 2>&1 | head -5 || true
sleep 1
$SC status vbk-ai:vbk-vllm-translator | head -1

# 6. Đợi vLLM ready
echo ""
echo "[6/7] Đợi vLLM translator ready (lần đầu 5-15 phút download + load Qwen3-14B-AWQ)..."
TIMEOUT_SEC=2400
START_TS=$(date +%s)
LAST_STATUS=""
LAST_TICK=0

# vLLM log nhiều dòng tqdm bằng \r (carriage return) — tail + tr để tách thành line riêng.
# `|| true` cuối pipe để grep no-match (log chưa có pattern) không trigger pipefail exit.
extract_progress() {
    {
        tail -c 16384 "$LOG_DIR/vllm-translator.log" "$LOG_DIR/vllm-translator-err.log" 2>/dev/null \
            | tr '\r' '\n' \
            | grep -aE 'Loading safetensors|Downloading|Fetching .* files|Starting to load|Capturing CUDA graph|Application startup complete|Uvicorn running|model_runner|memory profil|[0-9]{1,3}%\|' \
            | tail -n 1 \
            | sed -E 's/^.*(INFO|WARNING|ERROR)[^]]*\] ?//; s/^\[?[0-9:,. -]*\] ?//' \
            | cut -c1-130
    } || true
}

while true; do
    if curl -fsS http://127.0.0.1:9001/v1/models -m 3 >/dev/null 2>&1; then
        echo "    [$(($(date +%s)-START_TS))s] [OK] vllm-translator ready"
        break
    fi
    elapsed=$(($(date +%s)-START_TS))
    if [[ $elapsed -gt $TIMEOUT_SEC ]]; then
        echo "    [TIMEOUT] Sau ${TIMEOUT_SEC}s. Kiểm tra log:"
        echo "      tail -f $LOG_DIR/vllm-translator-err.log"
        exit 1
    fi

    STATUS=$(extract_progress)
    if [[ -n "$STATUS" && "$STATUS" != "$LAST_STATUS" ]]; then
        echo "    [${elapsed}s] $STATUS"
        LAST_STATUS="$STATUS"
        LAST_TICK=$elapsed
    elif (( elapsed - LAST_TICK >= 30 )); then
        if [[ -n "$STATUS" ]]; then
            echo "    [${elapsed}s] (vẫn đang xử lý) $STATUS"
        else
            FALLBACK=$( { tail -n 1 "$LOG_DIR/vllm-translator-err.log" "$LOG_DIR/vllm-translator.log" 2>/dev/null | tr '\r' '\n' | tail -n 1 | cut -c1-130; } || true)
            echo "    [${elapsed}s] ${FALLBACK:-(chưa có log — vLLM đang khởi động Python venv)}"
        fi
        LAST_TICK=$elapsed
    fi
    sleep 5
done

# 7. Start OCR service + API gateway + smoke test
echo ""
echo "[7/7] Start OCR service + API gateway (supervisord) + smoke test..."
: > "$LOG_DIR/ocr.log"; : > "$LOG_DIR/ocr-err.log"
: > "$LOG_DIR/api.log"; : > "$LOG_DIR/api-err.log"

$SC start vbk-ai:vbk-ocr 2>&1 | head -5 || true
echo "    Đợi OCR service ready (port 9003)..."
for i in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:9003/healthz/live >/dev/null 2>&1; then
        echo "    OCR service ready"
        break
    fi
    sleep 2
done

$SC start vbk-ai:vbk-translate 2>&1 | head -5 || true
echo "    Đợi API gateway ready (port 9002)..."
sleep 3
for i in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:9002/healthz/ready >/dev/null 2>&1; then
        echo "    API gateway ready"
        break
    fi
    sleep 2
done
$SC status vbk-ai:* | sed 's/^/    /'

echo ""
echo "================================================"
echo " DEPLOY XONG"
echo "================================================"
echo " API local:     http://127.0.0.1:9002"
echo " API docs:      http://127.0.0.1:9002/docs"
echo " Health:        http://127.0.0.1:9002/v1/health"
echo " Models:        http://127.0.0.1:9002/v1/models"
echo " Metrics:       http://127.0.0.1:9002/v1/metrics"
echo ""
echo " vLLM translator (internal): http://127.0.0.1:9001"
echo " OCR service (internal):     http://127.0.0.1:9003"
echo ""
echo " Anh access từ máy local qua SSH tunnel:"
echo "   ssh -p 12832 -i server/deploy_key -L 9002:127.0.0.1:9002 root@80.59.54.98"
echo "   curl http://localhost:9002/v1/health"
echo ""
echo " Quản process (supervisord):"
echo "   $SC status vbk-ai:*                          # check status"
echo "   $SC restart vbk-ai:vbk-vllm-translator       # restart vLLM"
echo "   $SC restart vbk-ai:vbk-translate                   # restart API"
echo "   $SC stop vbk-ai:* / $SC start vbk-ai:*       # tắt/bật toàn group"
echo "   tail -f $LOG_DIR/vllm-translator.log $LOG_DIR/api.log"
echo "   $ROOT/scripts/check_services.sh              # health snapshot"
echo "================================================"
