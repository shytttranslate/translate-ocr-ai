#!/usr/bin/env bash
# Check trạng thái stack mới (vLLM translator + FastAPI + PaddleOCR).
set -uo pipefail

ROOT="${VBK_ROOT:-/workspace/vbk-ai-server}"

echo "=== Supervisord status ==="
supervisorctl -c /etc/supervisor/supervisord.conf status vbk-ai:* 2>/dev/null \
    || echo "    supervisord chưa load vbk-ai group (chạy ./scripts/deploy.sh)"

echo ""
echo "=== Process ==="
pgrep -af "vllm.entrypoints|uvicorn main:app" || echo "    không có process nào đang chạy"

echo ""
echo "=== GPU usage ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1

echo ""
echo "=== Port listen ==="
ss -tlnp 2>/dev/null | grep -E ":(9001|9002|9003)" || echo "    không có port 9001/9002/9003 đang listen"

echo ""
echo "=== Endpoint test ==="
for url in \
    "http://127.0.0.1:9001/v1/models|vLLM translator (port 9001)" \
    "http://127.0.0.1:9003/healthz/live|OCR service (port 9003)" \
    "http://127.0.0.1:9002/healthz/live|API liveness (port 9002)" \
    "http://127.0.0.1:9002/healthz/ready|API readiness (deep)" \
    "http://127.0.0.1:9002/v1/health|API public health" \
    "http://127.0.0.1:9002/v1/models|API models meta"; do
    target=$(echo "$url" | cut -d'|' -f1)
    label=$(echo "$url" | cut -d'|' -f2)
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "$target" 2>/dev/null)
    if [[ "$code" == "200" ]]; then
        echo "[OK]   $label ($code)"
    else
        echo "[FAIL] $label ($code)"
    fi
done

echo ""
echo "=== Smoke OCR (PaddleOCR) ==="
# Tạo ảnh trắng 100x40 PNG base64 — request OCR với lang=en, không kỳ vọng text
b64=$(python3 -c "import base64,io;from PIL import Image;Image.new('RGB',(100,40),'white').save(b:=io.BytesIO(),'PNG');print(base64.b64encode(b.getvalue()).decode())" 2>/dev/null || echo "")
if [[ -n "$b64" ]]; then
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 30 -X POST \
        -H "Content-Type: application/json" \
        -d "{\"image\":\"$b64\",\"lang\":\"en\"}" \
        "http://127.0.0.1:9002/v1/ocr" 2>/dev/null)
    if [[ "$code" == "200" ]]; then
        echo "[OK]   PaddleOCR loaded ($code)"
    else
        echo "[FAIL] PaddleOCR ($code)"
    fi
else
    echo "[SKIP] PaddleOCR (Pillow không có sẵn ở shell hiện tại)"
fi

echo ""
echo "=== Log tail (10 dòng cuối) ==="
for log in vllm-translator api; do
    f="$ROOT/logs/$log.log"
    e="$ROOT/logs/$log-err.log"
    echo "--- $log ---"
    [[ -f "$f" ]] && tail -3 "$f" 2>/dev/null
    [[ -f "$e" ]] && tail -3 "$e" 2>/dev/null
done
