#!/usr/bin/env bash
# Phase 1 deploy script — chạy trên server có GPU NVIDIA + Docker + nvidia-container-toolkit
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "Chưa có file .env. Sao chép từ .env.example và sửa các giá trị:"
    echo "  cp .env.example .env && \$EDITOR .env"
    exit 1
fi

# shellcheck disable=SC1091
source .env

if [[ "${HF_TOKEN:-}" == "hf_xxxxxxxxxxxxxxxxxxxxxxxxx" || -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN chưa được set thực. Sửa file .env trước khi deploy."
    exit 1
fi

if [[ "${API_KEY_PEPPER:-}" == "replace-this-with-32-bytes-random-pepper-in-prod" ]]; then
    echo "Cảnh báo: API_KEY_PEPPER chưa thay. OK cho dev, BẮT BUỘC sửa cho prod."
fi

echo "[1/4] Kiểm tra GPU..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Không tìm thấy nvidia-smi. Cài NVIDIA driver + nvidia-container-toolkit trước."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[2/4] Build API image..."
docker compose build api

echo "[3/4] Pull vLLM + Redis + Prometheus + Grafana..."
docker compose pull vllm-translator vllm-ocr redis prometheus grafana

echo "[4/4] Khởi động stack (vLLM cần ~3-5 phút để load weights lần đầu)..."
docker compose up -d

echo ""
echo "Đang đợi readiness... (timeout 8 phút)"
for i in $(seq 1 96); do
    if curl -fsS http://localhost:8000/healthz/ready >/dev/null 2>&1; then
        echo "API ready sau ${i}*5s"
        break
    fi
    sleep 5
    if [[ $i -eq 96 ]]; then
        echo "Timeout. Kiểm tra log: docker compose logs --tail=200 vllm-translator vllm-ocr api"
        exit 1
    fi
done

echo ""
echo "=== Phase 1 ready ==="
echo "API:        http://localhost:8000"
echo "Docs:       http://localhost:8000/docs"
echo "Health:     http://localhost:8000/v1/health"
echo "Metrics:    http://localhost:8000/v1/metrics"
echo "Models:     http://localhost:8000/v1/models"
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000 (admin / \$GRAFANA_PASSWORD)"
echo ""
echo "Test smoke:"
echo "  curl -s http://localhost:8000/v1/health | jq"
echo "  curl -s http://localhost:8000/v1/models | jq"
