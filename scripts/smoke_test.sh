#!/usr/bin/env bash
# Smoke test Phase 1 — verify foundation hoạt động
set -euo pipefail

API="${API:-http://localhost:8000}"

echo "=== Smoke test Phase 1 ==="
echo "Target: $API"
echo ""

echo "[1] GET /"
curl -fsS "$API/" | python3 -m json.tool
echo ""

echo "[2] GET /healthz/live (liveness probe)"
curl -fsS "$API/healthz/live" | python3 -m json.tool
echo ""

echo "[3] GET /healthz/startup (model loaded check)"
curl -fsS "$API/healthz/startup" | python3 -m json.tool
echo ""

echo "[4] GET /healthz/ready (deep check)"
curl -fsS "$API/healthz/ready" | python3 -m json.tool
echo ""

echo "[5] GET /v1/health (public alias)"
curl -fsS "$API/v1/health" | python3 -m json.tool
echo ""

echo "[6] GET /v1/models (model fingerprint)"
curl -fsS "$API/v1/models" | python3 -m json.tool
echo ""

echo "[7] GET /v1/metrics (Prometheus exposition)"
curl -fsS "$API/v1/metrics" | head -20
echo ""

echo "=== Phase 1 smoke OK ==="
