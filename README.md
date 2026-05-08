# VietByte AI API — OCR + Translation

Service unified phục vụ OCR (Qwen2.5-VL-7B) và Translation (Qwen2.5-14B FP8) trên cùng 1 GPU, deploy qua Docker.

> **Phase hiện tại: 1 — Foundation.** Chỉ có health check, model registry, skeleton router. Chưa có business logic translate/ocr (sẽ thêm ở Phase 2-3).

## Yêu cầu hardware

- 1× GPU NVIDIA có FP8 hoặc BF16 support, **VRAM ≥ 48GB** khuyến nghị (RTX 6000 Ada, L40S, A100, H100, …)
- Driver NVIDIA ≥ 535
- Docker ≥ 24 + `nvidia-container-toolkit`
- 64GB RAM host khuyến nghị (Redis 6GB + buffer + image preprocessing)
- ≥ 50GB disk cho weight cache (`hf-cache` volume)

## Cấu trúc

```
AI_server/
├── docker-compose.yml         # 6 service: 2 vLLM + Redis + API + Prometheus + Grafana
├── .env.example               # Template env var
├── api/                       # FastAPI gateway
│   ├── main.py
│   ├── config.py
│   ├── routers/meta.py        # /healthz/*, /v1/health, /v1/models, /v1/metrics
│   ├── services/
│   │   ├── vllm_client.py     # Async client tới 2 vLLM upstream
│   │   ├── cache.py           # Redis + zstd + BLAKE3 + model_fingerprint
│   │   └── auth.py            # HMAC API key (skeleton)
│   ├── preprocessing/image.py # SSRF guard + decompression bomb guard + pHash
│   ├── models/schemas.py      # Pydantic v2 cho mọi endpoint
│   └── utils/logging.py       # Structlog JSON + redaction
├── monitoring/prometheus.yml
└── scripts/
    ├── deploy_phase1.sh       # Bring up stack + đợi ready
    └── smoke_test.sh          # Verify health
```

## Deploy nhanh (Phase 1)

### 1. Cấu hình

```bash
cp .env.example .env
$EDITOR .env
```

Cần sửa tối thiểu:
- `HF_TOKEN`: token Hugging Face để pull Qwen2.5 weights
- `API_KEY_PEPPER`: random 32+ ký tự, bắt buộc cho prod (sinh: `python -c "import secrets; print(secrets.token_urlsafe(32))"`)

### 2. Bring up stack

```bash
./scripts/deploy_phase1.sh
```

Lần đầu sẽ pull image vLLM (~5GB) + download Qwen2.5-14B (~28GB) + Qwen2.5-VL-7B (~16GB) → có thể mất 10-20 phút tuỳ băng thông. Lần sau cache lại trong volume `hf-cache`.

### 3. Smoke test

```bash
./scripts/smoke_test.sh
```

Mong đợi: status `ok` cho cả 7 step.

## Endpoint Phase 1

| Method | Path | Mục đích |
|---|---|---|
| GET | `/` | Banner |
| GET | `/healthz/live` | Liveness probe (chỉ check process) |
| GET | `/healthz/ready` | Readiness probe (deep check vLLM + Redis) |
| GET | `/healthz/startup` | Startup probe (model loaded chưa) |
| GET | `/v1/health` | Alias public của readiness |
| GET | `/v1/models` | List model + fingerprint |
| GET | `/v1/metrics` | Prometheus exposition |
| GET | `/docs` | OpenAPI Swagger UI |

## Cấu hình GPU memory

Mặc định mỗi vLLM dùng `--gpu-memory-utilization 0.35` (tổng 0.70 = 33.6GB / 48GB), chừa 14% buffer chống OOM khi 2 process expand KV cache đồng thời. Nếu có VRAM dư có thể nâng dần lên 0.40 nhưng cần monitor `nvidia-smi`.

OCR model dùng `--dtype bfloat16` thay vì FP8 vì Qwen2.5-VL-7B hiện chưa có official FP8 weights tính đến 2026-05. Khi Alibaba release sẽ chuyển sang `--quantization fp8`.

## Bảo mật đã áp dụng (Phase 1)

- API key: HMAC-SHA256 + pepper, constant-time compare (`hmac.compare_digest`), prefix `vbk_live_*`
- Image bomb: `MAX_IMAGE_PIXELS=25M`, allowlist format JPEG/PNG/WebP, reject animated
- SSRF: chỉ HTTPS, block private IP (RFC 1918, link-local, loopback, IPv6 unique-local), DNS resolve-then-fetch-by-IP, `follow_redirects=False`
- Logging: redact `image_url` thành `scheme://host/<redacted>`, API key chỉ giữ 8 ký tự đầu, không log image bytes ở mọi level
- Container API: non-root user 1000, `cap_drop: [ALL]`, `no-new-privileges`
- Header response: `X-Content-Type-Options: nosniff`, `X-Request-ID` cho tracing

## Hạn chế đã biết của Phase 1

Theo phản biện của Tier-1 đã được lưu lại trong session debate:

1. **Throughput target 150 req/s translation chưa verify trên hardware thật.** Cần Phase 0 benchmark (chưa làm theo yêu cầu deploy nhanh).
2. **SLO 99.9% với single-GPU single-host không khả thi.** Realistic phase 1 = 99.5%. Để đạt 99.9% cần multi-host failover (Phase sau).
3. **Zero-downtime model swap chưa hỗ trợ.** Hiện downtime ~30-60s khi swap model.
4. **OCR cache hit ratio target 25% có thể không đạt.** pHash đã chuẩn bị trong `preprocessing/image.py:perceptual_hash_64` nhưng cần multi-index bucket Redis (Phase 5).
5. **Priority queue cho enterprise tier chưa có.** vLLM không native priority — Phase 5 sẽ dùng `asyncio.PriorityQueue` per-worker.

## Phase tiếp theo

- **Phase 2**: `/v1/translate` end-to-end (vLLM client + cache + guided_json + Pydantic validation)
- **Phase 3**: `/v1/ocr` với image preprocessing + structured data extraction
- **Phase 4**: `/v1/ocr-translate` combined (default 2-pass cho quality)
- **Phase 5**: Rate limit Redis Lua, glossary, formality control, language detect (fastText)
- **Phase 6**: Streaming SSE cho translation
- **Phase 7**: OpenTelemetry tracing + Grafana dashboards
- **Phase 8**: Locust load test + FLORES-200 BLEU benchmark + CER OCR benchmark

## Troubleshooting

```bash
# Log từng service
docker compose logs --tail=200 vllm-translator
docker compose logs --tail=200 vllm-ocr
docker compose logs --tail=200 api
docker compose logs --tail=200 redis

# GPU usage realtime
nvidia-smi -l 1

# Redis introspect
docker compose exec redis redis-cli INFO memory
docker compose exec redis redis-cli DBSIZE

# Tear down
docker compose down            # giữ volume
docker compose down -v         # xoá luôn weight cache
```
