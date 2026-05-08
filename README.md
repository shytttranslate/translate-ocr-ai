# VietByte AI API — Translation + Dictionary + OCR

Service unified cho **Dịch thuật** (vLLM + `Qwen3-14B-AWQ`), **Từ điển song ngữ** (cùng model), và **OCR** (PaddleOCR v5 chạy CPU).

## Kiến trúc

```
                   FastAPI (port 9000)
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
   /v1/translate     /v1/dict            /v1/ocr
   /v1/json          (cùng model)        (PaddleOCR v5 CPU)
       │                   │
       └─────────┬─────────┘
                 ▼
        vLLM translator (port 9001)
        Qwen/Qwen3-14B-AWQ — GPU
```

- 1 GPU process duy nhất (vLLM serve Qwen3-14B-AWQ).
- 1 FastAPI process duy nhất, chạy luôn PaddleOCR ngay trong cùng venv (CPU).
- Không Redis, không auth, không rate-limit, không supervisord.

## Endpoint

| Method | Path | Mục đích |
|---|---|---|
| GET | `/` | Banner |
| GET | `/healthz/live` | Liveness probe (chỉ check process) |
| GET | `/healthz/ready` | Readiness probe (deep check vLLM bằng inference call) |
| GET | `/healthz/startup` | Startup probe (model loaded chưa) |
| GET | `/v1/health` | Alias public của readiness |
| GET | `/v1/models` | Translator fingerprint + OCR engine info |
| GET | `/v1/metrics` | Prometheus exposition |
| POST | `/v1/translate` | Dịch text đơn / batch |
| POST | `/v1/json` | Dịch mảng string, output mảng string cùng thứ tự |
| POST | `/v1/dict` | Tra từ điển song ngữ (Cambridge-style) |
| POST | `/v1/ocr` | OCR ảnh base64, trả text blocks + bbox |
| GET | `/docs` | Swagger UI |

## Yêu cầu hardware

- 1× GPU NVIDIA, **VRAM ≥ 24GB** (Qwen3-14B-AWQ ~9.5GB + KV cache).
- Driver NVIDIA ≥ 535.
- ≥ 50GB disk cho HF weight cache.
- 16GB+ RAM (PaddleOCR mỗi lang pack tốn ~200-500MB).

## Deploy nhanh

### Trên server (vast.ai container hoặc bare-metal Linux)

```bash
cd /workspace/vbk-ai-server

# 1. Cấu hình env
cp .env.example .env
$EDITOR .env                    # đặt HF_TOKEN

# 2. Deploy (cài venv + start 2 process)
./scripts/deploy_phase1_native.sh
```

Lần đầu mất 15-40 phút (pip install vLLM ~10GB + download Qwen3-14B-AWQ ~9.5GB + PaddleOCR model ~50MB/lang).

### Truy cập từ máy local qua SSH tunnel

```bash
ssh -p 12832 -i server/deploy_key -L 9002:127.0.0.1:9002 root@80.59.54.98
# terminal khác:
curl http://localhost:9002/v1/health
```

### Smoke test

```bash
# Translate
curl -s -X POST http://localhost:9002/v1/translate \
    -H 'Content-Type: application/json' \
    -d '{"text":"Xin chào","source_lang":"vi","target_lang":"en"}' | jq

# Dict
curl -s -X POST http://localhost:9002/v1/dict \
    -H 'Content-Type: application/json' \
    -d '{"word":"freedom","native_lang":"en","target_lang":"vi"}' | jq

# OCR (cần ảnh base64)
curl -s -X POST http://localhost:9002/v1/ocr \
    -H 'Content-Type: application/json' \
    -d "{\"image\":\"$(base64 -w0 < /path/to/image.png)\",\"lang\":\"auto\"}" | jq
```

Hoặc trên server:

```bash
./scripts/check_services.sh
```

## Quản process

Mọi process chạy bằng `nohup` + pid file ở `$ROOT/run/`.

```bash
ROOT=/workspace/vbk-ai-server

# Status
cat $ROOT/run/vllm-translator.pid    # → PID
ps -p $(cat $ROOT/run/vllm-translator.pid)
ps -p $(cat $ROOT/run/api.pid)

# Logs
tail -f $ROOT/logs/vllm-translator.log
tail -f $ROOT/logs/api.log

# Restart API
kill $(cat $ROOT/run/api.pid)
nohup $ROOT/scripts/run_api.sh > $ROOT/logs/api.log 2> $ROOT/logs/api-err.log &
echo $! > $ROOT/run/api.pid

# Restart translator
kill $(cat $ROOT/run/vllm-translator.pid)
nohup $ROOT/scripts/run_vllm_translator.sh > $ROOT/logs/vllm-translator.log 2> $ROOT/logs/vllm-translator-err.log &
echo $! > $ROOT/run/vllm-translator.pid
```

## Cấu trúc source

```
AI_server/
├── .env.example
├── README.md
├── requirements-vllm.txt
├── REFACTOR_SUMMARY.md
├── api/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── config.py               # Settings (env-driven)
│   ├── requirements.txt
│   ├── routers/
│   │   ├── meta.py             # /healthz/*, /v1/health, /v1/models
│   │   ├── translate.py        # /v1/translate, /v1/json
│   │   ├── dict.py             # /v1/dict
│   │   └── ocr.py              # /v1/ocr
│   ├── services/
│   │   ├── vllm_client.py      # Async client tới vLLM
│   │   ├── translator.py       # Prompt + parse + script-purity check
│   │   ├── dictionary.py       # Prompt Cambridge-style
│   │   └── ocr.py              # PaddleOCR wrapper (v3 API + fallback v2)
│   ├── models/schemas.py       # Pydantic v2
│   └── utils/logging.py        # Structlog JSON
└── scripts/
    ├── deploy_phase1_native.sh # Bring up stack đầy đủ
    ├── run_api.sh              # Wrapper FastAPI
    ├── run_vllm_translator.sh  # Wrapper vLLM translator
    ├── upgrade_vllm.sh         # Upgrade vLLM venv khi cần
    ├── check_services.sh       # Snapshot health
    ├── watch_progress.sh       # Theo dõi tiến độ deploy từ local
    ├── warmup_paddleocr.py     # Pre-load PaddleOCR engines
    └── sync_to_server.sh       # rsync code lên server
```

## Cấu hình GPU memory

Mặc định `--gpu-memory-utilization 0.65` cho Qwen3-14B-AWQ trên VRAM 48GB → ~31GB total (~9.5GB weights + ~20GB KV cache). Có thể nâng `0.80` nếu cần thêm batch capacity, monitor bằng `nvidia-smi`.

## Troubleshooting

```bash
# vLLM crash khi load
tail -100 $ROOT/logs/vllm-translator.log
tail -50 $ROOT/logs/vllm-translator-err.log

# GPU bị process zombie chiếm
nvidia-smi
kill -9 <PID>

# PaddleOCR cold-start chậm — chạy warm-up
$ROOT/.venv-api/bin/python $ROOT/scripts/warmup_paddleocr.py

# API không respond
curl -v http://127.0.0.1:9002/healthz/live
ps -p $(cat $ROOT/run/api.pid)
```
