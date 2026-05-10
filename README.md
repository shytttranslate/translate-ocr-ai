# AI API — Translation + Dictionary + OCR + TTS

Service unified cho **Dịch thuật** (vLLM + `Qwen3-14B-AWQ`), **Từ điển song ngữ** (cùng model), **OCR** (PaddleOCR PP-OCRv5 + manga-ocr trên GPU), và **TTS** (Chatterbox Multilingual 0.5B trên GPU).

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

4 service độc lập (KHÔNG có gateway proxy), mỗi service port riêng:

| Service | Port | Endpoint chính | Mục đích |
|---|---|---|---|
| vLLM translator | 9001 | `/v1/chat/completions` | Internal — Qwen3-14B-AWQ trên GPU |
| Translate | 9002 | `POST /v1/translate`, `/v1/json`, `/v1/dict` | Dịch + từ điển (gọi vLLM) |
| OCR | 9003 | `POST /v1/ocr`, `/v1/ocr/upload`, `GET /v1/languages` | PaddleOCR PP-OCRv5 + manga-ocr (mode=manga) |
| TTS | 9004 | `POST /v1/tts`, `GET /v1/voices`, `GET /v1/languages` | Chatterbox Multilingual 0.5B — 23 ngôn ngữ |

Mỗi service đều có `GET /healthz/live` + `GET /healthz/ready` (deep) + `GET /v1/metrics` (Prometheus).

### TTS — Chatterbox Multilingual 0.5B

- **Model**: [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) (MIT) — 0.5B Llama backbone + audio diffusion decoder.
- **Ngôn ngữ (23)**: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh. **KHÔNG có Vietnamese.** Request `language_id=vi` → 422.
- **Output**: JSON với `audio_base64` (WAV PCM 16-bit, **24kHz mono**). Decode:
  ```bash
  jq -r .audio_base64 response.json | base64 -d > out.wav
  ```
- **Voice (7)**: preset profile từ [tts_service/voices/voices.json](tts_service/voices/voices.json). Client chọn `voice_id`:
  - `default` — giọng mặc định Chatterbox (không audio prompt)
  - **3 giọng nam** (LibriTTS-R, F0 spread): `male_deep` (96Hz, bass), `male_warm` (141Hz, baritone), `male_bright` (157Hz, tenor)
  - **3 giọng nữ** (LibriTTS-R, F0 spread): `female_warm` (186Hz, alto), `female_clear` (220Hz, mezzo), `female_bright` (245Hz, soprano)
  - Voice prompts (.wav 24kHz mono, 6-12s) được commit vào repo (~3MB tổng) → deploy server khác chỉ cần rsync.
  - Regen/đổi speaker bằng [scripts/build_voice_pack.py](scripts/build_voice_pack.py) — stream LibriTTS-R, F0 detection, verify Chatterbox clone OK.
- **Watermark**: mọi audio đều có Perth watermark imperceptible (responsible AI, không gỡ).
- **VRAM**: ~6-9GB FP16. Concurrency=1 mặc định (model state share GPU).
- **Generation params**: `exaggeration` (0-1, cường điệu), `cfg_weight` (0-1, CFG — cũng ảnh hưởng tempo: thấp=chậm/tự do, cao=bám voice prompt), `temperature` (0-2), `seed` (reproducible nếu set). Tool preview hiển thị `cfg_weight` với label **Speed** cho dễ hình dung. KHÔNG có speed param riêng — Chatterbox không hỗ trợ direct speed control, post-process pitch không tự nhiên cho speech nên đã bỏ.
- **Auto-trim silence/noise**: Chatterbox đôi khi sinh `long_tail` (noise/silence đuôi sau câu — gặp ~65% request) → tạo cảm giác "rè rè". Server tự `librosa.effects.trim(top_db=35)` đầu+cuối audio sau generate. Đo trên 3 voice nữ/nam: noise đuôi từ 250-850ms giảm về <50ms. Tắt qua env `TTS_TRIM_SILENCE=false`.
- **Text normalization (auto)**: Chatterbox không có normalizer built-in → server pre-process [tts_service/services/text_normalizer.py](tts_service/services/text_normalizer.py) qua `num2words`:
  - Số: `1000` → `one thousand` (cover 23 ngôn ngữ; lang nào num2words thiếu thì fallback English)
  - Currency prefix: `$1000` → `1000 dollars` (en/ja/zh/ko style)
  - Currency suffix: `1000€` → `1000 euros` (fr/de/it/es style)
  - Percent: `50%` → `50 percent` (per-locale)
  - Decimal: `3.14` → `three point one four`
  - Thousand separator: `1,234,567` → `1234567` → `one million two hundred thirty-four thousand…`
  - Tắt qua env `TTS_NORMALIZE_NUMBERS=false` nếu input đã pre-normalized.
- **Tool preview**: `GET /preview/` mount [tools/tts_preview/index.html](tools/tts_preview/index.html) — single-page UI để test API qua browser (audio player + history + curl preview).

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

# OCR (cần ảnh base64) — port 9003
curl -s -X POST http://localhost:9003/v1/ocr \
    -H 'Content-Type: application/json' \
    -d "{\"image\":\"$(base64 -w0 < /path/to/image.png)\",\"lang\":\"auto\"}" | jq

# TTS — port 9004 — Chatterbox EN
curl -s -X POST http://localhost:9004/v1/tts \
    -H 'Content-Type: application/json' \
    -d '{"text":"Hello world","language_id":"en","voice_id":"default"}' \
    | jq -r .audio_base64 | base64 -d > /tmp/tts.wav
mpv /tmp/tts.wav   # hoặc: aplay /tmp/tts.wav
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
