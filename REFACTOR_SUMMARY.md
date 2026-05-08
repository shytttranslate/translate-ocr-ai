# Báo cáo refactor — clean source về 1 model duy nhất

**Người làm**: #161 Fullstack Max — Kiến trúc sư
**Ngày**: 2026-05-09
**Yêu cầu của anh Thịnh**: dùng `Qwen3-14B-AWQ` duy nhất cho dịch + dict, OCR dùng PaddleOCR v5 (nâng từ v2.8.1), bỏ Redis/auth/rate-limit/supervisord, xóa vLLM OCR (Qwen2-VL).

## Tổng quan thay đổi

| Loại | Số lượng | Ghi chú |
|---|---|---|
| File xóa | 13 | Code, scripts, docs, configs cũ |
| File thư mục xóa | 2 | `api/preprocessing/`, `monitoring/` |
| File sửa | 14 | Source + scripts + docs |
| File thêm | 1 | `REFACTOR_SUMMARY.md` (này) |

## File đã XÓA

### Source code
- `api/services/cache.py` — Redis cache layer (zstd + BLAKE3 + model_fingerprint)
- `api/services/auth.py` — HMAC-SHA256 API key auth
- `api/preprocessing/image.py` — SSRF guard + decompression bomb guard + pHash (không có router nào dùng)
- `api/preprocessing/__init__.py` — empty package marker
- `api/Dockerfile` — Docker image cho API (không deploy bằng Docker nữa)

### Scripts
- `scripts/run_vllm_ocr.sh` — wrapper Qwen2-VL (vLLM OCR)
- `scripts/deploy_phase1.sh` — deploy bằng docker-compose
- `scripts/smoke_test.sh` — smoke test bám docker-compose

### Configs / docs
- `vbk-supervisord.conf` — supervisord program config
- `docker-compose.yml` — compose stack 6 service
- `monitoring/prometheus.yml` — chỉ scrape config rỗng, chưa dùng thực
- `UPGRADE_VLLM.md` — hướng dẫn upgrade Qwen3-8B (đã thực hiện xong, lỗi thời)
- `DEPLOY_MANUAL.md` — manual deploy A1 (đã thực hiện xong, lỗi thời)

## File đã SỬA

### `api/config.py`
- Bỏ: `vllm_ocr_url`, `vllm_ocr_model`, `redis_url`, `redis_ratelimit_url`, `redis_pool_max_connections`, `cache_*_ttl_s`, `cache_compression_threshold_bytes`, `api_key_pepper`, `api_key_cache_ttl_s`, `ssrf_*`, `rate_limit_*`, `enable_otel`, `otel_endpoint`, `Field`, `HttpUrl` import.
- Giữ: `app_env`, `log_level`, `vllm_translator_*`, `image_max_*`, `enable_metrics`.

### `api/main.py`
- Bỏ import: `os`, `services.auth.seed_dev_key`, `services.cache.RedisCache`.
- Bỏ khởi tạo `VllmEndpoint(name="ocr", ...)` cho Qwen2-VL.
- `VllmRegistry(translator=...)` không còn `ocr=`.
- Bỏ `RedisCache` instance + `app.state.cache` + `cache.close()`.
- Bỏ block `seed_dev_key`.
- Version bump 0.1.0 → 0.2.0.

### `api/services/vllm_client.py`
- `VllmRegistry.__init__` giờ chỉ nhận `translator` (bỏ `ocr`).
- `start_all()`, `stop_all()`, `deep_health_check()` đơn lẻ cho translator.

### `api/routers/meta.py`
- Bỏ `cache.ping()` + `redis_ok` trong `/healthz/ready`.
- Bỏ `registry.ocr.list_models()` trong `/healthz/startup`.
- `/v1/models` giờ trả translator + OCR engine info dạng `{engine: "PaddleOCR v5 ...", mode: "CPU"}` thay vì OCR vLLM fingerprint.

### `api/services/ocr.py` — viết lại lớn
- Auto-detect PaddleOCR API: nếu init signature có `use_textline_orientation` hoặc `use_doc_orientation_classify` → API v3 (PP-OCRv5), dùng `engine.predict()`. Ngược lại fallback v2 (`engine.ocr(..., cls=True)`).
- Parse output v3: đọc `result.json` lấy `rec_texts`, `rec_scores`, `rec_polys`/`dt_polys`. Trả về cùng schema `OcrBlock(text, confidence, bbox)` như cũ → router không phải sửa.
- Tham số v3: `use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True`.
- Tham số v2 (fallback): giữ `ocr_version="PP-OCRv3"` + `enable_mkldnn=False` (chống SIGILL trên CPU không AVX-512).

### `api/requirements.txt`
- Bỏ: `redis[hiredis]`, `zstandard`, `opencv-python-headless`, `scipy`, `fasttext-wheel`, `tenacity`.
- Đổi: `numpy 2.1.3` → `numpy 1.26.4` (PaddlePaddle 3.x vẫn nhiều conflict với numpy 2.x).

### `requirements-vllm.txt`
- `vllm==0.6.4.post1` → `vllm>=0.7.0` (cần Qwen3 architecture).
- Bỏ pin `transformers`, `tokenizers`, `huggingface-hub` — vLLM mới tự resolve.

### `.env.example`
- Bỏ: `API_KEY_PEPPER`, `DEV_API_KEY`, `GRAFANA_PASSWORD`.
- Giữ: `APP_ENV`, `LOG_LEVEL`, `HF_TOKEN`.

### `scripts/run_api.sh`
- Bỏ export: `REDIS_URL`, `REDIS_RATELIMIT_URL`, `VLLM_OCR_URL`, `VLLM_OCR_MODEL`.

### `scripts/check_services.sh`
- Bỏ: `supervisorctl`, kiểm tra Redis, vLLM OCR, port 6379/8002.
- Thêm: smoke OCR endpoint (POST ảnh trắng test PaddleOCR có loaded chưa).

### `scripts/watch_progress.sh`
- Bỏ: `supervisorctl status` block.
- Thêm: pgrep `vllm.entrypoints|uvicorn main:app` để biết AI process còn sống.

### `scripts/upgrade_vllm.sh`
- Bỏ tất cả `supervisorctl`. Thay bằng `pkill -f vllm.entrypoints`.
- Hướng dẫn restart cuối: chạy `nohup ... &` + lưu pid file thay vì `supervisorctl restart`.

### `scripts/deploy_phase1_native.sh` — viết lại gọn
- Bỏ apt: `redis-server`, `supervisor`.
- Bỏ block cấu hình Redis (`redis.conf`).
- Bỏ block copy + reread supervisord conf.
- Cài PaddleOCR v3: thử `paddlepaddle-gpu>=3.0.0,<4.0` trước, fallback `paddlepaddle>=3.0.0,<4.0`. Sau đó `paddleocr>=3.0.0,<4.0`.
- Start process: `nohup` + pid file ở `$ROOT/run/`. Stop bằng `kill $(cat pid)`.
- Bỏ chờ Redis ready, bỏ chờ vLLM OCR ready — chỉ chờ vLLM translator.
- Bỏ smoke test docker-compose.

### `scripts/warmup_paddleocr.py`
- Auto-detect API v3/v2 (giống logic trong `services/ocr.py`).
- v3: dùng `engine.predict(img)`. v2: dùng `engine.ocr(img, cls=True)`.

### `README.md` — viết lại gọn
- Mô tả kiến trúc mới: 1 vLLM + 1 FastAPI (kèm PaddleOCR CPU).
- Bỏ phần Docker compose deploy đường B.
- Bỏ tham chiếu Phase 1-8 cũ, "Bảo mật đã áp dụng", "Hạn chế đã biết".
- Hướng dẫn quản process bằng `nohup`/pid file.

## Breaking changes vs trước

| Cũ | Mới |
|---|---|
| 4 process (Redis + 2 vLLM + API) qua supervisord | 2 process (1 vLLM + 1 API) qua `nohup` |
| `/v1/models` trả 2 endpoint translator + ocr (vLLM) | trả translator (vLLM) + ocr engine info (PaddleOCR) |
| `/healthz/ready` deep check translator + ocr vLLM + Redis | chỉ deep check translator |
| API key bắt buộc cho endpoint business | không auth (cá nhân dùng, đặt sau reverse proxy nếu cần) |
| Redis cache cho translation/OCR | không cache |
| Rate-limit Redis-Lua per tier | không rate-limit |
| `/v1/ocr` đã không dùng vLLM (PaddleOCR), nay xác nhận chính thức | tương tự nhưng PaddleOCR v5 thay vì v2.8.1 |
| Python deps: redis, zstandard, scipy, opencv, fasttext, tenacity | đã bỏ |
| numpy 2.1.3 | numpy 1.26.4 (compat PaddlePaddle 3.x) |
| vLLM 0.6.4.post1 (chỉ Qwen2) | vLLM >=0.7.0 (Qwen3) |
| PaddleOCR 2.8.1 (PP-OCRv3) | PaddleOCR 3.x (PP-OCRv5) |

## Hướng dẫn deploy mới (cho anh Thịnh)

### 1. Sync code lên server

```bash
cd /home/thinh/AI_server
./scripts/sync_to_server.sh
```

### 2. SSH vào server kèm tunnel port

```bash
ssh -p 26083 -i server/deploy_key \
    -o UserKnownHostsFile=server/known_hosts \
    -L 9000:127.0.0.1:9000 \
    root@89.221.67.144
```

### 3. Stop stack cũ + xóa supervisord program (1 lần duy nhất)

```bash
cd /workspace/vbk-ai-server
SC="supervisorctl -c /etc/supervisor/supervisord.conf"

# Stop hết group cũ
$SC stop vbk-ai:* 2>/dev/null || true

# Xóa config supervisord cũ
rm -f /etc/supervisor/conf.d/vbk-ai.conf
$SC reread 2>/dev/null
$SC update 2>/dev/null

# Stop Redis nếu còn
pkill -f "redis-server" 2>/dev/null || true

# Stop vLLM OCR nếu còn
pkill -f "Qwen2-VL" 2>/dev/null || true

# Verify GPU sạch
nvidia-smi
```

### 4. Cài venv mới + start stack

```bash
cd /workspace/vbk-ai-server

# Update .env (chỉ cần HF_TOKEN; nếu .env cũ vẫn có giá trị thật thì giữ nguyên)
# cp .env.example .env  # chỉ làm nếu .env cũ đã sai

# Deploy (script sẽ stop process cũ qua pkill nếu còn, rồi start 2 process mới)
./scripts/deploy_phase1_native.sh
```

Lần đầu (cài lại venv-vllm cho vLLM 0.7+): 15-30 phút.

Lần sau (đã có venv-vllm + HF cache): 1-3 phút.

### 5. Smoke test

```bash
# Trên server
./scripts/check_services.sh

# Hoặc từ máy local (qua tunnel)
curl -s http://localhost:9000/v1/health | jq
curl -s http://localhost:9000/v1/models | jq

curl -s -X POST http://localhost:9000/v1/translate \
    -H 'Content-Type: application/json' \
    -d '{"text":"Xin chào","source_lang":"vi","target_lang":"en"}' | jq

curl -s -X POST http://localhost:9000/v1/dict \
    -H 'Content-Type: application/json' \
    -d '{"word":"freedom","native_lang":"en","target_lang":"vi"}' | jq

curl -s -X POST http://localhost:9000/v1/ocr \
    -H 'Content-Type: application/json' \
    -d "{\"image\":\"$(base64 -w0 < /tmp/test.png)\",\"lang\":\"auto\"}" | jq
```

### 6. Quản process sau deploy

```bash
ROOT=/workspace/vbk-ai-server

# Status
ps -p $(cat $ROOT/run/vllm-translator.pid) 2>/dev/null
ps -p $(cat $ROOT/run/api.pid) 2>/dev/null

# Logs realtime
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

### 7. Stop toàn bộ

```bash
ROOT=/workspace/vbk-ai-server
kill $(cat $ROOT/run/api.pid) 2>/dev/null
kill $(cat $ROOT/run/vllm-translator.pid) 2>/dev/null
rm -f $ROOT/run/*.pid
```

## Verify đã chạy

- [x] `python3 ast.parse` 13 file Python — OK
- [x] `bash -n` 7 file shell — OK
- [x] grep import nội bộ — không còn ref tới `services.cache`, `services.auth`, `preprocessing`
- [x] grep deps đã bỏ — không còn ai dùng `scipy`, `cv2`, `fasttext`, `zstandard`, `tenacity`

## Cảnh báo cho anh Thịnh

1. **Lần deploy đầu sau refactor cần xóa venv-vllm cũ** vì pin vLLM khác (0.6.4 → 0.7.x). Script `deploy_phase1_native.sh` không tự xóa — nếu venv cũ còn, hãy chạy: `rm -rf $ROOT/.venv-vllm && ./scripts/deploy_phase1_native.sh` HOẶC chạy `./scripts/upgrade_vllm.sh` để upgrade trước rồi chạy deploy.

2. **PaddleOCR v3 lần đầu sẽ download lại model** (~50MB/lang × 5 lang = 250MB), vì cache PP-OCRv3 (v2 cũ) khác PP-OCRv5 (v3 mới). Step warm-up trong `deploy_phase1_native.sh` lo việc này.

3. **Supervisord cũ trên vast.ai container có thể vẫn auto-restart Redis/vLLM-ocr nếu chưa xóa config** `/etc/supervisor/conf.d/vbk-ai.conf`. Bước 3 trong hướng dẫn deploy là 1-lần-duy-nhất để dọn.

4. **Endpoint `/v1/ocr-translate` chưa có** trong source này (Phase 4 cũ). Nếu cần, em viết riêng — không phải scope refactor.

5. **Reverse proxy / TLS / public access**: hiện endpoint chỉ bind 127.0.0.1. Public ra ngoài thì đặt nginx/caddy phía trước, tự quản TLS + IP allowlist nếu muốn.
