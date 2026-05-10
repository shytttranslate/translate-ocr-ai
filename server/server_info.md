# Server GPU đích — vast.ai instance

> **Trạng thái:** Connected. Authenticated qua key `server/deploy_key`. Container đang chạy.

## Thông tin kết nối

| Field | Value |
|---|---|
| Provider | **vast.ai** (GPU rental, container-based) |
| Public IP | `89.221.67.144` |
| SSH port | `26083` (mapped từ container :22) |
| User | `root` |
| Container ID | `dbb84172d404` |
| Vast container label | `C.36332288` |
| Key dùng | `/home/thinh/AI_server/server/deploy_key` |

### Lệnh connect chuẩn

```bash
ssh -p 26083 -i /home/thinh/AI_server/server/deploy_key \
    -o IdentitiesOnly=yes \
    -o UserKnownHostsFile=/home/thinh/AI_server/server/known_hosts \
    root@89.221.67.144
```

Kèm port forward 8080 (theo lệnh anh đưa):
```bash
ssh -p 26083 -i /home/thinh/AI_server/server/deploy_key \
    -o UserKnownHostsFile=/home/thinh/AI_server/server/known_hosts \
    -L 8080:localhost:8080 \
    root@89.221.67.144
```

## Hardware

| Component | Spec |
|---|---|
| **GPU** | NVIDIA RTX 6000 Ada Generation, **48GB VRAM** |
| GPU driver | 565.57.01 |
| Compute capability | 8.9 (Ada Lovelace) |
| CPU | AMD EPYC 7C13 64-Core Processor (128 vCPU) |
| RAM | **503GB** total |
| Swap | 8GB |
| Disk `/workspace` (persistent) | **236GB NVMe** |
| Disk `/` (overlay, volatile) | 32GB |
| OS | Ubuntu 24.04.4 LTS, kernel 5.15.0-144 |

## Host key (port 26083)

| Algorithm | Fingerprint SHA256 |
|---|---|
| **ED25519** (active) | `Q8KMjO7q9sMgzDms3ceQhxjrwBV+ucbmBNodDLkWUwY` |
| RSA-3072 | `dAgXn5cX6Za8UgTjdANWMvcneWQp4/+izzu13wld7nQ` |
| ECDSA-P256 | `LvDAKMwjmsO36YbJUuIyzib4heKO2LLUTwSQLpvwSDI` |

## Vast.ai port forwarding

| Container port | Public port | Service đang chạy sẵn |
|---|---|---|
| 22 | **26083** | SSH (sshd) |
| 8080 | 23287 | **Jupyter Notebook** — đang chiếm port 8080 |
| 6006 | 28730 | TensorBoard |
| 1111 | 24507 | Instance Portal (Caddy + FastAPI) |
| 8384 | 31225 | Syncthing |

## CRITICAL: Đây là Docker container, KHÔNG có Docker engine bên trong

**Docker engine: KHÔNG có** — `docker` không tồn tại.
**nvidia-container-toolkit: KHÔNG có** — `nvidia-container-cli` không tồn tại.

Nghĩa là:
- ❌ `docker-compose.yml` em đã viết **không chạy được trực tiếp** trên container này.
- ✅ GPU truy cập được qua `nvidia-smi` (driver mount sẵn từ host).
- ✅ CUDA libraries có ở `/usr/local/cuda` và `/usr/local/nvidia/bin`.
- ✅ `/workspace` là persistent storage 236GB — phải deploy code vào đây.

## Process đã chạy sẵn (cần lưu ý conflict port)

```
sshd          0.0.0.0:22       (public 26083)
jupyter-noteb 0.0.0.0:8080     (public 23287)  ← chiếm 8080
tensorboard   127.0.0.1:16006
caddy         127.0.0.1:2019, *:1111, *:8384
fastapi       127.0.0.1:11111, 11112  (Vast Instance Portal)
cloudflared   127.0.0.1:20241-20244
```

→ **API phải bind port khác 8080** (ví dụ 9000), hoặc kill Jupyter trước.
→ Để Jupyter của anh không bị mất, em đề xuất **bind API port 9000**, public qua port forward riêng.

## Kế hoạch deploy điều chỉnh

`docker-compose.yml` đã viết KHÔNG dùng được. Em phải làm 1 trong 3 cách:

### Cách A — Native install trên container (em khuyến nghị)
- Cài Python 3.11 + venv qua `apt` (Ubuntu 24.04 sẵn)
- `pip install vllm` chạy trực tiếp 2 vLLM process
- Cài Redis qua `apt install redis-server`
- Chạy FastAPI API qua uvicorn
- Manage process bằng `tmux` hoặc `systemd --user` hoặc `supervisord`
- Persistent: tất cả vào `/workspace/vbk-ai-server/`

**Ưu**: tận dụng GPU + driver sẵn, không cần nested Docker. **Nhược**: phải rewrite deploy script.

### Cách B — Cài Docker trong container (Docker-in-Docker)
- Phức tạp, có thể không work do container privilege
- Vast.ai thường disable `--privileged` cho user → DinD fail

### Cách C — Đổi instance vast.ai có sẵn Docker (vd image `vastai/pytorch:cuda-12.4-base`)
- Tốn công, phải tear down + spawn instance mới
- Mất state hiện có

→ Em chọn **Cách A** trừ khi anh chỉ định khác.

## Bước tiếp theo

Em sẽ rewrite plan deploy native: tạo `scripts/deploy_phase1_native.sh` chạy thẳng trên `/workspace/`. Anh confirm em làm Cách A?
