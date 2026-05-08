#!/usr/bin/env bash
# Wrapper khởi động vLLM translator — gọi qua nohup từ deploy script.
set -euo pipefail

ROOT=/workspace/vbk-ai-server
cd "$ROOT"

# Load .env (trip vars sang export)
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export HF_HOME="$ROOT/data/hf-cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# HuggingFace Xet storage hay 500-internal trên một số repo (Qwen3-14B-AWQ).
# Disable Xet → fallback LFS classical, ổn định hơn.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Tăng retry + timeout cho mạng lag
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

# RTX PRO 6000 Blackwell = SM 12.0, FlashInfer build CUDA 12.8 chưa nhận diện được Blackwell.
# Force FlashAttention v2 backend (đã hoạt động OK trên log trước) + tắt FlashInfer sampler.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_FLASHINFER="${VLLM_USE_FLASHINFER:-0}"

exec "$ROOT/.venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B-AWQ \
    --quantization awq_marlin \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.50 \
    --enable-prefix-caching \
    --served-model-name translator \
    --port 9001 \
    --host 0.0.0.0
