# API gateway image — chạy FastAPI, không cần GPU
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Cài system deps cho opencv-python-headless và Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Tạo user non-root theo nguyên tắc least privilege (#18 Bảo mật)
RUN useradd -m -u 1000 -s /bin/bash apiuser

COPY --chown=apiuser:apiuser . /app/

USER apiuser

EXPOSE 8000

# Healthcheck: dùng /healthz/live (lightweight, không gọi vLLM)
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz/live || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop", "--http", "httptools", "--access-log"]
