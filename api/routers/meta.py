"""Endpoint meta: health, models.

Health check 3 tầng:
- /healthz/live: chỉ check process alive
- /healthz/ready: deep check vLLM translator (gọi inference thật)
- /healthz/startup: model đã load chưa
- /v1/health: alias public của readiness
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request, Response, status

from models.schemas import HealthStatus

router = APIRouter(tags=["meta"])


@router.get("/healthz/live", response_model=HealthStatus)
async def liveness() -> HealthStatus:
    """Liveness probe: ok nếu event loop xử lý được request."""
    return HealthStatus(status="ok", components={"event_loop": "ok"})


@router.get("/healthz/ready")
async def readiness(request: Request, response: Response) -> dict[str, Any]:
    """Readiness probe: deep check vLLM translator + OCR service."""
    registry = request.app.state.vllm_registry
    ocr_client = request.app.state.ocr_client

    async def _check_ocr() -> dict[str, Any]:
        try:
            r = await ocr_client.get("/healthz/live", timeout=2.0)
            return {"ok": r.status_code == 200, "status_code": r.status_code}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:120]}

    try:
        vllm_health, ocr_health = await asyncio.wait_for(
            asyncio.gather(registry.deep_health_check(), _check_ocr()),
            timeout=4.0,
        )
    except asyncio.TimeoutError:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "down", "error": "readiness check timeout"}

    components = {
        "vllm_translator": vllm_health["translator"],
        "ocr_service": ocr_health,
    }
    all_ok = vllm_health["translator"].get("ok") and ocr_health.get("ok")
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded", "components": components}

    return {"status": "ok", "components": components}


@router.get("/healthz/startup")
async def startup_probe(request: Request, response: Response) -> dict[str, Any]:
    """Startup probe: model đã load chưa (không gọi inference)."""
    registry = request.app.state.vllm_registry
    try:
        translator_models = await registry.translator.list_models()
    except Exception as exc:  # noqa: BLE001 — startup probe phải bắt mọi lỗi
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading", "error": str(exc)[:200]}

    return {
        "status": "ok",
        "translator_models": [m.get("id") for m in translator_models.get("data", [])],
    }


@router.get("/v1/health", response_model=HealthStatus)
async def public_health(request: Request, response: Response) -> HealthStatus:
    """Health public: alias gọn cho readiness."""
    deep = await readiness(request, response)
    components = deep.get("components", {})
    if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return HealthStatus(status="degraded", components=components)
    return HealthStatus(status="ok", components=components)


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    registry = request.app.state.vllm_registry
    translator_fp = await registry.translator.get_model_fingerprint()
    return {
        "translator": {
            "served_name": registry.translator.served_model_name,
            "fingerprint": translator_fp,
            "url": registry.translator.base_url,
        },
        "ocr": {
            "engine": "PaddleOCR v5 (PP-OCRv5)",
            "mode": "CPU",
        },
    }
