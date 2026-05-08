"""Endpoint meta: health, metrics, models.

Health check 3 tầng theo phản biện #27 SRE:
- /healthz/live: chỉ check process alive (cho liveness probe)
- /healthz/ready: deep check vLLM + Redis (cho readiness probe, LB drain)
- /healthz/startup: check model đã load chưa (cho startup probe, container orchestration)

Endpoint /v1/health là alias public-friendly cho ready check.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request, Response, status

from models.schemas import HealthStatus

router = APIRouter(tags=["meta"])


@router.get("/healthz/live", response_model=HealthStatus)
async def liveness() -> HealthStatus:
    """Liveness probe: trả ok nếu event loop còn xử lý request."""
    return HealthStatus(status="ok", components={"event_loop": "ok"})


@router.get("/healthz/ready")
async def readiness(request: Request, response: Response) -> dict[str, Any]:
    """Readiness probe: deep check toàn bộ dependency.

    Trả 503 nếu bất kỳ dependency nào down → load balancer drain instance.
    """
    registry = request.app.state.vllm_registry
    cache = request.app.state.cache

    deep_check_task = asyncio.create_task(registry.deep_health_check())
    redis_task = asyncio.create_task(cache.ping())

    try:
        vllm_health, redis_ok = await asyncio.wait_for(
            asyncio.gather(deep_check_task, redis_task),
            timeout=3.5,
        )
    except asyncio.TimeoutError:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "down", "error": "readiness check timeout"}

    components = {
        "vllm_translator": vllm_health["translator"],
        "vllm_ocr": vllm_health["ocr"],
        "redis": {"ok": redis_ok},
    }

    all_ok = (
        vllm_health["translator"].get("ok")
        and vllm_health["ocr"].get("ok")
        and redis_ok
    )
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded", "components": components}

    return {"status": "ok", "components": components}


@router.get("/healthz/startup")
async def startup_probe(request: Request, response: Response) -> dict[str, Any]:
    """Startup probe: chỉ check model đã load (không gọi inference)."""
    registry = request.app.state.vllm_registry
    try:
        translator_models, ocr_models = await asyncio.gather(
            registry.translator.list_models(),
            registry.ocr.list_models(),
        )
    except Exception as exc:  # noqa: BLE001 — startup probe phải bắt mọi lỗi
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading", "error": str(exc)[:200]}

    return {
        "status": "ok",
        "translator_models": [m.get("id") for m in translator_models.get("data", [])],
        "ocr_models": [m.get("id") for m in ocr_models.get("data", [])],
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
    translator_fp, ocr_fp = await asyncio.gather(
        registry.translator.get_model_fingerprint(),
        registry.ocr.get_model_fingerprint(),
    )
    return {
        "translator": {
            "served_name": registry.translator.served_model_name,
            "fingerprint": translator_fp,
            "url": registry.translator.base_url,
        },
        "ocr": {
            "served_name": registry.ocr.served_model_name,
            "fingerprint": ocr_fp,
            "url": registry.ocr.base_url,
        },
    }
