"""Translate service standalone — port 9002.

Chỉ chứa /v1/translate, /v1/json, /v1/dict (cùng dùng vLLM translator).
OCR là service riêng port 9003 — KHÔNG đi qua đây.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from config import get_settings
from routers import dict as dict_router, translate
from services.vllm_client import VllmEndpoint, VllmRegistry
from utils.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("translate.lifespan")

    translator = VllmEndpoint(
        name="translator",
        base_url=settings.vllm_translator_url,
        served_model_name=settings.vllm_translator_model,
        connect_timeout_s=settings.vllm_connect_timeout_s,
        request_timeout_s=settings.vllm_request_timeout_s,
    )
    registry = VllmRegistry(translator=translator)
    await registry.start_all()

    app.state.settings = settings
    app.state.vllm_registry = registry

    log.info(
        "translate_service_started",
        env=settings.app_env,
        translator_url=settings.vllm_translator_url,
    )

    try:
        yield
    finally:
        log.info("translate_service_shutting_down")
        await registry.stop_all()


app = FastAPI(
    title="VietByte Translate Service",
    version="0.3.0",
    description="Translate (text + JSON batch) + Dictionary — vLLM Qwen3-14B-AWQ backend.",
    lifespan=lifespan,
    default_response_class=JSONResponse,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


app.include_router(translate.router)
app.include_router(dict_router.router)


_settings = get_settings()
if _settings.enable_metrics:
    Instrumentator().instrument(app).expose(app, endpoint="/v1/metrics", include_in_schema=False)


@app.get("/healthz/live", include_in_schema=False)
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz/ready")
async def readiness(request: Request, response: Response) -> dict[str, Any]:
    """Deep check: vLLM translator phản hồi inference call."""
    registry = request.app.state.vllm_registry
    try:
        health = await asyncio.wait_for(registry.deep_health_check(), timeout=4.0)
    except asyncio.TimeoutError:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "down", "error": "readiness check timeout"}

    if not health["translator"].get("ok"):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded", "components": {"vllm_translator": health["translator"]}}
    return {"status": "ok", "components": {"vllm_translator": health["translator"]}}


@app.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    registry = request.app.state.vllm_registry
    fingerprint = await registry.translator.get_model_fingerprint()
    return {
        "service": "translate",
        "translator": {
            "served_name": registry.translator.served_model_name,
            "fingerprint": fingerprint,
            "url": registry.translator.base_url,
        },
    }


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "vietbyte-translate-service",
        "version": "0.3.0",
        "docs": "/docs",
        "endpoints": "/v1/translate, /v1/json, /v1/dict",
    }
