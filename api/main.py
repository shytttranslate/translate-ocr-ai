"""FastAPI entry point cho VietByte OCR + Translate API.

Phase 1: foundation only — health check, model registry, skeleton router.
Phase 2-3 sẽ thêm /v1/translate và /v1/ocr.
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from config import get_settings
from routers import meta
from services.auth import seed_dev_key
from services.cache import RedisCache
from services.vllm_client import VllmEndpoint, VllmRegistry
from utils.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("api.lifespan")

    translator = VllmEndpoint(
        name="translator",
        base_url=settings.vllm_translator_url,
        served_model_name=settings.vllm_translator_model,
        connect_timeout_s=settings.vllm_connect_timeout_s,
        request_timeout_s=settings.vllm_request_timeout_s,
    )
    ocr = VllmEndpoint(
        name="ocr",
        base_url=settings.vllm_ocr_url,
        served_model_name=settings.vllm_ocr_model,
        connect_timeout_s=settings.vllm_connect_timeout_s,
        request_timeout_s=settings.vllm_request_timeout_s,
    )
    registry = VllmRegistry(translator=translator, ocr=ocr)
    await registry.start_all()

    cache = RedisCache(settings)

    app.state.settings = settings
    app.state.vllm_registry = registry
    app.state.cache = cache

    if not settings.is_prod:
        dev_key = os.getenv("DEV_API_KEY", "vbk_live_dev_seed_key_for_local_only_xyz")
        seed_dev_key(settings, dev_key, tier="enterprise")
        log.info("dev_api_key_seeded", api_key=dev_key)

    log.info(
        "api_started",
        env=settings.app_env,
        translator_url=settings.vllm_translator_url,
        ocr_url=settings.vllm_ocr_url,
    )

    try:
        yield
    finally:
        log.info("api_shutting_down")
        await registry.stop_all()
        await cache.close()


app = FastAPI(
    title="VietByte AI API",
    version="0.1.0",
    description="Unified OCR + Translation API. Phase 1: foundation.",
    lifespan=lifespan,
    default_response_class=JSONResponse,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Inject request_id vào mỗi request để cross-component tracing."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


app.include_router(meta.router)


_settings = get_settings()
if _settings.enable_metrics:
    Instrumentator().instrument(app).expose(app, endpoint="/v1/metrics", include_in_schema=False)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "vietbyte-ai-api",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/v1/health",
    }
