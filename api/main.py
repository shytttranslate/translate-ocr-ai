"""FastAPI entry point cho VietByte AI API: 1 vLLM translator + PaddleOCR."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from config import get_settings
from routers import dict as dict_router, meta, ocr as ocr_router, translate
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
    registry = VllmRegistry(translator=translator)
    await registry.start_all()

    # OCR service riêng port 9003 — proxy qua HTTP, không nhúng PaddleOCR vào gateway.
    ocr_client = httpx.AsyncClient(
        base_url=settings.ocr_service_url,
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )

    app.state.settings = settings
    app.state.vllm_registry = registry
    app.state.ocr_client = ocr_client

    log.info(
        "api_started",
        env=settings.app_env,
        translator_url=settings.vllm_translator_url,
        ocr_service_url=settings.ocr_service_url,
    )

    try:
        yield
    finally:
        log.info("api_shutting_down")
        await registry.stop_all()
        await ocr_client.aclose()


app = FastAPI(
    title="VietByte AI API",
    version="0.2.0",
    description="Translation (Qwen3-14B-AWQ) + OCR (PaddleOCR v5).",
    lifespan=lifespan,
    default_response_class=JSONResponse,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Inject request_id để cross-component tracing."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


app.include_router(meta.router)
app.include_router(translate.router)
app.include_router(dict_router.router)
app.include_router(ocr_router.router)


_settings = get_settings()
if _settings.enable_metrics:
    Instrumentator().instrument(app).expose(app, endpoint="/v1/metrics", include_in_schema=False)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "vietbyte-ai-api",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/v1/health",
    }
