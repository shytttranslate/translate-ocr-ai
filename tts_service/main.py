"""TTS service standalone — port 9004.

Chatterbox Multilingual 0.5B (ResembleAI, MIT). 23 ngôn ngữ — KHÔNG có Vietnamese.
Output: JSON + audio_base64 (WAV 24kHz mono 16-bit).
Voice: preset profile từ tts_service/voices/voices.json.
Preview UI: GET /preview/ → tools/tts_preview/index.html (single-page test tool).
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from config import get_settings
from engine import ChatterboxEngine
from routers import tts as tts_router
from utils.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("tts.lifespan")

    log.info(
        "tts_service_starting",
        env=settings.app_env,
        model=settings.tts_model_id,
        device=settings.tts_device,
        dtype=settings.tts_dtype,
    )

    engine = ChatterboxEngine(settings)
    await engine.initialize()
    app.state.engine = engine
    app.state.settings = settings

    log.info(
        "tts_service_ready",
        voices=len(engine.list_voices()),
        sample_rate=engine.sample_rate,
    )

    try:
        yield
    finally:
        log.info("tts_service_shutting_down")


app = FastAPI(
    title="TTS Service",
    version="0.1.0",
    description=(
        "Text-to-Speech với Chatterbox Multilingual 0.5B (ResembleAI, MIT). "
        "Hỗ trợ 23 ngôn ngữ — KHÔNG có Vietnamese. "
        "Output WAV 24kHz mono base64, có Perth watermark imperceptible."
    ),
    lifespan=lifespan,
    default_response_class=JSONResponse,
)

# CORS open — service public-facing, web client gọi trực tiếp.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


app.include_router(tts_router.router)


# Mount static preview UI tại /preview/ — single-page tool để test API qua browser.
# Path resolve linh hoạt: dù service start từ tts_service/ (supervisord) hay từ root (dev).
_preview_dir = Path(__file__).resolve().parent.parent / "tools" / "tts_preview"
if _preview_dir.is_dir():
    app.mount("/preview", StaticFiles(directory=str(_preview_dir), html=True), name="preview")


_settings = get_settings()
if _settings.enable_metrics:
    Instrumentator().instrument(app).expose(
        app, endpoint="/v1/metrics", include_in_schema=False,
    )


@app.get("/healthz/live", include_in_schema=False)
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz/ready")
async def readiness(request: Request, response: Response) -> dict[str, Any]:
    """Deep check: engine load xong + voice registry sẵn sàng."""
    engine: ChatterboxEngine = request.app.state.engine
    if not engine.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading", "service": "tts"}
    return {
        "status": "ok",
        "service": "tts",
        "voices": len(engine.list_voices()),
        "sample_rate": engine.sample_rate,
    }


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "tts-service",
        "version": "0.1.0",
        "docs": "/docs",
        "preview": "/preview/" if _preview_dir.is_dir() else "(not mounted)",
        "endpoints": "/v1/tts, /v1/voices, /v1/languages",
        "engine": "ChatterboxMultilingualTTS (ResembleAI, MIT)",
    }
