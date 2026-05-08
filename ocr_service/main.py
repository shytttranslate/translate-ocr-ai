"""OCR service standalone — FastAPI app riêng port 9003.

Tách khỏi API gateway để OCR không block translate/dict requests.
API gateway proxy /v1/ocr → service này qua HTTP.
"""
from __future__ import annotations

import base64
import binascii
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from engine import (
    AUTO_DEFAULT_LANG,
    SUPPORTED_LANGS,
    OcrEngine,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("ocr_service")


class OcrRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str = Field(min_length=1, description="Ảnh dạng base64")
    lang: str = Field(default="auto")
    request_id: str | None = Field(default=None, description="Trace từ API gateway")


class OcrTextBlock(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[list[int]]


class OcrResponse(BaseModel):
    request_id: str
    service: Literal["ocr"] = "ocr"
    processing_time_ms: int
    lang: str
    detected_lang: str
    image_width: int
    image_height: int
    full_text: str
    text_blocks: list[OcrTextBlock]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("ocr_service_starting")
    engine = OcrEngine()
    app.state.engine = engine
    # Warm-up `en` ngay startup → request đầu tiên không phải đợi 10-20s engine init.
    # Các lang khác load lazy khi có request.
    log.info("ocr_service_warmup_start")
    await engine.warm_up([AUTO_DEFAULT_LANG])
    log.info("ocr_service_ready")
    try:
        yield
    finally:
        log.info("ocr_service_shutdown")


app = FastAPI(
    title="VietByte OCR Service",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/healthz/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz/ready")
async def ready() -> dict[str, str]:
    return {"status": "ok", "service": "ocr"}


@app.post("/ocr", response_model=OcrResponse)
async def ocr_endpoint(req: OcrRequest) -> OcrResponse:
    request_id = req.request_id or str(uuid.uuid4())

    if req.lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"lang '{req.lang}' không hợp lệ. Cho phép: {sorted(SUPPORTED_LANGS)}",
        )

    try:
        image_bytes = base64.b64decode(req.image, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Base64 decode lỗi: {exc}",
        ) from exc

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Ảnh rỗng",
        )

    engine: OcrEngine = app.state.engine

    started = time.perf_counter()
    try:
        if req.lang == "auto":
            blocks, width, height, detected_lang = await engine.ocr_auto(image_bytes)
        else:
            blocks, width, height = await engine.ocr(image_bytes, req.lang)
            detected_lang = req.lang
    except Exception as exc:
        log.exception("ocr_failed request_id=%s lang=%s", request_id, req.lang)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OCR engine lỗi: {exc!s}"[:500],
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    full_text = "\n".join(b.text for b in blocks)

    log.info(
        "ocr_ok request_id=%s lang=%s detected=%s n_blocks=%d size=%dx%d elapsed=%dms",
        request_id, req.lang, detected_lang, len(blocks), width, height, elapsed_ms,
    )

    return OcrResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        lang=req.lang,
        detected_lang=detected_lang,
        image_width=width,
        image_height=height,
        full_text=full_text,
        text_blocks=[
            OcrTextBlock(text=b.text, confidence=b.confidence, bbox=b.bbox)
            for b in blocks
        ],
    )
