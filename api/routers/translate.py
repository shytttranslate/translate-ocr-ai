"""Endpoint /v1/translate (text đơn/batch) và /v1/json (array of strings)."""
from __future__ import annotations

import time
from collections import Counter

from fastapi import APIRouter, HTTPException, Request, status

from models.schemas import (
    JsonTranslateRequest,
    JsonTranslateResponse,
    TranslateRequest,
    TranslateResponse,
    TranslationItem,
)
from services.translator import translate_batch, translate_one
from utils.logging import get_logger

router = APIRouter(tags=["translate"])
log = get_logger(__name__)


@router.post("/v1/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest, request: Request) -> TranslateResponse:
    """Dịch text đơn hoặc batch (list of strings).

    - source_lang=auto: model tự detect, trả `detected_source_lang` trong response.
    - source_lang=<code>: bypass detect, dịch trực tiếp.
    """
    request_id = request.state.request_id
    registry = request.app.state.vllm_registry
    endpoint = registry.translator
    fingerprint = await endpoint.get_model_fingerprint()

    started = time.perf_counter()

    texts = [req.text] if isinstance(req.text, str) else req.text

    try:
        if len(texts) == 1:
            result = await translate_one(
                endpoint=endpoint,
                text=texts[0],
                source_lang=req.source_lang,
                target_lang=req.target_lang,
            )
            results = [result]
        else:
            results = await translate_batch(
                endpoint=endpoint,
                texts=texts,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
            )
    except Exception as exc:
        log.error(
            "translate_failed",
            request_id=request_id,
            error=str(exc),
            batch_size=len(texts),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Translation upstream lỗi: {exc!s}"[:500],
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "translate_ok",
        request_id=request_id,
        batch_size=len(texts),
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        elapsed_ms=elapsed_ms,
    )

    return TranslateResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        model_used=fingerprint,
        translations=[
            TranslationItem(
                source_text=r.source_text,
                translated_text=r.translated_text,
                detected_source_lang=r.detected_source_lang,
                target_lang=req.target_lang,
            )
            for r in results
        ],
    )


@router.post("/v1/json", response_model=JsonTranslateResponse)
async def translate_json(
    req: JsonTranslateRequest, request: Request
) -> JsonTranslateResponse:
    """Dịch array of strings, trả lại array string cùng thứ tự.

    Dùng cho i18n batch hoặc khi client cần shape đơn giản nhất.
    """
    request_id = request.state.request_id
    registry = request.app.state.vllm_registry
    endpoint = registry.translator
    fingerprint = await endpoint.get_model_fingerprint()

    started = time.perf_counter()

    try:
        results = await translate_batch(
            endpoint=endpoint,
            texts=req.texts,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
        )
    except Exception as exc:
        log.error(
            "translate_json_failed",
            request_id=request_id,
            error=str(exc),
            batch_size=len(req.texts),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Translation upstream lỗi: {exc!s}"[:500],
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Detected language dominant của batch (mode)
    if req.source_lang == "auto":
        counter = Counter(r.detected_source_lang for r in results)
        detected = counter.most_common(1)[0][0] if counter else "unknown"
    else:
        detected = req.source_lang

    log.info(
        "translate_json_ok",
        request_id=request_id,
        batch_size=len(req.texts),
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        detected=detected,
        elapsed_ms=elapsed_ms,
    )

    return JsonTranslateResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        model_used=fingerprint,
        translations=[r.translated_text for r in results],
        detected_source_lang=detected,
        target_lang=req.target_lang,
    )
