"""Endpoint /v1/translate (text đơn/batch) và /v1/json (array of strings)."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, status

from models.schemas import (
    JsonTranslateRequest,
    JsonTranslateResponse,
    TranslateRequest,
    TranslateResponse,
    TranslationDetected,
)
from services.translator import TranslationResult, translate_batch, translate_one
from utils.logging import get_logger

router = APIRouter(tags=["translate"])
log = get_logger(__name__)


def _build_translations(
    results: list[TranslationResult], source_lang: str
) -> list[TranslationDetected] | list[str]:
    """source_lang=auto → list object kèm detected_source_lang. Explicit → list[str]."""
    if source_lang == "auto":
        return [
            TranslationDetected(
                translated_text=r.translated_text,
                detected_source_lang=r.detected_source_lang,
            )
            for r in results
        ]
    return [r.translated_text for r in results]


@router.post("/v1/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest, request: Request) -> TranslateResponse:
    """Dịch text đơn hoặc batch.

    Response `translations` theo thứ tự input:
    - source_lang=auto → list[{translated_text, detected_source_lang}]
    - explicit lang   → list[str]
    """
    request_id = request.state.request_id
    endpoint = request.app.state.vllm_registry.translator

    started = time.perf_counter()
    texts = [req.text] if isinstance(req.text, str) else req.text

    try:
        if len(texts) == 1:
            results = [
                await translate_one(
                    endpoint=endpoint,
                    text=texts[0],
                    source_lang=req.source_lang,
                    target_lang=req.target_lang,
                )
            ]
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
        translations=_build_translations(results, req.source_lang),
    )


@router.post("/v1/json", response_model=JsonTranslateResponse)
async def translate_json(
    req: JsonTranslateRequest, request: Request
) -> JsonTranslateResponse:
    """Dịch array of strings.

    Response `translations` theo thứ tự input:
    - source_lang=auto → list[{translated_text, detected_source_lang}]
    - explicit lang   → list[str]
    """
    request_id = request.state.request_id
    endpoint = request.app.state.vllm_registry.translator

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
    log.info(
        "translate_json_ok",
        request_id=request_id,
        batch_size=len(req.texts),
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        elapsed_ms=elapsed_ms,
    )

    return JsonTranslateResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        translations=_build_translations(results, req.source_lang),
    )
