"""Endpoint /v1/dict — tra từ điển đa ngôn ngữ.

Output rich JSON (camelCase): word, phonetic (ipa+romanization), shortMeaning,
definitions, examples, phrases, related (synonyms/antonyms/relatedWords/memoryTips).
"""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, status

from models.schemas import (
    DictDefinition,
    DictExample,
    DictPhonetic,
    DictRelated,
    DictRequest,
    DictResponse,
    DictWordRef,
)
from services.dictionary import dict_lookup
from utils.logging import get_logger

router = APIRouter(tags=["dictionary"])
log = get_logger(__name__)


@router.post("/v1/dict", response_model=DictResponse, response_model_by_alias=True)
async def lookup_word(req: DictRequest, request: Request) -> DictResponse:
    """Tra từ điển: nhập word ở native_lang, nhận entry chi tiết với meaning ở target_lang."""
    request_id = request.state.request_id
    registry = request.app.state.vllm_registry
    endpoint = registry.translator
    fingerprint = await endpoint.get_model_fingerprint()

    started = time.perf_counter()
    try:
        result = await dict_lookup(
            endpoint=endpoint,
            word=req.word,
            native_lang=req.native_lang,
            target_lang=req.target_lang,
        )
    except Exception as exc:
        log.error(
            "dict_failed",
            request_id=request_id,
            word=req.word,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Dictionary upstream lỗi: {exc!s}"[:500],
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if not result.definitions:
        log.warning(
            "dict_empty_result",
            request_id=request_id,
            word=req.word,
            native_lang=req.native_lang,
            target_lang=req.target_lang,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy entry cho '{req.word}' ({req.native_lang} → {req.target_lang})",
        )

    log.info(
        "dict_ok",
        request_id=request_id,
        word=req.word,
        native_lang=req.native_lang,
        target_lang=req.target_lang,
        n_definitions=len(result.definitions),
        n_examples=len(result.examples),
        n_phrases=len(result.phrases),
        elapsed_ms=elapsed_ms,
    )

    return DictResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        model_used=fingerprint,
        word=result.word or req.word.strip(),
        native_lang=req.native_lang,
        target_lang=req.target_lang,
        phonetic=DictPhonetic(
            ipa=result.phonetic_ipa,
            romanization=result.phonetic_romanization,
        ),
        short_meaning=result.short_meaning,
        definitions=[DictDefinition(**d) for d in result.definitions],
        examples=[DictExample(**e) for e in result.examples],
        phrases=[DictExample(**p) for p in result.phrases],
        related=DictRelated(
            synonyms=[DictWordRef(**s) for s in result.synonyms],
            antonyms=[DictWordRef(**a) for a in result.antonyms],
            related_words=[DictWordRef(**r) for r in result.related_words],
            memory_tips=result.memory_tips,
        ),
    )
