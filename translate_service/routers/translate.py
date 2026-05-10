"""Endpoint /v1/translate (text đơn/batch) và /v1/json (array of strings)."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, status

from models.schemas import (
    HtmlHealthInfo,
    JsonTranslateRequest,
    JsonTranslateResponse,
    JsonTranslationStats,
    TranslateHtmlRequest,
    TranslateHtmlResponse,
    TranslateJsonObjectRequest,
    TranslateJsonObjectResponse,
    TranslateRequest,
    TranslateResponse,
    TranslationDetected,
)
from services.html_translator import HtmlTooMalformed, translate_html
from services.json_translator import (
    JsonTooLarge,
    parse_excluded_paths,
    parse_string_or_list,
)
from services.json_translator import translate_json as _translate_json_object
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


@router.post("/v1/translate-html", response_model=TranslateHtmlResponse)
async def translate_html_endpoint(
    req: TranslateHtmlRequest, request: Request
) -> TranslateHtmlResponse:
    """Translate HTML preserving structure.

    Pipeline: parse (lxml + html5lib fallback) → score health → DOM walk →
    inline placeholder → batch translate → restore → reinsert → verify.

    Health levels:
    - `clean` / `minor` / `moderate` → 200 OK với warnings nếu có
    - `severe` → 422 với errors detail (HTML cần caller fix trước)

    `ignore_terms` (vd: tên thương hiệu) — match exact word boundary, giữ nguyên.
    """
    request_id = request.state.request_id
    endpoint = request.app.state.vllm_registry.translator

    started = time.perf_counter()

    try:
        result = await translate_html(
            raw_html=req.html,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            endpoint=endpoint,
            ignore_terms=req.ignore_terms,
            ignore_case=req.ignore_case,
            translate_attributes=req.translate_attributes,
        )
    except HtmlTooMalformed as exc:
        log.warning(
            "translate_html_rejected",
            request_id=request_id,
            health=exc.health.health,
            fatal_markers=exc.health.fatal_markers,
            error_rate=exc.health.error_rate,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "html_too_malformed",
                "health": exc.health.health,
                "metrics": {
                    "error_rate": exc.health.error_rate,
                    "structure_diff": exc.health.structure_diff,
                    "errors_total": exc.health.errors_total,
                    "fatals_total": exc.health.fatals_total,
                },
                "errors_sample": exc.health.sample_errors,
                "fatal_markers": exc.health.fatal_markers,
                "suggestion": (
                    "Run HTML through a sanitizer (e.g. `tidy -q -m -ashtml` or "
                    "`html-minifier-terser`) before retry."
                ),
            },
        ) from exc
    except Exception as exc:
        log.error(
            "translate_html_failed",
            request_id=request_id,
            error=str(exc),
            html_len=len(req.html),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"HTML translation lỗi: {exc!s}"[:500],
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "translate_html_ok",
        request_id=request_id,
        html_len=len(req.html),
        segments=result.segments_translated,
        chars=result.chars_translated,
        health=result.health.health,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        elapsed_ms=elapsed_ms,
        warnings_n=len(result.warnings),
    )

    return TranslateHtmlResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        html=result.html,
        detected_source_lang=result.detected_source_lang,
        health=HtmlHealthInfo(
            health=result.health.health,  # type: ignore[arg-type]
            error_rate=result.health.error_rate,
            structure_diff=result.health.structure_diff,
            errors_total=result.health.errors_total,
            fatals_total=result.health.fatals_total,
            parse_tier=result.health.parse_tier,
        ),
        segments_translated=result.segments_translated,
        chars_translated=result.chars_translated,
        warnings=result.warnings,
    )


@router.post("/v1/translate-json", response_model=TranslateJsonObjectResponse)
async def translate_json_object_endpoint(
    req: TranslateJsonObjectRequest, request: Request
) -> TranslateJsonObjectResponse:
    """Translate JSON object/array recursively, preserve structure.

    3 exclusion options accept cả string (separator `;`) và list[str]:

    - `words_not_to_translate`: từ/cụm giữ nguyên trong text (vd brand names).
      Match word boundary, default case-sensitive.
    - `paths_to_exclude`: dot-notation path skip toàn subtree (vd `product.media.img_desc`).
      Wildcard `*` cho array index (vd `items.*.image_url`).
    - `common_keys_to_exclude`: tên key skip ở any depth (vd `name; price`).

    Auto-skip filter (skip_non_text=true): numbers, currency, percent, URLs, emails,
    UUIDs, hash, ISO dates, code/ID patterns. Set false để dịch hết.

    Limits: max 2000 translatable strings, max nesting depth 50.
    """
    request_id = request.state.request_id
    endpoint = request.app.state.vllm_registry.translator

    started = time.perf_counter()

    words = parse_string_or_list(req.words_not_to_translate)
    paths = parse_excluded_paths(req.paths_to_exclude)
    keys = parse_string_or_list(req.common_keys_to_exclude)

    try:
        result = await _translate_json_object(
            json_data=req.json_data,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            endpoint=endpoint,
            words_not_to_translate=words,
            paths_to_exclude=paths,
            common_keys_to_exclude=keys,
            ignore_case=req.ignore_case,
            skip_non_text=req.skip_non_text,
        )
    except JsonTooLarge as exc:
        log.warning(
            "translate_json_object_too_large",
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": "json_too_large", "message": str(exc)},
        ) from exc
    except Exception as exc:
        log.error(
            "translate_json_object_failed",
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"JSON translation lỗi: {exc!s}"[:500],
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "translate_json_object_ok",
        request_id=request_id,
        strings_translated=result.strings_translated,
        strings_skipped=result.strings_skipped,
        chars=result.chars_translated,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        elapsed_ms=elapsed_ms,
    )

    return TranslateJsonObjectResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        json_data=result.json_data,
        detected_source_lang=result.detected_source_lang,
        stats=JsonTranslationStats(
            strings_translated=result.strings_translated,
            strings_skipped=result.strings_skipped,
            chars_translated=result.chars_translated,
        ),
        warnings=result.warnings,
    )
