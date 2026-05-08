"""Endpoint /v1/ocr — proxy đến OCR service standalone (port 9003).

API gateway không nhúng PaddleOCR nữa. Forward request sang ocr_service qua HTTP
để OCR-heavy không block translate/dict.
"""
from __future__ import annotations

import time

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from config import get_settings
from models.schemas import OcrRequest, OcrResponse
from utils.logging import get_logger

router = APIRouter(tags=["ocr"])
log = get_logger(__name__)


@router.post("/v1/ocr", response_model=OcrResponse)
async def ocr(req: OcrRequest, request: Request) -> OcrResponse:
    """Proxy /v1/ocr → OCR service (port 9003).

    OCR service xử lý: validate lang, decode base64, run PaddleOCR, normalize bbox.
    Gateway chỉ forward + propagate request_id.
    """
    request_id = request.state.request_id
    settings = get_settings()
    client: httpx.AsyncClient = request.app.state.ocr_client

    started = time.perf_counter()
    try:
        resp = await client.post(
            f"{settings.ocr_service_url}/ocr",
            json={"image": req.image, "lang": req.lang, "request_id": request_id},
            timeout=settings.ocr_request_timeout_s,
        )
    except httpx.TimeoutException as exc:
        log.warning("ocr_proxy_timeout", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"OCR service timeout sau {settings.ocr_request_timeout_s}s",
        ) from exc
    except httpx.RequestError as exc:
        log.error("ocr_proxy_unreachable", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OCR service không reachable: {exc!s}"[:300],
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if resp.status_code != 200:
        # Forward status + body từ OCR service (giữ nguyên 400/502/...)
        try:
            detail = resp.json().get("detail", resp.text[:300])
        except Exception:  # noqa: BLE001
            detail = resp.text[:300]
        log.warning(
            "ocr_proxy_error",
            request_id=request_id,
            upstream_status=resp.status_code,
            detail=detail,
        )
        raise HTTPException(status_code=resp.status_code, detail=detail)

    log.info(
        "ocr_proxy_ok",
        request_id=request_id,
        lang=req.lang,
        proxy_overhead_ms=elapsed_ms,
    )
    return OcrResponse.model_validate(resp.json())
