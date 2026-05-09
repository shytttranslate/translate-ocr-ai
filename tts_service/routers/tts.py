"""TTS endpoints — synthesize, voices, languages."""
from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException, Request, status

from config import get_settings
from engine import ChatterboxEngine
from models.schemas import (
    LANGUAGE_LABELS,
    SUPPORTED_LANGUAGES,
    LanguageEntry,
    LanguagesResponse,
    TTSRequest,
    TTSResponse,
    VoicesResponse,
)
from services.audio import tensor_to_wav_base64
from utils.logging import get_logger

router = APIRouter(tags=["tts"])
log = get_logger("tts.router")


@router.post("/v1/tts", response_model=TTSResponse)
async def synthesize(req: TTSRequest, request: Request) -> TTSResponse:
    request_id = (
        req.request_id
        or getattr(request.state, "request_id", None)
        or str(uuid.uuid4())
    )
    engine: ChatterboxEngine = request.app.state.engine

    if not engine.has_voice(req.voice_id):
        available = sorted(engine.list_voice_ids())
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"voice_id '{req.voice_id}' không tồn tại. Có sẵn: {available}",
        )

    started = time.perf_counter()
    try:
        wav, duration_ms, chunk_count = await engine.synthesize(
            text=req.text,
            language_id=req.language_id,
            voice_id=req.voice_id,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
            temperature=req.temperature,
            seed=req.seed,
        )
    except asyncio.TimeoutError:
        log.warning("tts_timeout", request_id=request_id, text=req.text)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="TTS inference timeout",
        )
    except KeyError as exc:
        # Race: voice bị unload sau check has_voice (không xảy ra ở v1).
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"voice_id '{exc.args[0]}' không tồn tại",
        ) from exc
    except Exception as exc:
        log.exception("tts_failed", request_id=request_id, language=req.language_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"TTS engine lỗi: {exc!s}"[:500],
        ) from exc

    audio_b64 = tensor_to_wav_base64(wav, engine.sample_rate)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "tts_ok",
        request_id=request_id,
        language=req.language_id,
        voice=req.voice_id,
        text=req.text,
        chunks=chunk_count,
        duration_ms=duration_ms,
        elapsed_ms=elapsed_ms,
        audio_b64_len=len(audio_b64),
    )
    return TTSResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        audio_base64=audio_b64,
        duration_ms=duration_ms,
        voice_id=req.voice_id,
        language_id=req.language_id,
        chunk_count=chunk_count,
        seed=req.seed,
    )


@router.get("/v1/voices", response_model=VoicesResponse)
async def list_voices(request: Request) -> VoicesResponse:
    engine: ChatterboxEngine = request.app.state.engine
    voices = engine.list_voices()
    return VoicesResponse(
        count=len(voices),
        default_voice_id=get_settings().tts_default_voice_id,
        voices=voices,
    )


@router.get("/v1/languages", response_model=LanguagesResponse)
async def list_languages() -> LanguagesResponse:
    return LanguagesResponse(
        count=len(SUPPORTED_LANGUAGES),
        languages=[
            LanguageEntry(code=code, label=LANGUAGE_LABELS[code])
            for code in SUPPORTED_LANGUAGES
        ],
    )
