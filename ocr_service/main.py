"""OCR service standalone — FastAPI app riêng port 9003.

Tách khỏi API gateway để OCR không block translate/dict requests.
API gateway proxy /v1/ocr → service này qua HTTP.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine import (
    AUTO_DEFAULT_LANG,
    SUPPORTED_LANGS,
    OcrEngine,
)
from paragraph_merger import (
    detect_reading_order,
    merge_blocks_into_paragraphs,
    paragraphs_to_full_text,
)


# Limit ảnh fetch từ URL — tránh DOS bằng URL trỏ tới file lớn.
URL_FETCH_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
URL_FETCH_TIMEOUT_S = 15.0

# LRU cache cho OCR result theo SHA256(image_bytes + lang). 256 entry × ~5KB response = ~1.3MB.
CACHE_MAX_ENTRIES = int(os.environ.get("PADDLEOCR_CACHE_SIZE", "256"))
_result_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(image_bytes: bytes, lang: str) -> str:
    h = hashlib.sha256()
    h.update(lang.encode("ascii"))
    h.update(b":")
    h.update(image_bytes)
    return h.hexdigest()


def _cache_get(key: str) -> dict | None:
    with _cache_lock:
        if key in _result_cache:
            _result_cache.move_to_end(key)
            return _result_cache[key]
        return None


def _cache_put(key: str, value: dict) -> None:
    with _cache_lock:
        _result_cache[key] = value
        _result_cache.move_to_end(key)
        while len(_result_cache) > CACHE_MAX_ENTRIES:
            _result_cache.popitem(last=False)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("ocr_service")


class OcrRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str | None = Field(default=None, description="Ảnh dạng base64 (PNG/JPG/WebP)")
    image_url: str | None = Field(
        default=None,
        description="URL ảnh — server tự fetch + decode. Alternative cho `image`. Max 10MB.",
    )
    lang: str = Field(default="auto")
    level: Literal["block", "word"] = Field(
        default="block",
        description="block = bbox theo dòng text (mặc định). word = bbox theo từng từ.",
    )
    merge_paragraphs: bool = Field(
        default=True,
        description=(
            "Bật heuristic gộp các text-line liền kề thành paragraph. "
            "True = trả thêm field `paragraphs` (gộp speech bubble / khối văn). "
            "False = chỉ trả raw `text_blocks` line-level."
        ),
    )
    reading_order: Literal["ltr", "rtl", "auto"] = Field(
        default="auto",
        description=(
            "Thứ tự đọc khi sort paragraph: ltr = trái→phải (default), "
            "rtl = phải→trái (manga JP, Arabic), auto = tự detect theo aspect ratio + CJK ratio."
        ),
    )
    mode: Literal["general", "manga"] = Field(
        default="general",
        description=(
            "general (default) = PaddleOCR pipeline cho document/screenshot.\n"
            "manga = specialized pipeline cho comic/manga JP: PaddleOCR detection + "
            "manga-ocr recognition (kha-white) + bubble clustering + RTL reading order. "
            "Khi mode=manga, lang/level/reading_order được override (lang=japan, RTL)."
        ),
    )
    request_id: str | None = Field(default=None, description="Trace từ API gateway")

    @model_validator(mode="after")
    def _exactly_one_input(self) -> "OcrRequest":
        if not self.image and not self.image_url:
            raise ValueError("Phải truyền 1 trong 2: `image` (base64) hoặc `image_url`")
        if self.image and self.image_url:
            raise ValueError("Chỉ truyền 1 trong 2: `image` HOẶC `image_url`, không cả hai")
        return self


class OcrWord(BaseModel):
    text: str
    bbox: list[list[int]]


class OcrTextBlock(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[list[int]]
    words: list[OcrWord] | None = None


class OcrParagraphPayload(BaseModel):
    text: str = Field(description="Text gộp từ các line trong paragraph")
    bbox: list[list[int]] = Field(
        description="Axis-aligned bbox bao quanh paragraph, format 4 góc [TL,TR,BR,BL]",
    )
    block_indices: list[int] = Field(
        description="Index (vị trí trong `text_blocks`) của các line gộp vào paragraph này",
    )
    avg_confidence: float = Field(ge=0.0, le=1.0)
    line_count: int = Field(ge=1, description="Số line gốc gộp lại")


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
    paragraphs: list[OcrParagraphPayload] | None = Field(
        default=None,
        description=(
            "Chỉ có khi request bật `merge_paragraphs=true`. Mỗi phần tử là 1 khối văn / "
            "speech bubble đã gộp từ nhiều line."
        ),
    )
    reading_order: Literal["ltr", "rtl"] | None = Field(
        default=None,
        description="Reading order thực tế dùng để sort paragraph (sau khi resolve 'auto').",
    )


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


async def _fetch_image_url(url: str) -> bytes:
    """Fetch ảnh từ URL — guard kích thước + scheme."""
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image_url phải bắt đầu bằng http:// hoặc https://",
        )
    # User-Agent thật vì 1 số CDN (Wikimedia, Cloudflare) chặn UA mặc định httpx.
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; OCR-Service/0.3; +https://ocr.spacecloud.fit)",
        "Accept": "image/*,*/*;q=0.8",
    }
    try:
        async with httpx.AsyncClient(timeout=URL_FETCH_TIMEOUT_S, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Fetch image_url lỗi: {exc!s}"[:300],
        ) from exc
    if len(data) > URL_FETCH_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Ảnh từ URL {len(data)} bytes vượt giới hạn {URL_FETCH_MAX_BYTES}",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image_url trả về body rỗng",
        )
    return data


app = FastAPI(
    title="OCR Service",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS cho tool preview (web standalone) — open vì đây là OCR service public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz/ready")
async def ready() -> dict[str, str]:
    return {"status": "ok", "service": "ocr"}


# Metadata cho /v1/languages — group theo recognition model.
# Source: https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
LANGUAGE_GROUPS: list[dict[str, object]] = [
    {
        "model": "PP-OCRv5_mobile_rec / server_rec",
        "label": "Default (CJK + English)",
        "languages": [
            {"code": "ch", "label": "Chinese Simplified + English"},
            {"code": "chinese_cht", "label": "Chinese Traditional (Đài Loan / Hong Kong)"},
            {"code": "japan", "label": "Japanese (Hiragana + Katakana + Kanji)"},
        ],
    },
    {
        "model": "en_PP-OCRv5_mobile_rec",
        "label": "English thuần (KHÔNG cover dấu Latin extended)",
        "languages": [
            {"code": "en", "label": "English"},
        ],
    },
    {
        "model": "korean_PP-OCRv5_mobile_rec",
        "label": "Korean",
        "languages": [
            {"code": "korean", "label": "Korean (Hangul)"},
        ],
    },
    {
        "model": "latin_PP-OCRv5_mobile_rec",
        "label": "Latin extended — 47 lang gồm Việt với đầy đủ diacritics",
        "languages": [
            {"code": "vi", "label": "Vietnamese"},
            {"code": "fr", "label": "French"},
            {"code": "de", "label": "German"},
            {"code": "es", "label": "Spanish"},
            {"code": "it", "label": "Italian"},
            {"code": "pt", "label": "Portuguese"},
            {"code": "nl", "label": "Dutch"},
            {"code": "pl", "label": "Polish"},
            {"code": "ro", "label": "Romanian"},
            {"code": "tr", "label": "Turkish"},
            {"code": "id", "label": "Indonesian"},
            {"code": "ms", "label": "Malay"},
            {"code": "tl", "label": "Tagalog"},
            {"code": "sv", "label": "Swedish"},
            {"code": "no", "label": "Norwegian"},
            {"code": "da", "label": "Danish"},
            {"code": "fi", "label": "Finnish"},
            {"code": "is", "label": "Icelandic"},
            {"code": "et", "label": "Estonian"},
            {"code": "lv", "label": "Latvian"},
            {"code": "lt", "label": "Lithuanian"},
            {"code": "cs", "label": "Czech"},
            {"code": "sk", "label": "Slovak"},
            {"code": "sl", "label": "Slovenian"},
            {"code": "hr", "label": "Croatian"},
            {"code": "bs", "label": "Bosnian"},
            {"code": "rs_latin", "label": "Serbian (Latin)"},
            {"code": "sq", "label": "Albanian"},
            {"code": "mt", "label": "Maltese"},
            {"code": "hu", "label": "Hungarian"},
            {"code": "ga", "label": "Irish"},
            {"code": "cy", "label": "Welsh"},
            {"code": "mi", "label": "Maori"},
            {"code": "sw", "label": "Swahili"},
            {"code": "af", "label": "Afrikaans"},
            {"code": "az", "label": "Azerbaijani"},
            {"code": "uz", "label": "Uzbek"},
            {"code": "ku", "label": "Kurdish (Latin script)"},
            {"code": "eu", "label": "Basque"},
            {"code": "ca", "label": "Catalan"},
            {"code": "gl", "label": "Galician"},
            {"code": "lb", "label": "Luxembourgish"},
            {"code": "rm", "label": "Romansh"},
            {"code": "oc", "label": "Occitan"},
            {"code": "qu", "label": "Quechua"},
            {"code": "la", "label": "Latin"},
            {"code": "pi", "label": "Pali"},
        ],
    },
    {
        "model": "eslav_PP-OCRv5_mobile_rec",
        "label": "East Slavic",
        "languages": [
            {"code": "ru", "label": "Russian"},
            {"code": "be", "label": "Belarusian"},
            {"code": "uk", "label": "Ukrainian"},
        ],
    },
    {
        "model": "cyrillic_PP-OCRv5_mobile_rec",
        "label": "Cyrillic — 29 lang khác (ngoài East Slavic)",
        "languages": [
            {"code": "rs_cyrillic", "label": "Serbian (Cyrillic)"},
            {"code": "bg", "label": "Bulgarian"},
            {"code": "mk", "label": "Macedonian"},
            {"code": "mn", "label": "Mongolian"},
            {"code": "kk", "label": "Kazakh"},
            {"code": "ky", "label": "Kyrgyz"},
            {"code": "tg", "label": "Tajik"},
            {"code": "tt", "label": "Tatar"},
            {"code": "ba", "label": "Bashkir"},
            {"code": "cv", "label": "Chuvash"},
            {"code": "mhr", "label": "Mari"},
            {"code": "udm", "label": "Udmurt"},
            {"code": "kv", "label": "Komi"},
            {"code": "os", "label": "Ossetian"},
            {"code": "sah", "label": "Sakha (Yakut)"},
            {"code": "kaa", "label": "Karakalpak"},
            {"code": "ce", "label": "Chechen"},
            {"code": "av", "label": "Avar"},
            {"code": "lez", "label": "Lezgian"},
            {"code": "dar", "label": "Dargwa"},
            {"code": "inh", "label": "Ingush"},
            {"code": "kbd", "label": "Kabardian"},
            {"code": "ady", "label": "Adyghe"},
            {"code": "ab", "label": "Abkhaz"},
            {"code": "lki", "label": "Lak"},
            {"code": "tab", "label": "Tabasaran"},
            {"code": "bua", "label": "Buriat"},
            {"code": "xal", "label": "Kalmyk"},
            {"code": "tyv", "label": "Tuvinian"},
            {"code": "mo", "label": "Moldovan"},
        ],
    },
    {
        "model": "arabic_PP-OCRv5_mobile_rec",
        "label": "Arabic script",
        "languages": [
            {"code": "ar", "label": "Arabic"},
            {"code": "fa", "label": "Persian (Farsi)"},
            {"code": "ur", "label": "Urdu"},
            {"code": "ug", "label": "Uyghur"},
            {"code": "ps", "label": "Pashto"},
            {"code": "sd", "label": "Sindhi"},
            {"code": "bal", "label": "Balochi"},
        ],
    },
    {
        "model": "devanagari_PP-OCRv5_mobile_rec",
        "label": "Devanagari script",
        "languages": [
            {"code": "hi", "label": "Hindi"},
            {"code": "mr", "label": "Marathi"},
            {"code": "ne", "label": "Nepali"},
            {"code": "sa", "label": "Sanskrit"},
            {"code": "gom", "label": "Konkani"},
            {"code": "bh", "label": "Bihari"},
            {"code": "bho", "label": "Bhojpuri"},
            {"code": "mai", "label": "Maithili"},
            {"code": "mah", "label": "Magahi"},
            {"code": "ang", "label": "Angika"},
            {"code": "sck", "label": "Sadri"},
            {"code": "new", "label": "Newari"},
            {"code": "bgc", "label": "Haryanvi"},
        ],
    },
    {
        "model": "th_PP-OCRv5_mobile_rec",
        "label": "Thai",
        "languages": [
            {"code": "th", "label": "Thai"},
        ],
    },
    {
        "model": "el_PP-OCRv5_mobile_rec",
        "label": "Greek",
        "languages": [
            {"code": "el", "label": "Greek"},
        ],
    },
    {
        "model": "ta_PP-OCRv5_mobile_rec",
        "label": "Tamil",
        "languages": [
            {"code": "ta", "label": "Tamil"},
        ],
    },
    {
        "model": "te_PP-OCRv5_mobile_rec",
        "label": "Telugu",
        "languages": [
            {"code": "te", "label": "Telugu"},
        ],
    },
]


def _flatten_languages() -> list[dict[str, object]]:
    """Flat list cho client cần lookup nhanh code → model."""
    flat: list[dict[str, object]] = [
        {"code": "auto", "label": "Auto detect",
         "note": "Thử 'en' trước, fallback parallel ch/japan/korean/ru/vi/ar/hi/th nếu confidence < 0.85"}
    ]
    for group in LANGUAGE_GROUPS:
        for lang in group["languages"]:  # type: ignore[union-attr]
            flat.append({**lang, "model": group["model"]})  # type: ignore[arg-type]
    return flat


@app.get("/v1/languages")
async def list_languages() -> dict[str, object]:
    """Liệt kê toàn bộ lang code OCR support (109 codes), kèm metadata model.

    Group theo recognition model (10 model + 1 alias auto).
    Mọi multi-script wrapper đều bao gồm `en` trong character set → có thể nhận
    diện song ngữ (ngôn ngữ chính + English) mà không cần đổi model.
    """
    flat = _flatten_languages()
    return {
        "service": "ocr",
        "engine": "PaddleOCR PP-OCRv5",
        "count": len(flat),
        "model_groups": LANGUAGE_GROUPS,
        "languages": flat,
    }


def _build_paragraphs_payload(
    blocks: list, reading_order: str,
) -> tuple[list[dict], str, str]:
    """Gọi merger và trả về (paragraphs_payload, resolved_reading_order, full_text_paragraphs)."""
    # Resolve "auto" trước, để cache được chính xác và client trace lại quyết định.
    if reading_order == "auto":
        resolved = detect_reading_order(blocks)
    else:
        resolved = reading_order
    paragraphs = merge_blocks_into_paragraphs(blocks, reading_order=resolved)  # type: ignore[arg-type]
    payload = [
        {
            "text": p.text,
            "bbox": p.bbox,
            "block_indices": p.block_indices,
            "avg_confidence": p.avg_confidence,
            "line_count": p.line_count,
        }
        for p in paragraphs
    ]
    full_text_paragraphs = paragraphs_to_full_text(paragraphs)
    return payload, resolved, full_text_paragraphs


async def _run_manga_ocr(image_bytes: bytes, request_id: str) -> OcrResponse:
    """Manga pipeline: PaddleOCR det + manga-ocr rec + bubble clustering + RTL sort."""
    from manga_pipeline import run_manga_pipeline

    # Cache key riêng cho manga mode
    cache_key = _cache_key(image_bytes, "mode=manga")
    cached = _cache_get(cache_key)
    if cached is not None:
        log.info("ocr_cache_hit_manga request_id=%s", request_id)
        return OcrResponse(
            request_id=request_id,
            processing_time_ms=cached["processing_time_ms"],
            lang="japan",
            detected_lang="japan",
            image_width=cached["image_width"],
            image_height=cached["image_height"],
            full_text=cached["full_text"],
            text_blocks=[
                OcrTextBlock(text=b["text"], confidence=b["confidence"], bbox=b["bbox"])
                for b in cached["text_blocks"]
            ],
            paragraphs=[OcrParagraphPayload(**p) for p in cached["paragraphs"]],
            reading_order="rtl",
        )

    engine: OcrEngine = app.state.engine
    started = time.perf_counter()
    try:
        blocks, bubbles, width, height = await run_manga_pipeline(
            image_bytes=image_bytes,
            engine=engine,
            use_manga_ocr_for_recognition=True,
        )
    except Exception as exc:
        log.exception("manga_pipeline_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Manga pipeline lỗi: {exc!s}"[:500],
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    paragraphs_payload = [
        {
            "text": b.text,
            "bbox": b.bbox,
            "block_indices": b.block_indices,
            "avg_confidence": b.avg_confidence,
            "line_count": b.line_count,
        }
        for b in bubbles
    ]
    full_text = "\n\n".join(b.text for b in bubbles)

    log.info(
        "manga_ocr_ok request_id=%s n_lines=%d n_bubbles=%d size=%dx%d elapsed=%dms",
        request_id, len(blocks), len(bubbles), width, height, elapsed_ms,
    )

    text_blocks_payload = [
        {"text": b.text, "confidence": b.confidence, "bbox": b.bbox}
        for b in blocks
    ]
    response_payload = {
        "processing_time_ms": elapsed_ms,
        "image_width": width,
        "image_height": height,
        "full_text": full_text,
        "text_blocks": text_blocks_payload,
        "paragraphs": paragraphs_payload,
    }
    _cache_put(cache_key, response_payload)

    return OcrResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        lang="japan",
        detected_lang="japan",
        image_width=width,
        image_height=height,
        full_text=full_text,
        text_blocks=[
            OcrTextBlock(text=b.text, confidence=b.confidence, bbox=b.bbox)
            for b in blocks
        ],
        paragraphs=[OcrParagraphPayload(**p) for p in paragraphs_payload],
        reading_order="rtl",
    )


async def _run_ocr(
    image_bytes: bytes,
    lang: str,
    level: str,
    request_id: str,
    merge_paragraphs: bool = True,
    reading_order: str = "auto",
    mode: str = "general",
) -> OcrResponse:
    """Logic OCR chính — share giữa endpoint /v1/ocr (JSON) và /v1/ocr/upload (multipart)."""
    # Mode = manga route sang specialized pipeline (manga-ocr + bubble clustering + RTL).
    if mode == "manga":
        return await _run_manga_ocr(image_bytes, request_id)

    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"lang '{lang}' không hợp lệ. Cho phép: {sorted(SUPPORTED_LANGS)}",
        )

    # Guard: PaddleOCR `predict(return_word_box=True)` crash worker khi ảnh > ~1.5M pixels
    # (test xác nhận: 1920x1080 = 2M crash, 800x200 = 160K OK).
    # Nếu user yêu cầu level=word mà ảnh quá lớn → degrade về block + log warning.
    # Sau khi resize MAX_DIMENSION 1600, ảnh max là 1600x1200 = 1.92M pixels → still risky.
    # Pixel-count guard chính xác hơn dimension guard.
    return_word_box = level == "word"
    if return_word_box:
        try:
            # Inspect dim mà không load full ảnh vào memory (PIL header check)
            import io
            from PIL import Image as _PILImage
            with _PILImage.open(io.BytesIO(image_bytes)) as _probe:
                w0, h0 = _probe.size
            # Tính dim sau resize MAX_IMAGE_DIMENSION
            from engine import MAX_IMAGE_DIMENSION
            max_dim = max(w0, h0)
            if max_dim > MAX_IMAGE_DIMENSION:
                scale = MAX_IMAGE_DIMENSION / max_dim
                pixels_after = int(w0 * scale) * int(h0 * scale)
            else:
                pixels_after = w0 * h0
            if pixels_after > 1_500_000:
                log.warning(
                    "word_level_degraded request_id=%s reason=image_too_large size=%dx%d pixels_after_resize=%d",
                    request_id, w0, h0, pixels_after,
                )
                return_word_box = False
                level = "block"  # phản ánh đúng level thực tế trong response/cache
        except Exception as exc:  # noqa: BLE001 — guard không được fail request
            log.warning("word_level_guard_failed request_id=%s error=%s", request_id, exc)

    # Cache key bao gồm cả merge config — cùng ảnh nhưng khác config phải tính lại merger.
    cache_key = _cache_key(
        image_bytes, f"{lang}:{level}:merge={int(merge_paragraphs)}:order={reading_order}",
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        log.info(
            "ocr_cache_hit request_id=%s lang=%s level=%s merge=%s order=%s",
            request_id, lang, level, merge_paragraphs, reading_order,
        )
        paragraphs_resp = (
            [OcrParagraphPayload(**p) for p in cached["paragraphs"]]
            if cached.get("paragraphs") is not None else None
        )
        return OcrResponse(
            request_id=request_id,
            processing_time_ms=cached["processing_time_ms"],
            lang=lang,
            detected_lang=cached["detected_lang"],
            image_width=cached["image_width"],
            image_height=cached["image_height"],
            full_text=cached["full_text"],
            text_blocks=[
                OcrTextBlock(
                    text=b["text"],
                    confidence=b["confidence"],
                    bbox=b["bbox"],
                    words=[OcrWord(text=w["text"], bbox=w["bbox"]) for w in b.get("words") or []] or None,
                )
                for b in cached["text_blocks"]
            ],
            paragraphs=paragraphs_resp,
            reading_order=cached.get("reading_order"),
        )

    engine: OcrEngine = app.state.engine
    started = time.perf_counter()
    try:
        if lang == "auto":
            blocks, width, height, detected_lang = await engine.ocr_auto(image_bytes, return_word_box)
        else:
            blocks, width, height = await engine.ocr(image_bytes, lang, return_word_box)
            detected_lang = lang
    except Exception as exc:
        log.exception("ocr_failed request_id=%s lang=%s level=%s", request_id, lang, level)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OCR engine lỗi: {exc!s}"[:500],
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    text_blocks_payload = []
    for b in blocks:
        words_payload = (
            [{"text": w.text, "bbox": w.bbox} for w in b.words]
            if b.words else None
        )
        text_blocks_payload.append({
            "text": b.text,
            "confidence": b.confidence,
            "bbox": b.bbox,
            "words": words_payload,
        })

    paragraphs_payload: list[dict] | None = None
    resolved_order: str | None = None
    if merge_paragraphs and blocks:
        paragraphs_payload, resolved_order, full_text = _build_paragraphs_payload(
            blocks, reading_order,
        )
    else:
        # Khi không merge, full_text giữ format cũ — line-by-line nối "\n".
        full_text = "\n".join(b.text for b in blocks)

    log.info(
        "ocr_ok request_id=%s lang=%s level=%s detected=%s n_blocks=%d n_para=%s "
        "size=%dx%d order=%s elapsed=%dms",
        request_id, lang, level, detected_lang, len(blocks),
        len(paragraphs_payload) if paragraphs_payload is not None else "off",
        width, height, resolved_order or "off", elapsed_ms,
    )

    response_payload = {
        "processing_time_ms": elapsed_ms,
        "detected_lang": detected_lang,
        "image_width": width,
        "image_height": height,
        "full_text": full_text,
        "text_blocks": text_blocks_payload,
        "paragraphs": paragraphs_payload,
        "reading_order": resolved_order,
    }
    _cache_put(cache_key, response_payload)

    return OcrResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        lang=lang,
        detected_lang=detected_lang,
        image_width=width,
        image_height=height,
        full_text=full_text,
        text_blocks=[
            OcrTextBlock(
                text=b.text,
                confidence=b.confidence,
                bbox=b.bbox,
                words=[OcrWord(text=w.text, bbox=w.bbox) for w in b.words] if b.words else None,
            )
            for b in blocks
        ],
        paragraphs=(
            [OcrParagraphPayload(**p) for p in paragraphs_payload]
            if paragraphs_payload is not None else None
        ),
        reading_order=resolved_order,  # type: ignore[arg-type]
    )


@app.post("/v1/ocr", response_model=OcrResponse)
async def ocr_endpoint(req: OcrRequest) -> OcrResponse:
    """OCR qua JSON body — input là `image` (base64) hoặc `image_url`."""
    request_id = req.request_id or str(uuid.uuid4())
    if req.image_url:
        image_bytes = await _fetch_image_url(req.image_url)
    else:
        try:
            image_bytes = base64.b64decode(req.image or "", validate=False)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Base64 decode lỗi: {exc}",
            ) from exc
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Ảnh rỗng",
            )
    return await _run_ocr(
        image_bytes, req.lang, req.level, request_id,
        merge_paragraphs=req.merge_paragraphs,
        reading_order=req.reading_order,
        mode=req.mode,
    )


@app.post("/v1/ocr/upload", response_model=OcrResponse)
async def ocr_upload_endpoint(
    file: UploadFile = File(..., description="File ảnh PNG/JPG/WebP, max 10MB"),
    lang: str = Form("auto"),
    level: Literal["block", "word"] = Form("block"),
    merge_paragraphs: bool = Form(True),
    reading_order: Literal["ltr", "rtl", "auto"] = Form("auto"),
    mode: Literal["general", "manga"] = Form("general"),
    request_id: str | None = Form(None),
) -> OcrResponse:
    """OCR qua multipart upload — phù hợp cho web tool drag-drop hoặc CLI curl -F."""
    rid = request_id or str(uuid.uuid4())
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File ảnh rỗng",
        )
    if len(image_bytes) > URL_FETCH_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File {len(image_bytes)} bytes vượt giới hạn {URL_FETCH_MAX_BYTES}",
        )
    return await _run_ocr(
        image_bytes, lang, level, rid,
        merge_paragraphs=merge_paragraphs,
        reading_order=reading_order,
        mode=mode,
    )
