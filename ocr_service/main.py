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
    is_meaningless_text,
    merge_blocks_into_paragraphs,
    paragraphs_to_full_text,
)


# Limit ảnh fetch từ URL — tránh DOS bằng URL trỏ tới file lớn.
URL_FETCH_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
URL_FETCH_TIMEOUT_S = 15.0


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
    reading_order: Literal["ltr", "rtl", "auto"] = Field(
        default="auto",
        description=(
            "Thứ tự đọc khi sort block: ltr = trái→phải (default), "
            "rtl = phải→trái (manga JP, Arabic), auto = tự detect theo aspect ratio + CJK ratio."
        ),
    )
    mode: Literal["general", "manga"] = Field(
        default="general",
        description=(
            "general (default) = PaddleOCR pipeline cho document/screenshot.\n"
            "manga = specialized pipeline cho comic/manga JP: PaddleOCR detection + "
            "manga-ocr recognition (kha-white) + bubble clustering + RTL reading order. "
            "Khi mode=manga, lang/reading_order được override (lang=japan, RTL)."
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


class BboxPoint(BaseModel):
    """1 điểm trong polygon bbox — dùng object thay vì array để client dễ đọc."""
    x: int
    y: int


# ML Kit hierarchy: block > line > word
class OcrWord(BaseModel):
    text: str
    bbox: list[BboxPoint] = Field(description="4-corner polygon [TL, TR, BR, BL]")
    confidence: float = Field(ge=0.0, le=1.0)


class OcrLine(BaseModel):
    text: str
    bbox: list[BboxPoint]
    confidence: float = Field(ge=0.0, le=1.0)
    words: list[OcrWord] = Field(
        default_factory=list,
        description="Chỉ có khi request `level=word`. Empty với block/line level.",
    )


class OcrBlock(BaseModel):
    """ML Kit Block = paragraph: gộp các line liền kề / cùng speech bubble."""
    text: str = Field(description="Text gộp từ các line, separator '\\n' (manga vertical: '')")
    bbox: list[BboxPoint] = Field(description="Axis-aligned bbox bao quanh block, 4 góc [TL,TR,BR,BL]")
    confidence: float = Field(ge=0.0, le=1.0, description="Avg confidence của lines trong block")
    line_count: int = Field(ge=1)
    lines: list[OcrLine] = Field(
        default_factory=list,
        description="Empty khi `level=block`. Có khi `level=line` hoặc `word`.",
    )


class OcrResponse(BaseModel):
    request_id: str
    processing_time_ms: int
    lang: str
    detected_lang: str
    image_width: int
    image_height: int
    full_text: str
    blocks: list[OcrBlock] = Field(
        default_factory=list,
        description=(
            "ML Kit hierarchy: blocks → lines → words. Luôn populate đầy đủ — "
            "ảnh lớn được auto-resize xuống cap an toàn (~1M pixels) trước khi OCR, "
            "bbox được scale ngược về coordinate gốc."
        ),
    )
    reading_order: Literal["ltr", "rtl"] | None = Field(
        default=None,
        description="Reading order thực tế dùng để sort blocks.",
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


def _flatten_languages() -> list[dict[str, str]]:
    """Flat list `{code, label}` — đơn giản cho client dropdown / i18n picker."""
    flat: list[dict[str, str]] = [{"code": "auto", "label": "Auto detect"}]
    for group in LANGUAGE_GROUPS:
        for lang in group["languages"]:  # type: ignore[union-attr]
            flat.append({"code": lang["code"], "label": lang["label"]})  # type: ignore[index]
    return flat


@app.get("/v1/languages")
async def list_languages() -> dict[str, object]:
    """Liệt kê toàn bộ lang code OCR support — chỉ trả count + flat list `{code, label}`."""
    flat = _flatten_languages()
    return {
        "count": len(flat),
        "languages": flat,
    }


def _bbox_to_points(bbox: list[list[int]]) -> list[dict[str, int]]:
    """Convert bbox engine [[x,y], ...] → [{"x":x,"y":y}, ...] cho response API.

    Engine + paragraph_merger nội bộ vẫn dùng list-of-lists để giữ vector ops gọn,
    chỉ convert ở boundary trước khi serialize qua Pydantic.
    """
    return [{"x": int(p[0]), "y": int(p[1])} for p in bbox]


def _build_lines_from_engine(blocks: list) -> list[dict]:
    """Convert engine OcrLine (line-level) → list[OcrLine] payload dict.

    Words luôn populated nếu engine có set blk.words (= return_word_box=True).
    """
    lines: list[dict] = []
    for blk in blocks:
        words_payload: list[dict] = []
        if blk.words:
            words_payload = [
                {
                    "text": w.text,
                    "bbox": _bbox_to_points(w.bbox),
                    "confidence": blk.confidence,
                }
                for w in blk.words
            ]
        lines.append({
            "text": blk.text,
            "bbox": _bbox_to_points(blk.bbox),
            "confidence": blk.confidence,
            "words": words_payload,
        })
    return lines


def _build_blocks_payload(
    blocks: list, reading_order: str, image_bytes: bytes | None = None,
) -> tuple[list[dict], str, str]:
    """Gộp lines thành block theo paragraph_merger + reading order.

    Returns: (blocks_payload, resolved_reading_order, full_text)
    """
    if reading_order == "auto":
        resolved = detect_reading_order(blocks)
    else:
        resolved = reading_order

    # Decode grayscale 1 lần cho boldness detection (paragraph style split)
    image_gray = None
    if image_bytes is not None:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np
            arr = np.frombuffer(image_bytes, np.uint8)
            image_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        except Exception as exc:  # noqa: BLE001
            log.warning("decode_for_boldness_failed error=%s", exc)

    paragraphs = merge_blocks_into_paragraphs(
        blocks, reading_order=resolved, image_gray=image_gray,
    )

    # Pre-build lines payload to attach vào block
    line_lookup = _build_lines_from_engine(blocks)

    blocks_payload: list[dict] = []
    for p in paragraphs:
        # Lines LUÔN populate (giống ML Kit). Words populate nếu engine có set.
        attached_lines = [
            line_lookup[i] for i in p.block_indices if 0 <= i < len(line_lookup)
        ]
        blocks_payload.append({
            "text": p.text,
            "bbox": _bbox_to_points(p.bbox),
            "confidence": p.avg_confidence,
            "line_count": p.line_count,
            "lines": attached_lines,
        })
    full_text = paragraphs_to_full_text(paragraphs)
    return blocks_payload, resolved, full_text


async def _run_manga_ocr(image_bytes: bytes, request_id: str) -> OcrResponse:
    """Manga pipeline: PaddleOCR det + manga-ocr rec + bubble clustering + RTL sort.

    Manga-ocr không trả word bbox → words=[] luôn ở manga mode.
    """
    from manga_pipeline import run_manga_pipeline

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
    line_lookup = _build_lines_from_engine(blocks)
    blocks_payload: list[dict] = []
    for b in bubbles:
        attached_lines = [
            line_lookup[i] for i in b.block_indices if 0 <= i < len(line_lookup)
        ]
        blocks_payload.append({
            "text": b.text,
            "bbox": _bbox_to_points(b.bbox),
            "confidence": b.avg_confidence,
            "line_count": b.line_count,
            "lines": attached_lines,
        })
    full_text = "\n\n".join(b.text for b in bubbles)

    log.info(
        "manga_ocr_ok request_id=%s n_lines=%d n_blocks=%d size=%dx%d elapsed=%dms",
        request_id, len(blocks), len(bubbles), width, height, elapsed_ms,
    )

    return OcrResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        lang="japan",
        detected_lang="japan",
        image_width=width,
        image_height=height,
        full_text=full_text,
        blocks=[OcrBlock(**p) for p in blocks_payload],
        reading_order="rtl",
    )


async def _run_ocr(
    image_bytes: bytes,
    lang: str,
    request_id: str,
    reading_order: str = "auto",
    mode: str = "general",
) -> OcrResponse:
    """Logic OCR chính — luôn trả full hierarchy block + line + word.

    Engine `_decode_image` áp dụng pixel cap thấp hơn (~1M) khi `return_word_box=True`,
    đảm bảo word-level luôn được trả ra, kể cả với ảnh lớn (bbox sẽ scale ngược về
    coordinate gốc trong engine).
    """
    if mode == "manga":
        return await _run_manga_ocr(image_bytes, request_id)

    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"lang '{lang}' không hợp lệ. Cho phép: {sorted(SUPPORTED_LANGS)}",
        )

    return_word_box = True
    engine: OcrEngine = app.state.engine
    started = time.perf_counter()
    try:
        if lang == "auto":
            blocks, width, height, detected_lang = await engine.ocr_auto(image_bytes, return_word_box)
        else:
            blocks, width, height = await engine.ocr(image_bytes, lang, return_word_box)
            detected_lang = lang
    except Exception as exc:
        log.exception("ocr_failed request_id=%s lang=%s", request_id, lang)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OCR engine lỗi: {exc!s}"[:500],
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Filter blocks có text vô nghĩa (vd `$\a$` từ icon misread, `\A`, toàn symbol).
    # Conservative — chỉ bắt backslash + pure-symbol ≥2 chars, giữ `?`/`!` đơn lẻ.
    pre_n = len(blocks)
    blocks = [b for b in blocks if not is_meaningless_text(b.text)]
    if len(blocks) < pre_n:
        log.info(
            "filter_meaningless request_id=%s dropped=%d kept=%d",
            request_id, pre_n - len(blocks), len(blocks),
        )

    blocks_payload, resolved_order, full_text = _build_blocks_payload(
        blocks, reading_order, image_bytes=image_bytes,
    )

    log.info(
        "ocr_ok request_id=%s lang=%s detected=%s n_lines=%d n_blocks=%d "
        "size=%dx%d order=%s wb=%s elapsed=%dms",
        request_id, lang, detected_lang, len(blocks),
        len(blocks_payload), width, height, resolved_order,
        return_word_box, elapsed_ms,
    )

    return OcrResponse(
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        lang=lang,
        detected_lang=detected_lang,
        image_width=width,
        image_height=height,
        full_text=full_text,
        blocks=[OcrBlock(**b) for b in blocks_payload],
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
        image_bytes, req.lang, request_id,
        reading_order=req.reading_order,
        mode=req.mode,
    )


@app.post("/v1/ocr/upload", response_model=OcrResponse)
async def ocr_upload_endpoint(
    file: UploadFile = File(..., description="File ảnh PNG/JPG/WebP, max 10MB"),
    lang: str = Form("auto"),
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
        image_bytes, lang, rid,
        reading_order=reading_order,
        mode=mode,
    )
