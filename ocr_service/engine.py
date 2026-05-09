"""PaddleOCR engine pool — async-safe wrapper.

Tách khỏi API gateway để OCR không block translate/dict requests.
Mỗi lang có pool N engine instance, request acquire/release từ pool → parallel thật.

Auto-detect GPU: nếu paddle build với CUDA → dùng GPU (nhanh 5-10x).
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

# PaddlePaddle 3.x: tắt mkldnn + PIR executor để tránh bug PIR ArrayAttribute<DoubleAttribute>.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

import numpy as np
from PIL import Image

_HAS_GPU = False
try:
    import paddle  # type: ignore[import-not-found]
    paddle.set_flags({
        "FLAGS_use_mkldnn": False,
        "FLAGS_enable_pir_in_executor": False,
    })
    _HAS_GPU = bool(paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
except Exception:  # noqa: BLE001
    pass

log = logging.getLogger("ocr_service.engine")

# PaddleOCR PP-OCRv5 supported lang codes (109 total).
# Source: https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
#
# Mỗi lang code map sang 1 trong các model:
# - PP-OCRv5_mobile_rec / server_rec  (default, cover ch+chinese_cht+en+japan)
# - en_PP-OCRv5_mobile_rec            (English thuần, KHÔNG cover dấu Latin extended)
# - korean_PP-OCRv5_mobile_rec        (Hangul + en)
# - latin_PP-OCRv5_mobile_rec         (47 lang Latin extended bao gồm vi, fr, de, es...)
# - eslav_PP-OCRv5_mobile_rec         (East Slavic: ru, be, uk)
# - cyrillic_PP-OCRv5_mobile_rec      (34 lang Cyrillic)
# - arabic_PP-OCRv5_mobile_rec        (9 lang Arabic script)
# - devanagari_PP-OCRv5_mobile_rec    (14 lang Devanagari)
# - th/el/ta/te dedicated mobile rec  (Thai, Greek, Tamil, Telugu)

_DEFAULT_LANGS = {"ch", "chinese_cht", "en", "japan"}  # PP-OCRv5_mobile_rec default
_KOREAN_LANGS = {"korean"}
_LATIN_LANGS = {
    "fr", "de", "af", "it", "es", "bs", "pt", "cs", "cy", "da", "et", "ga",
    "hr", "uz", "hu", "rs_latin", "id", "oc", "is", "lt", "mi", "ms", "nl",
    "no", "pl", "sk", "sl", "sq", "sv", "sw", "tl", "tr", "la", "az", "ku",
    "lv", "mt", "pi", "ro", "vi", "fi", "eu", "gl", "lb", "rm", "ca", "qu",
}
_ESLAV_LANGS = {"ru", "be", "uk"}
_CYRILLIC_LANGS = {
    "rs_cyrillic", "bg", "mn", "ab", "ady", "kbd", "av", "dar", "inh", "ce",
    "lki", "lez", "tab", "kk", "ky", "tg", "mk", "tt", "cv", "ba", "mhr",
    "mo", "udm", "kv", "os", "bua", "xal", "tyv", "sah", "kaa",
}
_ARABIC_LANGS = {"ar", "fa", "ug", "ur", "ps", "sd", "bal"}  # ku cũng cover Arabic
_DEVANAGARI_LANGS = {
    "hi", "mr", "ne", "bh", "mai", "ang", "bho", "mah", "sck", "new",
    "gom", "sa", "bgc",
}
_DEDICATED_LANGS = {"th", "el", "ta", "te"}

SUPPORTED_LANGS = frozenset(
    {"auto"}
    | _DEFAULT_LANGS
    | _KOREAN_LANGS
    | _LATIN_LANGS
    | _ESLAV_LANGS
    | _CYRILLIC_LANGS
    | _ARABIC_LANGS
    | _DEVANAGARI_LANGS
    | _DEDICATED_LANGS
)

AUTO_DEFAULT_CONF_THRESHOLD = 0.85
# Default fallback model — dùng khi OSD low-conf hoặc script không nằm trong PaddleOCR coverage.
# PP-OCRv5_mobile_rec cover ch + chinese_cht + en + japan trong 1 model.
AUTO_DEFAULT_LANG = "ch"

# === Tesseract OSD-based auto-detect ===
# Threshold script_conf: theo Google paper "reasonably confident" ≥ 1.5, "very confident" ≥ 2.5.
# OSD chạy trên FULL image (post-resize) chứ không crop — Tesseract OSD work tốt nhất ở
# page-level vì cần đủ char để decide script. Crop từng bbox thường thiếu char → "Too few".
TESSERACT_OSD_CONF_THRESHOLD = float(os.environ.get("OSD_CONF_THRESHOLD", "1.5"))

# Map Tesseract OSD script → PaddleOCR PP-OCRv5 lang code.
# OSD label: xem `osd.traineddata` (Google Research). Một số script Tesseract trả không có
# trong PaddleOCR (Hebrew, Bengali, Tibetan, Mongolian, ...) → fallback về default `ch`.
OSD_SCRIPT_TO_PADDLE_LANG: dict[str, str] = {
    "Latin": "vi",                # latin_PP-OCRv5_mobile_rec cover 47 lang Latin extended
    "Cyrillic": "ru",             # eslav_PP-OCRv5_mobile_rec
    "Han": "ch",                  # PP-OCRv5_mobile_rec cover ch + chinese_cht + en + japan
    "HanS": "ch",
    "HanT": "chinese_cht",
    "HanS_vert": "ch",
    "HanT_vert": "chinese_cht",
    "Japanese": "ch",             # ch model cover japan native
    "Japanese_vert": "ch",
    "Korean": "korean",           # korean_PP-OCRv5_mobile_rec
    "Korean_vert": "korean",
    "Arabic": "ar",               # arabic_PP-OCRv5_mobile_rec
    "Devanagari": "hi",           # devanagari_PP-OCRv5_mobile_rec
    "Thai": "th",
    "Greek": "el",
    "Tamil": "ta",
    # Telugu OSD chưa có training data riêng — Tesseract OSD label dùng "Latin" cho Telugu là chuyện thường.
    # → User truyền lang=te explicit nếu cần. Auto sẽ map về vi (latin) — sai cho Telugu nhưng rare.
}

# === Legacy Unicode script → lang (giữ lại làm sanity-check post-recognition) ===
SCRIPT_TO_LANG: dict[str, str] = {
    "latin_ext": "vi",
    "cyrillic": "ru",
    "korean": "korean",
    "arabic": "ar",
    "devanagari": "hi",
    "thai": "th",
    "greek": "el",
    "tamil": "ta",
    "telugu": "te",
}
DEFAULT_NATIVE_SCRIPTS = frozenset({"ascii", "cjk", "japanese_kana"})

POOL_SIZE = int(os.environ.get("PADDLEOCR_POOL_SIZE", "8"))
PADDLEOCR_DEVICE = os.environ.get("PADDLEOCR_DEVICE", "auto")
USE_MOBILE_MODEL = os.environ.get("PADDLEOCR_USE_MOBILE", "0") == "1"

# Resize ảnh down nếu max(width, height) > MAX_DIMENSION → giảm latency 2-3x với ảnh lớn.
# 1600px đủ cho text rõ; ảnh > 1600 thường là document scan high-res không cần thiết cho OCR.
MAX_IMAGE_DIMENSION = int(os.environ.get("PADDLEOCR_MAX_DIMENSION", "1600"))
# Hard cap pixel count — PaddleOCR detection crash với ảnh > ~1.5-2M pixels (ConvKernel segfault).
# Sau resize MAX_DIMENSION, nếu vẫn vượt cap → tiếp tục downscale theo sqrt(cap/pixels).
MAX_IMAGE_PIXELS = int(os.environ.get("PADDLEOCR_MAX_PIXELS", "1500000"))
# textline orientation classification — cần khi text xoay 90/180/270°. Tắt nhanh hơn ~10-15%.
USE_TEXTLINE_ORIENTATION = os.environ.get("PADDLEOCR_USE_TEXTLINE_ORI", "0") == "1"
# Filter nhiễu: bỏ text block có confidence < threshold. Default 0.3 (30%).
# Override bằng env PADDLEOCR_MIN_CONFIDENCE (giá trị 0.0-1.0).
MIN_CONFIDENCE = float(os.environ.get("PADDLEOCR_MIN_CONFIDENCE", "0.3"))


def _detect_scripts(text: str) -> set[str]:
    """Phân loại Unicode script của text → set các script gặp được.

    Dùng cho auto-detect lang: sau pass 1 OCR, scan text decode ra để biết
    có script nào nằm ngoài coverage của default model không.
    """
    scripts: set[str] = set()
    for ch in text:
        cp = ord(ch)
        if cp < 0x80:
            scripts.add("ascii")
        elif 0x0080 <= cp <= 0x024F:
            # Latin-1 supplement + Latin Extended A/B (dấu Việt, Pháp, Đức...)
            scripts.add("latin_ext")
        elif 0x0370 <= cp <= 0x03FF:
            scripts.add("greek")
        elif 0x0400 <= cp <= 0x04FF:
            scripts.add("cyrillic")
        elif 0x0590 <= cp <= 0x05FF:
            scripts.add("hebrew")
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            scripts.add("arabic")
        elif 0x0900 <= cp <= 0x097F:
            scripts.add("devanagari")
        elif 0x0B80 <= cp <= 0x0BFF:
            scripts.add("tamil")
        elif 0x0C00 <= cp <= 0x0C7F:
            scripts.add("telugu")
        elif 0x0E00 <= cp <= 0x0E7F:
            scripts.add("thai")
        elif 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
            # Hiragana + Katakana — model "ch" cover được
            scripts.add("japanese_kana")
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            scripts.add("cjk")
        elif 0xAC00 <= cp <= 0xD7AF:
            scripts.add("korean")
    return scripts


def _choose_fallback_lang(scripts: set[str]) -> str | None:
    """Trả về lang code cho pass 2 dựa script chiếm ưu thế ngoài CJK/ASCII.

    Trả None nếu không có script nào cần fallback.
    Ưu tiên thứ tự: cyrillic > arabic > devanagari > thai > greek > tamil > telugu
    > korean > latin_ext. (Latin extended ưu tiên cuối vì hay xuất hiện kèm CJK
    trong document mix → tránh bias.)
    """
    priority = (
        "cyrillic", "arabic", "devanagari", "thai",
        "greek", "tamil", "telugu", "korean", "latin_ext",
    )
    for s in priority:
        if s in scripts:
            return SCRIPT_TO_LANG.get(s)
    return None


@dataclass
class OcrWord:
    text: str
    bbox: list[list[int]]  # 4-corner polygon (giống bbox của block)


@dataclass
class OcrLine:
    text: str
    confidence: float
    bbox: list[list[int]]
    words: list[OcrWord] | None = None  # chỉ có khi level="word"


class ScriptDetector:
    """Wrap Tesseract OSD (Orientation & Script Detection).

    OSD trả về (script_name, script_conf). Conf >= 1.5 được Google paper coi là
    "reasonably confident", >= 2.5 là "very confident". Nếu Tesseract không
    install hoặc OSD lỗi, `detect()` trả None — caller phải fallback.

    Note: chạy trên page-level (full image post-resize) cho accuracy tốt nhất —
    crop từng bbox thường thiếu char → Tesseract báo "Too few characters".
    """

    def __init__(self) -> None:
        self._available = False
        self._pytesseract = None
        try:
            import pytesseract  # type: ignore[import-not-found]
            v = pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
            self._available = True
            log.info("script_detector_init tesseract=%s", v)
        except Exception as exc:  # noqa: BLE001
            log.warning("script_detector_unavailable error=%s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, image_bgr: np.ndarray) -> tuple[str | None, float]:
        """Run OSD on a BGR image. Return (script, confidence).

        Returns (None, 0.0) khi tesseract không install hoặc OSD lỗi (vd
        "Too few characters" trên ảnh không có text).
        """
        if not self._available or self._pytesseract is None:
            return None, 0.0
        try:
            from PIL import Image as _PILImage
            rgb = image_bgr[:, :, ::-1]
            img = _PILImage.fromarray(rgb)
            res = self._pytesseract.image_to_osd(
                img, output_type=self._pytesseract.Output.DICT,
            )
            script = res.get("script")
            conf = float(res.get("script_conf", 0.0))
            return script, conf
        except Exception as exc:  # noqa: BLE001
            log.debug("osd_detect_failed error=%s", exc)
            return None, 0.0


class _LangEnginePool:
    def __init__(self, lang: str, size: int) -> None:
        self.lang = lang
        self.size = size
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def ensure_initialized(self, build_fn: Any) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            log.info("pool_initializing lang=%s size=%d", self.lang, self.size)
            for i in range(self.size):
                engine = await asyncio.to_thread(build_fn, self.lang)
                self._queue.put_nowait(engine)
                log.info("pool_engine_built lang=%s idx=%d/%d", self.lang, i + 1, self.size)
            self._initialized = True

    async def acquire(self) -> Any:
        return await self._queue.get()

    async def release(self, engine: Any) -> None:
        self._queue.put_nowait(engine)


class OcrEngine:
    def __init__(self) -> None:
        self._pools: dict[str, _LangEnginePool] = {}
        self._pools_lock = threading.Lock()
        self._api_version: str | None = None
        self._device = self._resolve_device()
        # Tesseract OSD wrapper — graceful degrade nếu không có tesseract.
        self._script_detector = ScriptDetector()
        log.info(
            "engine_init pool_size=%d device=%s mobile=%s cuda=%s osd=%s",
            POOL_SIZE, self._device, USE_MOBILE_MODEL, _HAS_GPU,
            self._script_detector.available,
        )

    @staticmethod
    def _resolve_device() -> str:
        if PADDLEOCR_DEVICE == "auto":
            return "gpu" if _HAS_GPU else "cpu"
        return PADDLEOCR_DEVICE

    @staticmethod
    def _resolve_lang(lang: str) -> str:
        # `auto` → 'en' làm pass đầu (fallback chain xử lý các script khác).
        # `vi` GIỮ NGUYÊN — PaddleOCR sẽ load latin_PP-OCRv5_mobile_rec (cover diacritics Việt đầy đủ).
        # Map cũ vi→en SAI vì en model không cover dấu Việt.
        if lang == "auto":
            return AUTO_DEFAULT_LANG
        return lang

    def _detect_api_version(self) -> str:
        if self._api_version is not None:
            return self._api_version
        import inspect
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]

        sig = inspect.signature(PaddleOCR.__init__).parameters
        if "use_textline_orientation" in sig or "use_doc_orientation_classify" in sig:
            self._api_version = "v3"
        else:
            self._api_version = "v2"
        log.info("api_version_detected version=%s", self._api_version)
        return self._api_version

    def _build_engine(self, actual_lang: str) -> Any:
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]

        version = self._detect_api_version()
        if version == "v3":
            kwargs: dict[str, Any] = {
                "lang": actual_lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
                "enable_mkldnn": False,
                "device": self._device,
            }
            if USE_MOBILE_MODEL:
                kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
            return PaddleOCR(**kwargs)
        return PaddleOCR(
            use_angle_cls=True, lang=actual_lang, show_log=False,
            use_gpu=(self._device != "cpu"),
            ocr_version="PP-OCRv3", enable_mkldnn=False,
        )

    def _get_pool(self, lang: str) -> _LangEnginePool:
        actual = self._resolve_lang(lang)
        with self._pools_lock:
            if actual not in self._pools:
                self._pools[actual] = _LangEnginePool(lang=actual, size=POOL_SIZE)
        return self._pools[actual]

    def _run_inference(self, engine: Any, bgr: np.ndarray, return_word_box: bool = False) -> list[OcrLine]:
        version = self._detect_api_version()
        if version == "v3":
            kwargs = {"return_word_box": True} if return_word_box else {}
            return self._parse_v3_result(engine.predict(bgr, **kwargs))
        return self._parse_v2_result(engine.ocr(bgr, cls=True))

    @staticmethod
    def _parse_v3_result(raw: Any) -> list[OcrLine]:
        blocks: list[OcrLine] = []
        if not raw:
            return blocks
        for result in raw:
            data = result.json if hasattr(result, "json") else result
            if isinstance(data, dict) and "res" in data:
                data = data["res"]
            if not isinstance(data, dict):
                continue
            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []
            polys = data.get("rec_polys") or data.get("dt_polys") or []
            # Word-level (chỉ có khi predict với return_word_box=True)
            text_words = data.get("text_word") or []
            text_word_boxes = data.get("text_word_boxes") or []
            for i, text in enumerate(texts):
                t = str(text).strip()
                if not t:
                    continue
                conf = float(scores[i]) if i < len(scores) else 0.0
                # Lọc nhiễu: bỏ block có confidence < threshold
                if conf < MIN_CONFIDENCE:
                    continue
                if i < len(polys):
                    poly = polys[i]
                    bbox = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in poly]
                else:
                    bbox = [[0, 0], [0, 0], [0, 0], [0, 0]]

                # Parse word-level cho line này
                words: list[OcrWord] | None = None
                if i < len(text_words) and i < len(text_word_boxes):
                    line_words = text_words[i]
                    line_boxes = text_word_boxes[i]
                    words = []
                    for w_text, w_box in zip(line_words, line_boxes):
                        wt = str(w_text)
                        if not wt or wt.isspace():
                            continue
                        # box format: [xmin, ymin, xmax, ymax] → 4-corner polygon
                        x1, y1, x2, y2 = (int(round(float(v))) for v in w_box)
                        words.append(OcrWord(
                            text=wt,
                            bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        ))
                blocks.append(OcrLine(text=t, confidence=conf, bbox=bbox, words=words))
        return blocks

    @staticmethod
    def _parse_v2_result(raw: Any) -> list[OcrLine]:
        blocks: list[OcrLine] = []
        if not raw or not raw[0]:
            return blocks
        for entry in raw[0]:
            if entry is None or len(entry) < 2:
                continue
            bbox_pts, text_conf = entry[0], entry[1]
            if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                continue
            text = str(text_conf[0]).strip()
            if not text:
                continue
            conf = float(text_conf[1])
            # Lọc nhiễu: bỏ block có confidence < threshold
            if conf < MIN_CONFIDENCE:
                continue
            bbox_int = [[int(round(p[0])), int(round(p[1]))] for p in bbox_pts]
            blocks.append(OcrLine(text=text, confidence=conf, bbox=bbox_int))
        return blocks

    @staticmethod
    def _decode_image(image_bytes: bytes) -> tuple[np.ndarray, int, int]:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        original_width, original_height = img.size

        # 2-pass resize: (1) cap dimension, (2) cap pixel count.
        # PaddleOCR crash với pixels > ~1.5-2M ngay cả khi dimension nhỏ (vd ảnh vuông 1500x1500).
        scale = 1.0
        cur_w, cur_h = original_width, original_height
        if max(cur_w, cur_h) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(cur_w, cur_h)
            cur_w, cur_h = int(cur_w * scale), int(cur_h * scale)
        if cur_w * cur_h > MAX_IMAGE_PIXELS:
            extra = (MAX_IMAGE_PIXELS / (cur_w * cur_h)) ** 0.5
            scale *= extra
            cur_w, cur_h = int(cur_w * extra), int(cur_h * extra)
        if scale < 1.0:
            img = img.resize((cur_w, cur_h), Image.LANCZOS)
            log.info("image_resized %dx%d → %dx%d (scale=%.3f, pixels=%d)",
                     original_width, original_height, cur_w, cur_h, scale, cur_w * cur_h)

        rgb = np.asarray(img)
        bgr = rgb[:, :, ::-1].copy()
        return bgr, original_width, original_height, scale  # type: ignore[return-value]

    async def ocr(
        self, image_bytes: bytes, lang: str, return_word_box: bool = False,
    ) -> tuple[list[OcrLine], int, int]:
        bgr, width, height, scale = await asyncio.to_thread(self._decode_image, image_bytes)  # type: ignore[misc]
        pool = self._get_pool(lang)
        await pool.ensure_initialized(self._build_engine)
        engine = await pool.acquire()
        try:
            blocks = await asyncio.to_thread(self._run_inference, engine, bgr, return_word_box)
        finally:
            await pool.release(engine)
        # Scale bbox ngược về tỷ lệ ảnh gốc (nếu có resize).
        if scale != 1.0 and blocks:
            inv = 1.0 / scale
            for blk in blocks:
                blk.bbox = [[int(round(p[0] * inv)), int(round(p[1] * inv))] for p in blk.bbox]
                if blk.words:
                    for w in blk.words:
                        w.bbox = [[int(round(p[0] * inv)), int(round(p[1] * inv))] for p in w.bbox]
        return blocks, width, height

    async def ocr_auto(
        self, image_bytes: bytes, return_word_box: bool = False,
    ) -> tuple[list[OcrLine], int, int, str]:
        """Auto-detect lang via Tesseract OSD on full (resized) image.

        Pipeline:
            1. Decode + resize ảnh (re-use logic chung với recognition path).
            2. Tesseract OSD trên ảnh đã resize → (script, conf).
            3. Map script → PaddleOCR lang nếu conf đạt threshold.
            4. Recognition full với lang đã chọn.

        Fallback ladder:
            - Tesseract unavailable / OSD throw error → default `ch`.
            - script_conf < 1.5 → default `ch`.
            - Script không có trong PaddleOCR coverage (Hebrew, Bengali, ...) → default `ch`.
            - Recognition lang chọn lỗi → retry với `ch`.
        """
        # Decode + resize 1 lần (reuse cho cả OSD và recognition pass dưới).
        bgr, width, height, _scale = await asyncio.to_thread(self._decode_image, image_bytes)  # type: ignore[misc]

        chosen_lang = AUTO_DEFAULT_LANG
        chosen_script: str | None = None
        osd_conf = 0.0
        if self._script_detector.available:
            chosen_script, osd_conf = await asyncio.to_thread(
                self._script_detector.detect, bgr,
            )
            log.info("auto_osd script=%s conf=%.2f", chosen_script, osd_conf)
            if chosen_script and osd_conf >= TESSERACT_OSD_CONF_THRESHOLD:
                mapped = OSD_SCRIPT_TO_PADDLE_LANG.get(chosen_script)
                if mapped is not None:
                    chosen_lang = mapped
                else:
                    log.info(
                        "auto_script_unsupported script=%s → fallback %s",
                        chosen_script, AUTO_DEFAULT_LANG,
                    )
            else:
                log.info(
                    "auto_osd_low_conf conf=%.2f<%.2f → fallback %s",
                    osd_conf, TESSERACT_OSD_CONF_THRESHOLD, AUTO_DEFAULT_LANG,
                )
        else:
            log.info("auto_osd_unavailable → fallback %s", AUTO_DEFAULT_LANG)

        try:
            blocks, _, _ = await self.ocr(image_bytes, chosen_lang, return_word_box)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "auto_recognition_failed lang=%s error=%s → retry ch",
                chosen_lang, exc,
            )
            if chosen_lang != AUTO_DEFAULT_LANG:
                blocks, _, _ = await self.ocr(image_bytes, AUTO_DEFAULT_LANG, return_word_box)
                chosen_lang = AUTO_DEFAULT_LANG
            else:
                raise

        return blocks, width, height, chosen_lang

    async def warm_up(self, langs: list[str]) -> None:
        for lang in langs:
            try:
                pool = self._get_pool(lang)
                await pool.ensure_initialized(self._build_engine)
            except Exception as exc:  # noqa: BLE001
                log.warning("warmup_failed lang=%s error=%s", lang, exc)
