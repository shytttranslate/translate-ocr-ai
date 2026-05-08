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

SUPPORTED_LANGS = frozenset({
    "auto", "en", "vi", "ch", "chinese_cht", "japan", "korean", "ru",
})

AUTO_DEFAULT_CONF_THRESHOLD = 0.85
AUTO_DEFAULT_LANG = "en"
AUTO_FALLBACK_LANGS = ("ch", "japan", "korean", "ru")

POOL_SIZE = int(os.environ.get("PADDLEOCR_POOL_SIZE", "8"))
PADDLEOCR_DEVICE = os.environ.get("PADDLEOCR_DEVICE", "auto")
USE_MOBILE_MODEL = os.environ.get("PADDLEOCR_USE_MOBILE", "0") == "1"


@dataclass
class OcrBlock:
    text: str
    confidence: float
    bbox: list[list[int]]


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
        log.info(
            "engine_init pool_size=%d device=%s mobile=%s cuda=%s",
            POOL_SIZE, self._device, USE_MOBILE_MODEL, _HAS_GPU,
        )

    @staticmethod
    def _resolve_device() -> str:
        if PADDLEOCR_DEVICE == "auto":
            return "gpu" if _HAS_GPU else "cpu"
        return PADDLEOCR_DEVICE

    @staticmethod
    def _resolve_lang(lang: str) -> str:
        if lang in ("vi", "auto"):
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
                "use_textline_orientation": True,
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

    def _run_inference(self, engine: Any, bgr: np.ndarray) -> list[OcrBlock]:
        version = self._detect_api_version()
        if version == "v3":
            return self._parse_v3_result(engine.predict(bgr))
        return self._parse_v2_result(engine.ocr(bgr, cls=True))

    @staticmethod
    def _parse_v3_result(raw: Any) -> list[OcrBlock]:
        blocks: list[OcrBlock] = []
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
            for i, text in enumerate(texts):
                t = str(text).strip()
                if not t:
                    continue
                conf = float(scores[i]) if i < len(scores) else 0.0
                if i < len(polys):
                    poly = polys[i]
                    bbox = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in poly]
                else:
                    bbox = [[0, 0], [0, 0], [0, 0], [0, 0]]
                blocks.append(OcrBlock(text=t, confidence=conf, bbox=bbox))
        return blocks

    @staticmethod
    def _parse_v2_result(raw: Any) -> list[OcrBlock]:
        blocks: list[OcrBlock] = []
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
            bbox_int = [[int(round(p[0])), int(round(p[1]))] for p in bbox_pts]
            blocks.append(OcrBlock(text=text, confidence=conf, bbox=bbox_int))
        return blocks

    @staticmethod
    def _decode_image(image_bytes: bytes) -> tuple[np.ndarray, int, int]:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size
        rgb = np.asarray(img)
        bgr = rgb[:, :, ::-1].copy()
        return bgr, width, height

    async def ocr(self, image_bytes: bytes, lang: str) -> tuple[list[OcrBlock], int, int]:
        bgr, width, height = await asyncio.to_thread(self._decode_image, image_bytes)
        pool = self._get_pool(lang)
        await pool.ensure_initialized(self._build_engine)
        engine = await pool.acquire()
        try:
            blocks = await asyncio.to_thread(self._run_inference, engine, bgr)
        finally:
            await pool.release(engine)
        return blocks, width, height

    async def ocr_auto(self, image_bytes: bytes) -> tuple[list[OcrBlock], int, int, str]:
        default_blocks, width, height = await self.ocr(image_bytes, AUTO_DEFAULT_LANG)
        default_avg = (
            sum(b.confidence for b in default_blocks) / len(default_blocks)
            if default_blocks else 0.0
        )
        if default_blocks and default_avg >= AUTO_DEFAULT_CONF_THRESHOLD:
            log.info("auto_pass1_win lang=%s conf=%.3f n=%d",
                     AUTO_DEFAULT_LANG, default_avg, len(default_blocks))
            return default_blocks, width, height, AUTO_DEFAULT_LANG

        log.info("auto_pass1_weak conf=%.3f n=%d → fallback CJK+ru",
                 default_avg, len(default_blocks))

        async def _try(lang: str) -> tuple[str, list[OcrBlock], float]:
            try:
                blocks, _, _ = await self.ocr(image_bytes, lang)
                avg = sum(b.confidence for b in blocks) / len(blocks) if blocks else 0.0
                return lang, blocks, avg
            except Exception as exc:  # noqa: BLE001
                log.warning("auto_fallback_failed lang=%s error=%s", lang, exc)
                return lang, [], 0.0

        results = await asyncio.gather(*[_try(lang) for lang in AUTO_FALLBACK_LANGS])

        import math
        candidates = [(AUTO_DEFAULT_LANG, default_blocks, default_avg)] + list(results)
        best_lang, best_blocks, best_avg = max(
            candidates, key=lambda x: x[2] * math.log1p(len(x[1])),
        )
        log.info("auto_pass2_win lang=%s conf=%.3f n=%d",
                 best_lang, best_avg, len(best_blocks))
        return best_blocks, width, height, best_lang

    async def warm_up(self, langs: list[str]) -> None:
        for lang in langs:
            try:
                pool = self._get_pool(lang)
                await pool.ensure_initialized(self._build_engine)
            except Exception as exc:  # noqa: BLE001
                log.warning("warmup_failed lang=%s error=%s", lang, exc)
