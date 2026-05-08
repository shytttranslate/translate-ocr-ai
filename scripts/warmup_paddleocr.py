#!/usr/bin/env python3
"""Pre-download + init PaddleOCR engines để tránh cold-start runtime.

Chạy 1 lần sau khi pip install paddleocr. Tự detect API v3 (PP-OCRv5) hay v2.

Usage:
    /workspace/vbk-ai-server/.venv-api/bin/python scripts/warmup_paddleocr.py
    /workspace/vbk-ai-server/.venv-api/bin/python scripts/warmup_paddleocr.py en ch
"""
from __future__ import annotations

import os

# PaddlePaddle 3.x oneDNN + PIR executor chưa tương thích → spam WARN/ERROR
# "ConvertPirAttribute2RuntimeAttribute not support". Tắt cả 2 trước khi import paddle.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

import inspect  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

try:
    import paddle  # type: ignore[import-not-found]  # noqa: E402
    paddle.set_flags({
        "FLAGS_use_mkldnn": False,
        "FLAGS_enable_pir_in_executor": False,
    })
except Exception:  # noqa: BLE001
    pass

from paddleocr import PaddleOCR  # noqa: E402

DEFAULT_LANGS = ["en", "ch", "japan", "korean"]


def _detect_version() -> str:
    sig_params = inspect.signature(PaddleOCR.__init__).parameters
    if "use_textline_orientation" in sig_params or "use_doc_orientation_classify" in sig_params:
        return "v3"
    return "v2"


def _build(lang: str, version: str):
    if version == "v3":
        return PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    return PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        show_log=False,
        use_gpu=False,
        ocr_version="PP-OCRv3",
        enable_mkldnn=False,
    )


def _infer(engine, img: np.ndarray, version: str) -> None:
    if version == "v3":
        engine.predict(img)
    else:
        engine.ocr(img, cls=True)


def warm_up(langs: list[str]) -> None:
    version = _detect_version()
    print(f"PaddleOCR API version detected: {version}")
    print()

    dummy = np.full((80, 200, 3), 255, dtype=np.uint8)
    dummy[20:60, 30:60] = 0
    dummy[20:60, 80:110] = 0
    dummy[20:60, 130:160] = 0

    for lang in langs:
        t0 = time.time()
        print(f"[{lang}] khởi tạo PaddleOCR...", flush=True)
        try:
            engine = _build(lang, version)
        except Exception as exc:
            print(f"[{lang}] FAIL init: {exc}", flush=True)
            continue
        elapsed_init = time.time() - t0
        print(f"[{lang}] init OK ({elapsed_init:.1f}s), chạy dummy inference...", flush=True)

        try:
            _infer(engine, dummy, version)
        except Exception as exc:
            print(f"[{lang}] dummy inference WARN: {exc}", flush=True)

        elapsed_total = time.time() - t0
        print(f"[{lang}] total {elapsed_total:.1f}s", flush=True)
        print()


if __name__ == "__main__":
    langs = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_LANGS
    print(f"Warm-up {len(langs)} engine: {', '.join(langs)}")
    print(f"Cache đích: ~/.paddlex/ hoặc ~/.paddleocr/whl/")
    print()
    t0 = time.time()
    warm_up(langs)
    print(f"=== Tất cả engines đã warm-up — tổng {time.time()-t0:.1f}s ===")
