#!/usr/bin/env python3
"""Sinh OpenAPI 3.0.3 spec cho OCR API — phục vụ upload lên RapidAPI marketplace.

Run:    python3 tools/gen_openapi.py
Output:
    postman/openapi.yaml   (RapidAPI prefer YAML)
    postman/openapi.json   (backup format)

Spec chỉ cover OCR (5 endpoints), KHÔNG bao gồm Translate / Dictionary.
Mọi description/summary trong spec viết tiếng Anh để fit marketplace international;
log/comment trong file generator vẫn tiếng Việt theo quy chuẩn nội bộ.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
OUT_YAML = ROOT / "postman" / "openapi.yaml"
OUT_JSON = ROOT / "postman" / "openapi.json"

OPENAPI_VERSION = "3.0.3"
SPEC_VERSION = "0.3.0"

PROD_SERVER = "https://ocr.spacecloud.fit"
RAPIDAPI_SERVER = "https://vbk-ocr.p.rapidapi.com"

SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "0/0a/Newspaper_clipping.png/640px-Newspaper_clipping.png"
)


# ----------------------------------------------------------------------------
# Schemas — components.schemas
# ----------------------------------------------------------------------------
def _bbox_schema() -> dict:
    """4-corner polygon [TL, TR, BR, BL], mỗi corner là object {x, y} integer pixel."""
    return {
        "type": "array",
        "description": "4-corner polygon ordered [top-left, top-right, bottom-right, bottom-left]. Each corner is an object `{x, y}` of pixel coordinates.",
        "minItems": 4,
        "maxItems": 4,
        "items": {"$ref": "#/components/schemas/BboxPoint"},
        "example": [
            {"x": 10, "y": 12},
            {"x": 320, "y": 12},
            {"x": 320, "y": 48},
            {"x": 10, "y": 48},
        ],
    }


def build_schemas() -> dict:
    schemas: dict = {}

    schemas["BboxPoint"] = {
        "type": "object",
        "description": "Single pixel coordinate within a bbox polygon.",
        "required": ["x", "y"],
        "properties": {
            "x": {"type": "integer", "description": "Horizontal pixel coordinate."},
            "y": {"type": "integer", "description": "Vertical pixel coordinate."},
        },
    }

    schemas["OcrWord"] = {
        "type": "object",
        "description": "Word-level recognition entry (lowest hierarchy level).",
        "required": ["text", "bbox", "confidence"],
        "properties": {
            "text": {"type": "string", "description": "Recognized word text."},
            "bbox": _bbox_schema(),
            "confidence": {
                "type": "number",
                "format": "float",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score 0–1 (inherits parent line score).",
            },
        },
    }

    schemas["OcrLine"] = {
        "type": "object",
        "description": "Single text line. `words` may be empty when image exceeds 1.5M pixels (PaddleOCR auto-degrades word boxes for memory safety).",
        "required": ["text", "bbox", "confidence", "words"],
        "properties": {
            "text": {"type": "string", "description": "Recognized line text."},
            "bbox": _bbox_schema(),
            "confidence": {
                "type": "number",
                "format": "float",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "words": {
                "type": "array",
                "description": "Word-level breakdown. May be empty `[]` when word boxes are degraded.",
                "items": {"$ref": "#/components/schemas/OcrWord"},
            },
        },
    }

    schemas["OcrBlock"] = {
        "type": "object",
        "description": "Paragraph-level block (ML Kit Block hierarchy). Lines from the same speech bubble or paragraph are merged together.",
        "required": ["text", "bbox", "confidence", "line_count", "lines"],
        "properties": {
            "text": {
                "type": "string",
                "description": "Concatenated text of all lines. Separator is `\\n` (or empty for vertical manga text).",
            },
            "bbox": _bbox_schema(),
            "confidence": {
                "type": "number",
                "format": "float",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Average confidence across child lines.",
            },
            "line_count": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of source lines merged into this block.",
            },
            "lines": {
                "type": "array",
                "description": "Child lines (always populated to mirror Google ML Kit hierarchy).",
                "items": {"$ref": "#/components/schemas/OcrLine"},
            },
        },
    }

    schemas["OcrRequest"] = {
        "type": "object",
        "description": (
            "OCR request payload. Exactly **one** of `image` or `image_url` MUST be provided "
            "(server returns 422 if both or neither are supplied)."
        ),
        "additionalProperties": False,
        # Schema-level XOR: helps OpenAPI codegen tools (RapidAPI Code Snippets,
        # openapi-generator) emit a typed union on the client side instead of two
        # independent optional fields. Pydantic still enforces at runtime.
        "oneOf": [
            {"required": ["image"]},
            {"required": ["image_url"]},
        ],
        "properties": {
            "image": {
                "type": "string",
                "format": "byte",
                "nullable": True,
                "description": "Base64-encoded image bytes (PNG / JPEG / WebP). Mutually exclusive with `image_url`.",
            },
            "image_url": {
                "type": "string",
                "format": "uri",
                "nullable": True,
                "description": (
                    "HTTP/HTTPS URL of the image. Server fetches it (max 10 MB, 15s timeout). "
                    "Mutually exclusive with `image`. "
                    "**Security note:** the server fetches arbitrary URLs server-side; do not pass "
                    "URLs that resolve to private/internal networks (RFC1918, loopback, link-local, "
                    "metadata endpoints). Operators should restrict outbound egress accordingly."
                ),
            },
            "lang": {
                "type": "string",
                "default": "auto",
                "description": (
                    "Language hint. Use `auto` for automatic detection (tries `en` first, then "
                    "ch / japan / korean / ru / vi / ar / hi / th in parallel if confidence < 0.85). "
                    "See `GET /v1/languages` for the full list of supported codes."
                ),
                "example": "auto",
            },
            "reading_order": {
                "type": "string",
                "enum": ["ltr", "rtl", "auto"],
                "default": "auto",
                "description": (
                    "Block sort order. `ltr` = left-to-right (default), `rtl` = right-to-left "
                    "(manga JP, Arabic), `auto` = inferred from CJK character ratio + page aspect."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["general", "manga"],
                "default": "general",
                "description": (
                    "`general` (default) = PaddleOCR pipeline for documents/screenshots. "
                    "`manga` = specialized pipeline for comics: PaddleOCR detection + manga-ocr "
                    "recognition + speech-bubble clustering + RTL ordering. In manga mode the "
                    "`lang` and `reading_order` fields are forced to `japan` / `rtl`."
                ),
            },
            "request_id": {
                "type": "string",
                "nullable": True,
                "description": "Optional client-supplied trace ID. Echoed back in the response. A UUID is generated when omitted.",
            },
        },
    }

    schemas["OcrResponse"] = {
        "type": "object",
        "required": [
            "request_id",
            "processing_time_ms",
            "lang",
            "detected_lang",
            "image_width",
            "image_height",
            "full_text",
            "blocks",
        ],
        "properties": {
            "request_id": {"type": "string", "description": "Echo of request ID (or server-generated UUID)."},
            "processing_time_ms": {
                "type": "integer",
                "minimum": 0,
                "description": "Server-side wall-clock duration of the OCR pipeline in milliseconds.",
            },
            "lang": {
                "type": "string",
                "description": "Language code that was actually requested (echoes the input or `auto`).",
            },
            "detected_lang": {
                "type": "string",
                "description": "Language code resolved by auto-detect (equals `lang` when an explicit code was supplied).",
            },
            "image_width": {"type": "integer", "minimum": 1, "description": "Decoded image width in pixels."},
            "image_height": {"type": "integer", "minimum": 1, "description": "Decoded image height in pixels."},
            "full_text": {
                "type": "string",
                "description": "Concatenated text of all blocks in reading order. Blocks separated by `\\n\\n`.",
            },
            "blocks": {
                "type": "array",
                "description": "ML Kit hierarchy: blocks > lines > words. Always populated. `words` arrays may be empty for very large images.",
                "items": {"$ref": "#/components/schemas/OcrBlock"},
            },
            "reading_order": {
                "type": "string",
                "enum": ["ltr", "rtl"],
                "nullable": True,
                "description": "Reading order actually used to sort blocks (resolved from `auto` if needed).",
            },
        },
    }

    schemas["LanguageItem"] = {
        "type": "object",
        "required": ["code", "label"],
        "properties": {
            "code": {"type": "string", "description": "Language code accepted by the `lang` field.", "example": "vi"},
            "label": {"type": "string", "description": "Human-readable name.", "example": "Vietnamese"},
        },
    }

    schemas["LanguagesResponse"] = {
        "type": "object",
        "required": ["count", "languages"],
        "properties": {
            "count": {"type": "integer", "description": "Total number of language codes (including the `auto` alias)."},
            "languages": {
                "type": "array",
                "description": "Flat list of every supported code with its human-readable label.",
                "items": {"$ref": "#/components/schemas/LanguageItem"},
            },
        },
    }

    schemas["HealthResponse"] = {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"type": "string", "enum": ["ok"]},
            "service": {"type": "string", "enum": ["ocr"], "description": "Present on the readiness endpoint only."},
        },
    }

    schemas["ErrorResponse"] = {
        "type": "object",
        "description": "Default FastAPI error envelope.",
        "required": ["detail"],
        "properties": {
            "detail": {"type": "string", "description": "Human-readable error message."},
        },
    }

    schemas["ValidationErrorItem"] = {
        "type": "object",
        "required": ["loc", "msg", "type"],
        "properties": {
            "loc": {
                "type": "array",
                "items": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                "description": "JSON pointer to the offending field.",
            },
            "msg": {"type": "string"},
            "type": {"type": "string"},
        },
    }

    schemas["ValidationError"] = {
        "type": "object",
        "description": "FastAPI 422 validation error envelope.",
        "required": ["detail"],
        "properties": {
            "detail": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/ValidationErrorItem"},
            },
        },
    }

    return schemas


# ----------------------------------------------------------------------------
# Examples — reusable response/request examples
# ----------------------------------------------------------------------------
def _bbox_example(x1: int, y1: int, x2: int, y2: int) -> list[dict[str, int]]:
    """Build a 4-corner bbox example in [TL, TR, BR, BL] order."""
    return [
        {"x": x1, "y": y1},
        {"x": x2, "y": y1},
        {"x": x2, "y": y2},
        {"x": x1, "y": y2},
    ]


def _ocr_response_example() -> dict:
    """Realistic but trimmed sample (1 block / 1 line / 2 words)."""
    return {
        "request_id": "8a4c2b6e-1f3a-4d2c-9c11-7e58a2d4b6f0",
        "processing_time_ms": 612,
        "lang": "auto",
        "detected_lang": "en",
        "image_width": 640,
        "image_height": 480,
        "full_text": "Hello world",
        "blocks": [
            {
                "text": "Hello world",
                "bbox": _bbox_example(10, 12, 320, 48),
                "confidence": 0.973,
                "line_count": 1,
                "lines": [
                    {
                        "text": "Hello world",
                        "bbox": _bbox_example(10, 12, 320, 48),
                        "confidence": 0.973,
                        "words": [
                            {"text": "Hello", "bbox": _bbox_example(10, 12, 140, 48), "confidence": 0.973},
                            {"text": "world", "bbox": _bbox_example(160, 12, 320, 48), "confidence": 0.973},
                        ],
                    }
                ],
            }
        ],
        "reading_order": "ltr",
    }


def _ocr_request_examples() -> dict:
    return {
        "imageUrlAuto": {
            "summary": "Image URL + auto language",
            "description": "Most common case — let the server fetch the image and detect the language.",
            "value": {"image_url": SAMPLE_IMAGE_URL, "lang": "auto"},
        },
        "imageUrlManga": {
            "summary": "Manga mode (Japanese, RTL)",
            "description": "Specialised pipeline: PaddleOCR detection + manga-ocr recognition + speech-bubble clustering.",
            "value": {"image_url": SAMPLE_IMAGE_URL, "lang": "japan", "mode": "manga"},
        },
        "imageUrlRtl": {
            "summary": "Force right-to-left reading order",
            "description": "Useful for Arabic / Hebrew documents where auto-detection might be ambiguous.",
            "value": {"image_url": SAMPLE_IMAGE_URL, "lang": "ar", "reading_order": "rtl"},
        },
        "imageBase64": {
            "summary": "Inline base64 image",
            "description": "Send the image bytes directly (PNG/JPEG/WebP) base64-encoded. Recommended for private images.",
            "value": {
                "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
                "lang": "en",
            },
        },
    }


def _languages_response_example() -> dict:
    return {
        "count": 110,
        "languages": [
            {"code": "auto", "label": "Auto detect"},
            {"code": "en", "label": "English"},
            {"code": "vi", "label": "Vietnamese"},
            {"code": "japan", "label": "Japanese (Hiragana + Katakana + Kanji)"},
        ],
    }


# ----------------------------------------------------------------------------
# Reusable error responses
# ----------------------------------------------------------------------------
def _err_400() -> dict:
    return {
        "description": "Bad request — invalid input (e.g. base64 decode failed, empty image, bad URL scheme, unsupported lang).",
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                "example": {"detail": "lang 'xx' không hợp lệ. Cho phép: ['ar', 'ch', 'en', ...]"},
            }
        },
    }


def _err_413() -> dict:
    return {
        "description": "Payload too large — image exceeds the 10 MB limit (URL fetch or upload).",
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                "example": {"detail": "File 12582912 bytes vượt giới hạn 10485760"},
            }
        },
    }


def _err_422() -> dict:
    return {
        "description": "Validation error — request body failed Pydantic validation (missing/conflicting fields, wrong types).",
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/ValidationError"},
                "example": {
                    "detail": [
                        {
                            "loc": ["body"],
                            "msg": "Phải truyền 1 trong 2: `image` (base64) hoặc `image_url`",
                            "type": "value_error",
                        }
                    ]
                },
            }
        },
    }


def _err_502() -> dict:
    return {
        "description": "Upstream OCR engine failed (PaddleOCR or manga-ocr crashed). Safe to retry once.",
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                "example": {"detail": "OCR engine lỗi: CUDA out of memory"},
            }
        },
    }


# ----------------------------------------------------------------------------
# Path operations
# ----------------------------------------------------------------------------
def build_paths() -> dict:
    paths: dict = {}

    paths["/v1/languages"] = {
        "get": {
            "tags": ["Languages"],
            "operationId": "listSupportedLanguages",
            "summary": "List supported OCR languages",
            "description": (
                "Returns the flat list of every language code accepted by the `lang` field, "
                "each with its human-readable label. Convenient for client-side dropdowns / pickers.\n\n"
                "Includes the special `auto` alias that triggers language auto-detection."
            ),
            "responses": {
                "200": {
                    "description": "Supported languages catalog.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/LanguagesResponse"},
                            "example": _languages_response_example(),
                        }
                    },
                }
            },
        }
    }

    ocr_response_200 = {
        "description": "OCR succeeded. Response always carries the full ML Kit hierarchy (blocks > lines > words).",
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/OcrResponse"},
                "example": _ocr_response_example(),
            }
        },
    }

    paths["/v1/ocr"] = {
        "post": {
            "tags": ["OCR"],
            "operationId": "recognizeImage",
            "summary": "Recognize text from a JSON payload",
            "description": (
                "Run OCR against an image supplied either as a base64 string (`image`) or as an "
                "HTTP/HTTPS URL (`image_url`). Exactly one of the two fields must be provided.\n\n"
                "**Engines**\n"
                "- `mode=general` (default): PaddleOCR PP-OCRv5 with detection + recognition + "
                "  paragraph clustering. Supports 110 language codes.\n"
                "- `mode=manga`: specialised pipeline using PaddleOCR for detection and "
                "  [manga-ocr](https://github.com/kha-white/manga-ocr) for Japanese recognition, "
                "  with speech-bubble clustering and RTL reading order.\n\n"
                "**Limits**\n"
                "- URL-fetched images: 10 MB max, 15s fetch timeout.\n"
                "- Word-level boxes auto-degrade to `[]` for images larger than ~1.5M pixels."
            ),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/OcrRequest"},
                        "examples": _ocr_request_examples(),
                    }
                },
            },
            "responses": {
                "200": ocr_response_200,
                "400": _err_400(),
                "413": _err_413(),
                "422": _err_422(),
                "502": _err_502(),
            },
        }
    }

    paths["/v1/ocr/upload"] = {
        "post": {
            "tags": ["OCR"],
            "operationId": "uploadAndRecognize",
            "summary": "Recognize text from a multipart upload",
            "description": (
                "OCR endpoint that accepts a raw image file via `multipart/form-data`. Best for "
                "browser drag-and-drop tools, mobile uploads, or `curl -F` scripts where "
                "base64 encoding adds unnecessary overhead.\n\n"
                "Supports the same `lang`, `reading_order` and `mode` parameters as `POST /v1/ocr`."
            ),
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "Image file (PNG / JPEG / WebP). Max 10 MB.",
                                },
                                "lang": {
                                    "type": "string",
                                    "default": "auto",
                                    "description": "Same as the `lang` field of `POST /v1/ocr`. See `GET /v1/languages`.",
                                },
                                "reading_order": {
                                    "type": "string",
                                    "enum": ["ltr", "rtl", "auto"],
                                    "default": "auto",
                                },
                                "mode": {
                                    "type": "string",
                                    "enum": ["general", "manga"],
                                    "default": "general",
                                },
                                "request_id": {
                                    "type": "string",
                                    "nullable": True,
                                    "description": "Optional trace ID echoed in the response.",
                                },
                            },
                        },
                        "encoding": {
                            "file": {
                                "contentType": "image/png, image/jpeg, image/webp",
                            }
                        },
                    }
                },
            },
            "responses": {
                "200": ocr_response_200,
                "400": _err_400(),
                "413": _err_413(),
                "422": _err_422(),
                "502": _err_502(),
            },
        }
    }

    paths["/healthz/live"] = {
        "get": {
            "tags": ["Health"],
            "operationId": "livenessCheck",
            "summary": "Liveness probe",
            "description": "Lightweight check that the process is up. Does not exercise the OCR engine.",
            "security": [],  # Public — no auth required (RapidAPI proxy still injects keys but provider can bypass)
            "responses": {
                "200": {
                    "description": "Process is alive.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HealthResponse"},
                            "example": {"status": "ok"},
                        }
                    },
                }
            },
        }
    }

    paths["/healthz/ready"] = {
        "get": {
            "tags": ["Health"],
            "operationId": "readinessCheck",
            "summary": "Readiness probe",
            "description": "Indicates the OCR service has completed its warm-up (`en` model loaded) and can accept traffic.",
            "security": [],
            "responses": {
                "200": {
                    "description": "Service is ready.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HealthResponse"},
                            "example": {"status": "ok", "service": "ocr"},
                        }
                    },
                }
            },
        }
    }

    return paths


# ----------------------------------------------------------------------------
# Top-level spec assembly
# ----------------------------------------------------------------------------
def build_spec() -> dict:
    description = (
        "**OCR API** — production-grade OCR powered by **PaddleOCR PP-OCRv5** "
        "and **manga-ocr** (kha-white) for Japanese comics.\n\n"
        "## Highlights\n"
        "- 110 language codes across 12 script families (Latin, CJK, Cyrillic, Arabic, Devanagari, Thai, Greek, Tamil, Telugu, ...).\n"
        "- Two pipelines: `general` (PaddleOCR) and `manga` (PaddleOCR detection + manga-ocr recognition + speech-bubble clustering).\n"
        "- Always returns the full **ML Kit hierarchy**: blocks (paragraphs) > lines > words, each with confidence and 4-corner bbox.\n"
        "- Two ingest modes: JSON (base64 or URL) and multipart upload.\n"
        "- Auto language detection with parallel fallback when confidence drops below 0.85.\n"
        "- Auto reading-order detection (LTR / RTL) based on script family and page aspect.\n\n"
        "## Quick start\n"
        "1. Subscribe on RapidAPI to obtain your `X-RapidAPI-Key`.\n"
        "2. `POST /v1/ocr` with `{ \"image_url\": \"https://...\", \"lang\": \"auto\" }`.\n"
        "3. Read `full_text` for the concatenated result, or walk `blocks[].lines[].words[]` for layout."
    )

    spec: dict = {
        "openapi": OPENAPI_VERSION,
        "info": {
            "title": "OCR API",
            "version": SPEC_VERSION,
            "description": description,
            "termsOfService": "https://ocr.spacecloud.fit/terms",
            "contact": {
                "name": "OCR Support",
                "url": "https://ocr.spacecloud.fit",
                "email": "support@spacecloud.fit",
            },
            "license": {"name": "Proprietary"},
            # x-logo là vendor extension để RapidAPI / ReDoc render logo card.
            "x-logo": {
                "url": "https://ocr.spacecloud.fit/logo.png",
                "altText": "OCR API",
            },
        },
        "servers": [
            {"url": PROD_SERVER, "description": "Production server — direct"},
            {"url": RAPIDAPI_SERVER, "description": "RapidAPI gateway"},
        ],
        "tags": [
            {"name": "OCR", "description": "Recognize text from images."},
            {"name": "Languages", "description": "Metadata about supported languages and models."},
            {"name": "Health", "description": "Liveness and readiness probes."},
        ],
        "security": [
            # Cả 2 header này được RapidAPI inject vào mọi consumer request — phải khai báo
            # đồng thời để marketplace nhận diện đúng spec.
            {"RapidAPIKey": [], "RapidAPIHost": []},
        ],
        "paths": build_paths(),
        "components": {
            "schemas": build_schemas(),
            "securitySchemes": {
                "RapidAPIKey": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-RapidAPI-Key",
                    "description": "RapidAPI consumer key (injected automatically by the RapidAPI marketplace).",
                },
                "RapidAPIHost": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-RapidAPI-Host",
                    "description": "RapidAPI host identifier (injected automatically by the RapidAPI marketplace).",
                },
                "RapidAPIProxySecret": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-RapidAPI-Proxy-Secret",
                    "description": (
                        "Shared secret injected by the RapidAPI proxy and verified by the provider. "
                        "Configured in the RapidAPI provider dashboard; not exposed to consumers."
                    ),
                },
            },
        },
    }

    return spec


# ----------------------------------------------------------------------------
# YAML rendering helper — block style for readability
# ----------------------------------------------------------------------------
def _yaml_dump(spec: dict) -> str:
    """Block-style YAML, không sort keys (giữ thứ tự semantic info → servers → paths → components)."""
    return yaml.safe_dump(
        spec,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=120,
    )


def main() -> None:
    spec = build_spec()

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(_yaml_dump(spec), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    n_paths = len(spec["paths"])
    n_schemas = len(spec["components"]["schemas"])
    print(f"✓ Wrote {OUT_YAML}  ({OUT_YAML.stat().st_size:,} bytes)")
    print(f"✓ Wrote {OUT_JSON}  ({OUT_JSON.stat().st_size:,} bytes)")
    print(f"  openapi:  {spec['openapi']}")
    print(f"  paths:    {n_paths}")
    print(f"  schemas:  {n_schemas}")
    print(f"  servers:  {len(spec['servers'])}")


if __name__ == "__main__":
    main()
