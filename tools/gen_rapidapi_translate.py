#!/usr/bin/env python3
"""Sinh OpenAPI 3.0.3 spec dành RIÊNG cho RapidAPI marketplace.

Khác với gen_translate_openapi.py (full spec internal có /healthz, /v1/models):
- CHỈ business endpoints: /v1/translate, /v1/json, /v1/translate-html,
  /v1/translate-json, /v1/dict.
- KHÔNG include security scheme cho RapidAPI header (RapidAPI tự inject
  X-RapidAPI-Key, X-RapidAPI-Host khi forward request).
- KHÔNG include health/info endpoints (không có giá trị marketing).
- Examples đầy đủ request + response thật từ smoke test live.
- Mỗi endpoint có description marketing-grade + use case.
- termsOfService, contact, license, x-logo đầy đủ.

Run:    python3 tools/gen_rapidapi_translate.py
Output: postman/translate-rapidapi.yaml + postman/translate-rapidapi.json
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
OUT_YAML = ROOT / "postman" / "translate-rapidapi.yaml"
OUT_JSON = ROOT / "postman" / "translate-rapidapi.json"

OPENAPI_VERSION = "3.0.3"
SPEC_VERSION = "1.0.0"

# Production direct URL — RapidAPI sẽ proxy tới đây qua gateway của họ.
PROD_SERVER = "https://translate.spacecloud.fit"


# ============================================================================
# Schemas
# ============================================================================
def build_schemas() -> dict:
    schemas: dict = {}

    # ---- Translate (single + batch) -----------------------------------------
    schemas["TranslateRequest"] = {
        "type": "object",
        "required": ["text", "target_lang"],
        "properties": {
            "text": {
                "description": (
                    "Text to translate. Either a single string or an array of strings "
                    "(up to 100 items, each up to 50 000 characters)."
                ),
                "oneOf": [
                    {"type": "string", "minLength": 1, "maxLength": 50000},
                    {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 100,
                        "items": {"type": "string", "minLength": 1, "maxLength": 50000},
                    },
                ],
                "example": "Hello world",
            },
            "source_lang": {
                "type": "string",
                "default": "auto",
                "pattern": "^(auto|[a-z]{2,3}(-[A-Z]{2})?)$",
                "description": (
                    "Source language. `auto` = the model detects per item. "
                    "Otherwise an ISO 639-1 code (`en`, `vi`, `ja`, `zh`, `ko`, `fr`, "
                    "`de`, `es`, `ru`, `th`, `id`, `pt`, `it`, `ar`, `hi`, ...)."
                ),
                "example": "auto",
            },
            "target_lang": {
                "type": "string",
                "minLength": 2,
                "maxLength": 8,
                "pattern": "^[a-z]{2,3}(-[A-Z]{2})?$",
                "description": "Required ISO 639-1 target language. Cannot be `auto`.",
                "example": "vi",
            },
        },
    }

    schemas["TranslationDetected"] = {
        "type": "object",
        "description": (
            "Returned when `source_lang=auto` — each item carries its own detected language."
        ),
        "required": ["translated_text", "detected_source_lang"],
        "properties": {
            "translated_text": {"type": "string"},
            "detected_source_lang": {
                "type": "string",
                "description": "ISO 639-1 detected from the source text.",
                "example": "en",
            },
        },
    }

    _translations_field = {
        "description": (
            "Translations in the same order as the input. "
            "When `source_lang=auto`, each item is `{translated_text, detected_source_lang}`. "
            "When an explicit `source_lang` is given, items are plain strings."
        ),
        "oneOf": [
            {"type": "array", "items": {"$ref": "#/components/schemas/TranslationDetected"}},
            {"type": "array", "items": {"type": "string"}},
        ],
    }

    schemas["TranslateResponse"] = {
        "type": "object",
        "required": ["request_id", "processing_time_ms", "translations"],
        "properties": {
            "request_id": {"type": "string", "format": "uuid"},
            "processing_time_ms": {"type": "integer", "minimum": 0},
            "translations": _translations_field,
        },
    }

    # ---- JSON i18n batch ---------------------------------------------------
    schemas["JsonTranslateRequest"] = {
        "type": "object",
        "required": ["texts", "target_lang"],
        "properties": {
            "texts": {
                "type": "array",
                "minItems": 1,
                "maxItems": 100,
                "items": {"type": "string", "minLength": 1, "maxLength": 50000},
                "description": "Array of strings to translate (up to 100 items, 50 000 chars each).",
                "example": ["Welcome", "Sign in", "Sign up"],
            },
            "source_lang": {
                "type": "string",
                "default": "auto",
                "pattern": "^(auto|[a-z]{2,3}(-[A-Z]{2})?)$",
                "example": "en",
            },
            "target_lang": {
                "type": "string",
                "minLength": 2,
                "maxLength": 8,
                "pattern": "^[a-z]{2,3}(-[A-Z]{2})?$",
                "example": "ja",
            },
        },
    }

    schemas["JsonTranslateResponse"] = {
        "type": "object",
        "required": ["request_id", "processing_time_ms", "translations"],
        "properties": {
            "request_id": {"type": "string", "format": "uuid"},
            "processing_time_ms": {"type": "integer", "minimum": 0},
            "translations": _translations_field,
        },
    }

    # ---- Translate HTML ----------------------------------------------------
    schemas["TranslateHtmlRequest"] = {
        "type": "object",
        "required": ["html", "target_lang"],
        "properties": {
            "html": {
                "type": "string",
                "minLength": 1,
                "maxLength": 20000000,
                "description": (
                    "HTML to translate (up to 20 MB / 20 million characters). "
                    "Tag structure, attributes and inline formatting are preserved."
                ),
                "example": "<p>Hello <b>world</b>!</p>",
            },
            "source_lang": {
                "type": "string",
                "default": "auto",
                "pattern": "^(auto|[a-z]{2,3}(-[A-Z]{2})?)$",
                "example": "en",
            },
            "target_lang": {
                "type": "string",
                "minLength": 2,
                "maxLength": 8,
                "pattern": "^[a-z]{2,3}(-[A-Z]{2})?$",
                "example": "vi",
            },
            "translate_attributes": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Translate `alt`, `title`, `aria-label`, `placeholder` attributes and "
                    "`<meta name=\"description\">` content. Set false to translate text nodes only."
                ),
            },
            "ignore_terms": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 200},
                "maxItems": 500,
                "default": [],
                "description": (
                    "Words or multi-word phrases to keep verbatim — typically brand names, "
                    "product names, or technical jargon. Word-boundary match. Default case-sensitive."
                ),
                "example": ["Apple", "MacBook Pro", "iPhone 15 Pro Max"],
            },
            "ignore_case": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Case-insensitive match for `ignore_terms`. Default false (so `Apple` ≠ `apple`)."
                ),
            },
        },
    }

    schemas["HtmlHealth"] = {
        "type": "object",
        "required": ["health", "error_rate", "structure_diff", "errors_total", "fatals_total", "parse_tier"],
        "properties": {
            "health": {
                "type": "string",
                "enum": ["clean", "minor", "moderate"],
                "description": (
                    "Health classification of input HTML. `severe` triggers 422 (and never appears here)."
                ),
            },
            "error_rate": {
                "type": "number",
                "format": "float",
                "minimum": 0.0,
                "description": "Weighted parser-error rate (fatal × 5) per tag.",
            },
            "structure_diff": {
                "type": "number",
                "format": "float",
                "minimum": 0.0,
                "description": "|raw_open_tags − parsed_tags| / raw_open_tags.",
            },
            "errors_total": {"type": "integer", "minimum": 0},
            "fatals_total": {"type": "integer", "minimum": 0},
            "parse_tier": {
                "type": "string",
                "enum": ["lxml", "html5lib", "fragment_wrap"],
                "description": (
                    "Parser used: `lxml` (fast path), `html5lib` (browser-grade fallback), "
                    "or `fragment_wrap` (auto-wrapped fragment)."
                ),
            },
        },
    }

    schemas["TranslateHtmlResponse"] = {
        "type": "object",
        "required": [
            "request_id",
            "processing_time_ms",
            "html",
            "health",
            "segments_translated",
            "chars_translated",
            "warnings",
        ],
        "properties": {
            "request_id": {"type": "string", "format": "uuid"},
            "processing_time_ms": {"type": "integer", "minimum": 0},
            "html": {"type": "string", "description": "Translated HTML, structure preserved."},
            "detected_source_lang": {
                "type": "string",
                "nullable": True,
                "description": "Dominant detected language when `source_lang=auto`.",
                "example": "en",
            },
            "health": {"$ref": "#/components/schemas/HtmlHealth"},
            "segments_translated": {"type": "integer", "minimum": 0},
            "chars_translated": {"type": "integer", "minimum": 0},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }

    schemas["HtmlTooMalformedError"] = {
        "type": "object",
        "required": ["detail"],
        "properties": {
            "detail": {
                "type": "object",
                "required": ["error", "health", "metrics"],
                "properties": {
                    "error": {"type": "string", "enum": ["html_too_malformed"]},
                    "health": {"type": "string", "enum": ["severe"]},
                    "metrics": {
                        "type": "object",
                        "properties": {
                            "error_rate": {"type": "number"},
                            "structure_diff": {"type": "number"},
                            "errors_total": {"type": "integer"},
                            "fatals_total": {"type": "integer"},
                        },
                    },
                    "errors_sample": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "line": {"type": "integer"},
                                "column": {"type": "integer"},
                                "severity": {"type": "string"},
                                "message": {"type": "string"},
                            },
                        },
                    },
                    "fatal_markers": {"type": "array", "items": {"type": "string"}},
                    "suggestion": {"type": "string"},
                },
            },
        },
    }

    # ---- Translate JSON object ---------------------------------------------
    schemas["TranslateJsonObjectRequest"] = {
        "type": "object",
        "required": ["json_data", "target_lang"],
        "properties": {
            "json_data": {
                "description": (
                    "JSON to translate — object, array or string. All string values are "
                    "translated unless excluded by the rules below."
                ),
                "oneOf": [{"type": "object"}, {"type": "array"}, {"type": "string"}],
            },
            "source_lang": {
                "type": "string",
                "default": "auto",
                "pattern": "^(auto|[a-z]{2,3}(-[A-Z]{2})?)$",
                "example": "en",
            },
            "target_lang": {
                "type": "string",
                "minLength": 2,
                "maxLength": 8,
                "pattern": "^[a-z]{2,3}(-[A-Z]{2})?$",
                "example": "vi",
            },
            "words_not_to_translate": {
                "description": (
                    "Words/phrases to keep verbatim inside translated text. Accepts a "
                    "string with `;` separator or an array of strings. Word-boundary match, "
                    "default case-sensitive."
                ),
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "example": "Earbuds; New York",
            },
            "paths_to_exclude": {
                "description": (
                    "Dot-notation JSON paths to skip entirely. `*` matches any array index. "
                    "A pattern matches the path AND its subtree (e.g. `product.media` skips "
                    "everything under `product.media`). Accepts `;`-separated string or list."
                ),
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "example": "product.media.img_desc; items.*.image_url",
            },
            "common_keys_to_exclude": {
                "description": (
                    "Key names to skip at any nesting depth (e.g. `name; price`)."
                    " Accepts `;`-separated string or list."
                ),
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "example": "name; price",
            },
            "ignore_case": {
                "type": "boolean",
                "default": False,
                "description": "Case-insensitive match for `words_not_to_translate`.",
            },
            "skip_non_text": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Auto-skip strings that are not human text: pure numbers, currency, "
                    "percent, URLs, emails, UUIDs, hashes, ISO dates, code/IDs."
                ),
            },
        },
    }

    schemas["JsonTranslationStats"] = {
        "type": "object",
        "required": ["strings_translated", "strings_skipped", "chars_translated"],
        "properties": {
            "strings_translated": {"type": "integer", "minimum": 0},
            "strings_skipped": {
                "type": "integer",
                "minimum": 0,
                "description": "Strings auto-skipped by the non-text filter.",
            },
            "chars_translated": {"type": "integer", "minimum": 0},
        },
    }

    schemas["TranslateJsonObjectResponse"] = {
        "type": "object",
        "required": ["request_id", "processing_time_ms", "json_data", "stats", "warnings"],
        "properties": {
            "request_id": {"type": "string", "format": "uuid"},
            "processing_time_ms": {"type": "integer", "minimum": 0},
            "json_data": {
                "description": "Translated JSON, structure preserved.",
                "oneOf": [{"type": "object"}, {"type": "array"}, {"type": "string"}],
            },
            "detected_source_lang": {
                "type": "string",
                "nullable": True,
                "example": "en",
            },
            "stats": {"$ref": "#/components/schemas/JsonTranslationStats"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }

    # ---- Dictionary --------------------------------------------------------
    schemas["DictRequest"] = {
        "type": "object",
        "required": ["word", "targetLang"],
        "properties": {
            "word": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
                "description": "Word or phrase to look up (in `targetLang` — the language being learned).",
                "example": "book",
            },
            "nativeLang": {
                "type": "string",
                "default": "vi",
                "minLength": 2,
                "maxLength": 8,
                "pattern": "^[a-z]{2,3}(-[A-Z]{2})?$",
                "description": "User's native language — meanings/explanations are emitted in this language.",
                "example": "vi",
            },
            "targetLang": {
                "type": "string",
                "minLength": 2,
                "maxLength": 8,
                "pattern": "^[a-z]{2,3}(-[A-Z]{2})?$",
                "description": "Language being learned — the language of the input `word`.",
                "example": "en",
            },
        },
    }

    schemas["DictPhonetic"] = {
        "type": "object",
        "properties": {
            "ipa": {
                "type": "string",
                "nullable": True,
                "description": "IPA transcription. Null when not available.",
                "example": "/bʊk/",
            },
            "romanization": {
                "type": "string",
                "nullable": True,
                "description": (
                    "Romanization for non-Latin scripts (pinyin, romaji, revised romanization, ...)."
                    " Null for Latin-script words."
                ),
                "example": None,
            },
        },
    }

    schemas["DictDefinition"] = {
        "type": "object",
        "required": ["partOfSpeech", "meaning"],
        "properties": {
            "partOfSpeech": {
                "type": "string",
                "description": (
                    "One of: `noun`, `verb`, `adjective`, `adverb`, `preposition`, `conjunction`, "
                    "`interjection`, `pronoun`, `determiner`, `particle`, `phrase`, `idiom`."
                ),
                "example": "noun",
            },
            "meaning": {"type": "string", "description": "Concise definition in `nativeLang`."},
        },
    }

    schemas["DictExample"] = {
        "type": "object",
        "required": ["text", "meaning"],
        "properties": {
            "text": {"type": "string", "description": "Sentence/phrase in `targetLang`."},
            "meaning": {"type": "string", "description": "Translation in `nativeLang`."},
        },
    }

    schemas["DictWordRef"] = {
        "type": "object",
        "required": ["text", "meaning"],
        "properties": {
            "text": {"type": "string"},
            "meaning": {"type": "string"},
        },
    }

    schemas["DictRelated"] = {
        "type": "object",
        "properties": {
            "synonyms": {"type": "array", "items": {"$ref": "#/components/schemas/DictWordRef"}},
            "antonyms": {"type": "array", "items": {"$ref": "#/components/schemas/DictWordRef"}},
            "relatedWords": {"type": "array", "items": {"$ref": "#/components/schemas/DictWordRef"}},
            "memoryTips": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1–3 short mnemonic tips written in `nativeLang`.",
            },
        },
    }

    schemas["DictResponse"] = {
        "type": "object",
        "required": [
            "request_id",
            "processing_time_ms",
            "model_used",
            "word",
            "nativeLang",
            "targetLang",
            "phonetic",
            "shortMeaning",
            "definitions",
            "examples",
            "related",
        ],
        "properties": {
            "request_id": {"type": "string", "format": "uuid"},
            "processing_time_ms": {"type": "integer", "minimum": 0},
            "model_used": {"type": "string"},
            "word": {"type": "string"},
            "nativeLang": {"type": "string"},
            "targetLang": {"type": "string"},
            "phonetic": {"$ref": "#/components/schemas/DictPhonetic"},
            "shortMeaning": {
                "type": "string",
                "description": "One-line meaning in `nativeLang`.",
            },
            "definitions": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DictDefinition"},
            },
            "examples": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DictExample"},
            },
            "phrases": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DictExample"},
                "description": "0–4 common phrases / collocations.",
            },
            "related": {"$ref": "#/components/schemas/DictRelated"},
        },
    }

    # ---- Generic error -----------------------------------------------------
    schemas["ErrorResponse"] = {
        "type": "object",
        "required": ["detail"],
        "properties": {
            "detail": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "object"}},
                    {"type": "object"},
                ],
                "description": (
                    "Error detail. String for upstream errors; array for FastAPI validation errors; "
                    "object for structured errors (e.g. `html_too_malformed`)."
                ),
            },
        },
    }

    return schemas


# ============================================================================
# Examples — request + response thật từ smoke test live
# ============================================================================

# /v1/translate
TRANSLATE_REQ_EX = {
    "single_auto": {
        "summary": "Single text + auto detect",
        "value": {"text": "Hello world", "target_lang": "vi"},
    },
    "single_explicit": {
        "summary": "Single text + explicit source",
        "value": {"text": "Tôi yêu Việt Nam", "source_lang": "vi", "target_lang": "en"},
    },
    "batch_explicit": {
        "summary": "Batch + explicit source (en→vi)",
        "value": {
            "text": ["Good morning", "Thank you", "How are you?"],
            "source_lang": "en",
            "target_lang": "vi",
        },
    },
    "batch_auto_mixed": {
        "summary": "Batch + auto detect on mixed input",
        "value": {"text": ["こんにちは", "안녕하세요", "Bonjour"], "target_lang": "vi"},
    },
}

TRANSLATE_RESP_EX = {
    "auto_with_detected": {
        "summary": "Auto: each item carries detected_source_lang",
        "value": {
            "request_id": "03f0ece3-9ad8-4b52-ad0c-40da09487722",
            "processing_time_ms": 126,
            "translations": [
                {"translated_text": "Xin chào thế giới", "detected_source_lang": "en"},
            ],
        },
    },
    "explicit_strings": {
        "summary": "Explicit source: list of strings",
        "value": {
            "request_id": "f4bfc370-747e-40c7-a3c0-eb1cb86683af",
            "processing_time_ms": 86,
            "translations": ["Chào buổi sáng", "Cảm ơn", "Bạn khỏe không?"],
        },
    },
    "auto_mixed_batch": {
        "summary": "Auto mixed batch",
        "value": {
            "request_id": "263aafcd-ea31-423e-bd4b-1591d2f6cbe3",
            "processing_time_ms": 192,
            "translations": [
                {"translated_text": "Xin chào", "detected_source_lang": "ja"},
                {"translated_text": "Xin chào", "detected_source_lang": "ko"},
                {"translated_text": "Xin chào", "detected_source_lang": "fr"},
            ],
        },
    },
}

# /v1/json
JSON_REQ_EX = {
    "i18n_explicit": {
        "summary": "i18n keys (en→ja)",
        "value": {
            "texts": ["Welcome", "Sign in", "Sign up", "Forgot password?"],
            "source_lang": "en",
            "target_lang": "ja",
        },
    },
    "i18n_auto": {
        "summary": "i18n auto detect",
        "value": {"texts": ["Welcome", "ようこそ", "환영합니다"], "target_lang": "vi"},
    },
}

JSON_RESP_EX = {
    "explicit_strings": {
        "summary": "Explicit output (en→ja)",
        "value": {
            "request_id": "8a24186d-df3b-4843-8caf-63702cc1e84c",
            "processing_time_ms": 71,
            "translations": ["ようこそ", "サインイン", "サインアップ", "パスワードをお忘れですか？"],
        },
    },
}

# /v1/translate-html
HTML_REQ_EX = {
    "simple_inline": {
        "summary": "Simple HTML with inline tags",
        "value": {
            "html": "<p>Hello <b>red</b> car!</p>",
            "target_lang": "vi",
        },
    },
    "ignore_terms_brands": {
        "summary": "Brand names preserved (ignore_terms)",
        "value": {
            "html": "<p>Apple makes the new MacBook Pro and iPhone. Buy now!</p>",
            "source_lang": "en",
            "target_lang": "vi",
            "ignore_terms": ["Apple", "MacBook Pro", "iPhone"],
        },
    },
    "attributes_alt_title": {
        "summary": "Translate alt + title attributes",
        "value": {
            "html": '<img src="cat.jpg" alt="A picture of a cat" title="Click to enlarge"><p>Image gallery</p>',
            "source_lang": "en",
            "target_lang": "vi",
        },
    },
    "skip_script_code": {
        "summary": "Skip <script>, <code>, <pre>, ...",
        "value": {
            "html": '<p>Use <code>print()</code> to debug.</p><script>console.log("skip")</script>',
            "source_lang": "en",
            "target_lang": "vi",
        },
    },
    "article_full": {
        "summary": "Full article with nested inline + <ul>",
        "value": {
            "html": (
                "<article><h1>The Future of AI</h1>"
                "<p>Artificial intelligence is changing the world. <b>Machine learning</b> models "
                "like GPT and Claude are revolutionizing how we interact with computers.</p>"
                "<p>However, <em>responsible AI</em> requires careful consideration of "
                '<a href="/ethics">ethics</a> and <a href="/safety">safety</a>.</p>'
                "<ul><li>Item one</li><li>Item two</li></ul></article>"
            ),
            "source_lang": "en",
            "target_lang": "vi",
        },
    },
}

HTML_RESP_EX = {
    "simple_inline": {
        "summary": "Inline tags preserved",
        "value": {
            "request_id": "e3a1109d-4dda-4c4e-a207-8e7a4efe7d85",
            "processing_time_ms": 192,
            "html": "<p>Chào xe <b>đỏ</b>!</p>",
            "detected_source_lang": "en",
            "health": {
                "health": "clean",
                "error_rate": 0.0,
                "structure_diff": 0.0,
                "errors_total": 0,
                "fatals_total": 0,
                "parse_tier": "fragment_wrap",
            },
            "segments_translated": 1,
            "chars_translated": 28,
            "warnings": [],
        },
    },
    "ignore_terms": {
        "summary": "Brand names kept verbatim",
        "value": {
            "request_id": "e01239ca-ac36-4cea-a9f7-628b046f3e2b",
            "processing_time_ms": 113,
            "html": "<p>Apple tạo ra MacBook Pro mới và iPhone. Mua ngay!</p>",
            "detected_source_lang": "en",
            "health": {
                "health": "clean",
                "error_rate": 0.0,
                "structure_diff": 0.0,
                "errors_total": 0,
                "fatals_total": 0,
                "parse_tier": "fragment_wrap",
            },
            "segments_translated": 1,
            "chars_translated": 56,
            "warnings": [],
        },
    },
    "article_full": {
        "summary": "Full article, structure preserved",
        "value": {
            "request_id": "abc123de-4567-89f0-abcd-ef1234567890",
            "processing_time_ms": 461,
            "html": (
                "<article><h1>Tương lai của trí tuệ nhân tạo</h1>"
                "<p>Trí tuệ nhân tạo đang thay đổi thế giới. Các mô hình <b>machine learning</b> "
                "như GPT và Claude đang cách mạng hóa cách chúng ta tương tác với máy tính.</p>"
                "<p>Tuy nhiên, <em>trí tuệ nhân tạo có trách nhiệm</em> đòi hỏi phải cân nhắc "
                'cẩn trọng đến <a href="/ethics">đạo đức</a> và <a href="/safety">an toàn</a>.</p>'
                "<ul><li>Mục một</li><li>Mục hai</li></ul></article>"
            ),
            "detected_source_lang": "en",
            "health": {
                "health": "clean",
                "error_rate": 0.0,
                "structure_diff": 0.0,
                "errors_total": 0,
                "fatals_total": 0,
                "parse_tier": "fragment_wrap",
            },
            "segments_translated": 6,
            "chars_translated": 320,
            "warnings": [],
        },
    },
}

HTML_ERR_EX = {
    "severe_malformed": {
        "summary": "HTML too malformed → 422 with details",
        "value": {
            "detail": {
                "error": "html_too_malformed",
                "health": "severe",
                "metrics": {
                    "error_rate": 0.75,
                    "structure_diff": 0.0,
                    "errors_total": 6,
                    "fatals_total": 0,
                },
                "errors_sample": [
                    {
                        "line": 1,
                        "column": 77,
                        "severity": "error",
                        "message": "Opening and ending tag mismatch: b and i",
                    },
                ],
                "fatal_markers": [],
                "suggestion": "Run HTML through a sanitizer (e.g. `tidy -q -m -ashtml` or `html-minifier-terser`) before retry.",
            },
        },
    },
}

# /v1/translate-json
JSON_OBJ_REQ_EX = {
    "all_three_options_string": {
        "summary": "All 3 exclusion options (string format with `;`)",
        "value": {
            "json_data": {
                "title": "Premium Wireless Earbuds",
                "price": 99.99,
                "product": {
                    "name": "Earbuds Pro",
                    "description": "Best quality from New York",
                    "media": {
                        "img_desc": "Detailed product photo",
                        "title": "Main image",
                    },
                },
                "items": [
                    {"name": "Item A", "image_url": "https://x.com/a.jpg", "desc": "Item A description"},
                    {"name": "Item B", "image_url": "https://x.com/b.jpg", "desc": "Item B description"},
                ],
            },
            "source_lang": "en",
            "target_lang": "vi",
            "words_not_to_translate": "Earbuds; New York",
            "paths_to_exclude": "product.media.img_desc; items.*.image_url",
            "common_keys_to_exclude": "name; price",
        },
    },
    "list_format_options": {
        "summary": "Same options as list[str]",
        "value": {
            "json_data": {"title": "Apple makes Earbuds Pro"},
            "target_lang": "vi",
            "words_not_to_translate": ["Apple", "Earbuds"],
        },
    },
    "ecommerce_catalog": {
        "summary": "E-commerce catalog (multiple products)",
        "value": {
            "json_data": {
                "category": "Electronics",
                "products": [
                    {
                        "sku": "PROD-0001",
                        "name": "Wireless Mouse",
                        "price": 29.99,
                        "description": "Ergonomic wireless mouse with long battery life",
                    },
                    {
                        "sku": "PROD-0002",
                        "name": "Mechanical Keyboard",
                        "price": 89.99,
                        "description": "RGB backlit mechanical keyboard with cherry MX switches",
                    },
                ],
            },
            "target_lang": "vi",
            "common_keys_to_exclude": "sku; price",
        },
    },
}

JSON_OBJ_RESP_EX = {
    "all_three_options": {
        "summary": "All 3 exclusions applied",
        "value": {
            "request_id": "fa0e1d22-5a3c-4c7e-9f1d-2b1e8d7a4c33",
            "processing_time_ms": 461,
            "json_data": {
                "title": "Không dây cao cấp Earbuds",
                "price": 99.99,
                "product": {
                    "name": "Earbuds Pro",
                    "description": "Chất lượng tốt nhất từ New York",
                    "media": {
                        "img_desc": "Detailed product photo",
                        "title": "Hình ảnh chính",
                    },
                },
                "items": [
                    {"name": "Item A", "image_url": "https://x.com/a.jpg", "desc": "Mô tả mục A"},
                    {"name": "Item B", "image_url": "https://x.com/b.jpg", "desc": "Mô tả mục B"},
                ],
            },
            "detected_source_lang": "en",
            "stats": {"strings_translated": 5, "strings_skipped": 0, "chars_translated": 99},
            "warnings": [],
        },
    },
    "ecommerce_catalog": {
        "summary": "E-commerce catalog translated",
        "value": {
            "request_id": "b1c2d3e4-f5a6-7b8c-9d0e-1f2a3b4c5d6e",
            "processing_time_ms": 312,
            "json_data": {
                "category": "Điện tử",
                "products": [
                    {
                        "sku": "PROD-0001",
                        "name": "Wireless Mouse",
                        "price": 29.99,
                        "description": "Chuột không dây ergonomic với thời lượng pin dài",
                    },
                    {
                        "sku": "PROD-0002",
                        "name": "Mechanical Keyboard",
                        "price": 89.99,
                        "description": "Bàn phím cơ có đèn nền RGB với switch cherry MX",
                    },
                ],
            },
            "detected_source_lang": "en",
            "stats": {"strings_translated": 3, "strings_skipped": 0, "chars_translated": 138},
            "warnings": [],
        },
    },
}

# /v1/dict
DICT_REQ_EX = {
    "vi_learning_en_book": {
        "summary": "VI speaker learning EN — 'book'",
        "value": {"word": "book", "nativeLang": "vi", "targetLang": "en"},
    },
    "vi_learning_ja_kanji": {
        "summary": "VI speaker learning JA — '本'",
        "value": {"word": "本", "nativeLang": "vi", "targetLang": "ja"},
    },
    "en_learning_vi": {
        "summary": "EN speaker learning VI — 'tự do'",
        "value": {"word": "tự do", "nativeLang": "en", "targetLang": "vi"},
    },
}

DICT_RESP_EX = {
    "vi_learning_en_book": {
        "summary": "Full dictionary entry — VI learning EN word 'book'",
        "value": {
            "request_id": "fa0e1d22-5a3c-4c7e-9f1d-2b1e8d7a4c33",
            "processing_time_ms": 2640,
            "model_used": "translator-3a9f2c10",
            "word": "book",
            "nativeLang": "vi",
            "targetLang": "en",
            "phonetic": {"ipa": "/bʊk/", "romanization": None},
            "shortMeaning": "quyển sách",
            "definitions": [
                {"partOfSpeech": "noun", "meaning": "vật phẩm gồm các trang giấy đóng lại với nhau, dùng để đọc"},
                {"partOfSpeech": "verb", "meaning": "đặt chỗ trước (nhà hàng, vé, phòng)"},
            ],
            "examples": [
                {"text": "I bought a book yesterday.", "meaning": "Tôi đã mua một quyển sách hôm qua."},
                {"text": "Please book a table for two.", "meaning": "Làm ơn đặt bàn cho hai người."},
            ],
            "phrases": [
                {"text": "by the book", "meaning": "đúng theo quy tắc"},
                {"text": "book club", "meaning": "câu lạc bộ đọc sách"},
            ],
            "related": {
                "synonyms": [{"text": "volume", "meaning": "tập sách"}],
                "antonyms": [],
                "relatedWords": [{"text": "novel", "meaning": "tiểu thuyết"}],
                "memoryTips": ["'book' nghe gần với 'búc' — 'búc cuốn sách'"],
            },
        },
    },
}


# ============================================================================
# Paths
# ============================================================================
def _err_responses() -> dict:
    return {
        "422": {
            "description": "Validation error.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                },
            },
        },
        "502": {
            "description": "Upstream LLM error.",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                },
            },
        },
    }


def build_paths() -> dict:
    paths: dict = {}

    paths["/v1/translate"] = {
        "post": {
            "tags": ["Translation"],
            "summary": "Translate text — single string or batch.",
            "description": (
                "**Use case:** translate a single sentence, paragraph or up to 100 strings in one call.\n\n"
                "**Behavior:** when `source_lang=auto`, every item carries its own "
                "`detected_source_lang` — useful for mixed-language batches. With an explicit "
                "`source_lang`, the response is a plain `list[str]` for compact integration.\n\n"
                "**Supported languages:** Vietnamese, English, Japanese, Chinese (Simplified + "
                "Traditional), Korean, French, German, Spanish, Russian, Thai, Indonesian, "
                "Portuguese, Italian, Arabic, Hindi and more.\n\n"
                "**Quality:** powered by Qwen3-14B-AWQ on vLLM with idiom-aware prompting, "
                "script-purity post-validation and single-pass retry on mixed-script outputs.\n\n"
                "**Latency:** ~80–200 ms per item (single text), batches benefit from continuous "
                "batching on the GPU."
            ),
            "operationId": "translate",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/TranslateRequest"},
                        "examples": TRANSLATE_REQ_EX,
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "Translation result.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/TranslateResponse"},
                            "examples": TRANSLATE_RESP_EX,
                        },
                    },
                },
                **_err_responses(),
            },
        },
    }

    paths["/v1/json"] = {
        "post": {
            "tags": ["Translation"],
            "summary": "Translate an array of strings (i18n batch).",
            "description": (
                "**Use case:** translate localization key-strings for an app's i18n bundle. "
                "Same response shape as `/v1/translate`, but the request always uses an array, "
                "matching how i18n libraries (i18next, FormatJS, ICU, ...) store keys.\n\n"
                "**Order is preserved** — `translations[i]` corresponds to `texts[i]`."
            ),
            "operationId": "translateJson",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/JsonTranslateRequest"},
                        "examples": JSON_REQ_EX,
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "Translations in input order.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/JsonTranslateResponse"},
                            "examples": JSON_RESP_EX,
                        },
                    },
                },
                **_err_responses(),
            },
        },
    }

    paths["/v1/translate-html"] = {
        "post": {
            "tags": ["Translation"],
            "summary": "Translate HTML — preserve structure, ignore brand names.",
            "description": (
                "**Use case:** translate articles, product descriptions, emails, CMS pages or any "
                "HTML where tags, attributes and inline formatting must be preserved verbatim.\n\n"
                "**How it works:**\n"
                "1. Parse with `lxml` (fast path) → fall back to `html5lib` (browser-grade) when "
                "errors are heavy.\n"
                "2. Score health into `clean` / `minor` / `moderate` / `severe`. "
                "Severely malformed HTML returns **422** asking the caller to fix the HTML first.\n"
                "3. Walk the DOM. Skip `<script>`, `<style>`, `<noscript>`, `<svg>`, `<math>`, "
                "`<template>`, `<textarea>`, `<pre>`, `<code>`, `<kbd>`, `<samp>`, `<var>`.\n"
                "4. Build inline-aware segments at leaf-block level using XLIFF-style placeholders "
                "so inline `<b>`, `<i>`, `<a>` etc. survive any reordering by the model — "
                "critical when target word order differs from source.\n"
                "5. `ignore_terms` — replace exact word-boundary matches with HTML void-tag "
                "placeholders; restored verbatim after translation. Use this for brand names, "
                "product names and technical jargon.\n"
                "6. Translate all segments in parallel via vLLM continuous batching.\n"
                "7. Re-insert into DOM, serialize, and verify tag count consistency.\n\n"
                "**Translatable attributes** (when `translate_attributes=true`, default): `alt`, "
                "`title`, `aria-label`, `placeholder`, `<meta name=\"description\">` content. "
                "URLs (`href`, `src`), classes, IDs and data-attributes are never touched.\n\n"
                "**Limits:** up to 20 MB input. Throughput ~40 segments/sec — a 5 000-segment "
                "article (~700 KB) takes ~2 minutes. **Set your client HTTP timeout accordingly.**"
            ),
            "operationId": "translateHtml",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/TranslateHtmlRequest"},
                        "examples": HTML_REQ_EX,
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "Translated HTML (clean / minor / moderate health).",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/TranslateHtmlResponse"},
                            "examples": HTML_RESP_EX,
                        },
                    },
                },
                "422": {
                    "description": "HTML too malformed (`severe`) — caller must fix HTML before retry.",
                    "content": {
                        "application/json": {
                            "schema": {
                                "oneOf": [
                                    {"$ref": "#/components/schemas/HtmlTooMalformedError"},
                                    {"$ref": "#/components/schemas/ErrorResponse"},
                                ],
                            },
                            "examples": HTML_ERR_EX,
                        },
                    },
                },
                "502": {
                    "description": "Upstream LLM error.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        },
                    },
                },
            },
        },
    }

    paths["/v1/translate-json"] = {
        "post": {
            "tags": ["Translation"],
            "summary": "Translate a JSON object/array — preserve structure, smart skip rules.",
            "description": (
                "**Use case:** translate product catalogs, CMS exports, structured documents or "
                "any nested JSON where you want every text field translated but identifiers, "
                "URLs, prices and codes left untouched.\n\n"
                "**How it works:** walks the JSON tree depth-first, translates every "
                "human-readable string value, and writes the translation back into the same "
                "position. Numbers, booleans and `null` are untouched.\n\n"
                "**Three exclusion options** (each accepts a `;`-separated string OR a list[str]):\n\n"
                "- **`words_not_to_translate`** — words/phrases to keep verbatim *inside* "
                "translated text (brand names, place names, product names). Word-boundary match, "
                "default case-sensitive.\n"
                "- **`paths_to_exclude`** — dot-notation JSON paths to skip entirely. `*` matches "
                "any array index. A pattern matches the path *and its subtree*. Examples: "
                "`product.media.img_desc`, `items.*.image_url`.\n"
                "- **`common_keys_to_exclude`** — key names to skip at any nesting depth. "
                "Example: `name; price` skips every `.name` and `.price` field anywhere in the tree.\n\n"
                "**Auto skip filter** (`skip_non_text=true`, default): pure numbers, currency "
                "(`$99.99`), percent (`50%`), URLs, emails, UUIDs, hashes, ISO dates, code/ID "
                "patterns (`PROD-123`, `SKU-0001`).\n\n"
                "**Limits:** max 20 000 translatable strings, max nesting depth 200. "
                "Throughput ~78 strings/sec — 4 000 strings ≈ 50 s, 20 000 strings ≈ 4–5 min. "
                "**Set your client HTTP timeout accordingly.**"
            ),
            "operationId": "translateJsonObject",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/TranslateJsonObjectRequest"},
                        "examples": JSON_OBJ_REQ_EX,
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "Translated JSON, structure preserved.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/TranslateJsonObjectResponse"},
                            "examples": JSON_OBJ_RESP_EX,
                        },
                    },
                },
                "413": {
                    "description": "JSON exceeds size/depth limits (>20 000 strings or >200 depth).",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        },
                    },
                },
                **_err_responses(),
            },
        },
    }

    paths["/v1/dict"] = {
        "post": {
            "tags": ["Dictionary"],
            "summary": "Multilingual dictionary lookup for language learners.",
            "description": (
                "**Use case:** language-learning apps, browser extensions, vocabulary builders. "
                "Returns a rich entry with phonetic, definitions, examples, common phrases, "
                "synonyms/antonyms and mnemonic memory tips — all in the learner's native language.\n\n"
                "**Semantics:**\n"
                "- `nativeLang` — learner's mother tongue. All meanings and tips are emitted here.\n"
                "- `targetLang` — the language being learned. The input `word` is in this language.\n\n"
                "**Output uses camelCase** (`shortMeaning`, `partOfSpeech`, `relatedWords`, "
                "`memoryTips`, ...). `phonetic.romanization` is non-null for ja / zh / ko / ru / "
                "th / ar / hi (pinyin, romaji, revised romanization, etc.) and null for "
                "Latin-script words.\n\n"
                "**Latency:** typically 2.5–3.5 s per request."
            ),
            "operationId": "dictLookup",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DictRequest"},
                        "examples": DICT_REQ_EX,
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "Dictionary entry.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DictResponse"},
                            "examples": DICT_RESP_EX,
                        },
                    },
                },
                "404": {
                    "description": "No entry found for the requested word.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        },
                    },
                },
                **_err_responses(),
            },
        },
    }

    return paths


# ============================================================================
# Top-level spec
# ============================================================================
def build_spec() -> dict:
    description = (
        "**Translate API** — production-grade translation, i18n batch, structure-preserving "
        "**HTML** and **JSON** translation, plus a multilingual **dictionary** for language "
        "learners. Powered by **Qwen3-14B-AWQ** served on vLLM.\n\n"
        "## Why this API\n"
        "- **Five purpose-built endpoints** — text, i18n strings, HTML, JSON object, dictionary. "
        "Pick the one that matches your data shape.\n"
        "- **Structure-aware HTML & JSON** — tags, attributes, inline formatting, nested objects "
        "and arrays are preserved 100%. No more broken markup or stripped IDs.\n"
        "- **Brand-safe `ignore_terms`** — pin product names, place names and technical jargon "
        "to keep them verbatim inside translated copy.\n"
        "- **Smart skip filters for JSON** — auto-skip URLs, emails, UUIDs, hashes, ISO dates, "
        "currency and code/ID patterns. Plus per-path and per-key exclusion rules with wildcard "
        "support.\n"
        "- **Idiom-aware prompting + script-purity validation** — output is always in the target "
        "language's native script, never mixed with foreign characters.\n"
        "- **High throughput** — vLLM continuous batching gives ~78 strings/s on JSON, ~40 "
        "segments/s on HTML.\n\n"
        "## Endpoints\n"
        "| Endpoint | Best for |\n"
        "|---|---|\n"
        "| `POST /v1/translate` | Single sentences and small batches |\n"
        "| `POST /v1/json` | i18n key-string arrays |\n"
        "| `POST /v1/translate-html` | Articles, product descriptions, CMS exports |\n"
        "| `POST /v1/translate-json` | Product catalogs, structured documents |\n"
        "| `POST /v1/dict` | Language-learning apps, dictionaries, vocabulary builders |\n\n"
        "## Supported languages\n"
        "Vietnamese (`vi`), English (`en`), Japanese (`ja`), Chinese Simplified (`zh`) and "
        "Traditional (`zh-TW`), Korean (`ko`), French (`fr`), German (`de`), Spanish (`es`), "
        "Russian (`ru`), Thai (`th`), Indonesian (`id`), Portuguese (`pt`), Italian (`it`), "
        "Arabic (`ar`), Hindi (`hi`) and many more ISO 639-1 codes.\n\n"
        "## Quick start\n"
        "```bash\n"
        "curl -X POST https://translate.spacecloud.fit/v1/translate \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d '{\"text\":\"Hello world\",\"target_lang\":\"vi\"}'\n"
        "```\n\n"
        "Returns: `{\"translations\":[{\"translated_text\":\"Xin chào thế giới\",\"detected_source_lang\":\"en\"}]}`\n\n"
        "## Pricing & rate limits\n"
        "Managed by RapidAPI based on your subscription tier."
    )

    spec: dict = {
        "openapi": OPENAPI_VERSION,
        "info": {
            "title": "Translate API",
            "version": SPEC_VERSION,
            "description": description,
            "termsOfService": "https://translate.spacecloud.fit/terms",
            "contact": {
                "name": "Translate API Support",
                "url": "https://translate.spacecloud.fit",
                "email": "support@spacecloud.fit",
            },
            "license": {"name": "Proprietary"},
            "x-logo": {
                "url": "https://translate.spacecloud.fit/logo.png",
                "altText": "Translate API",
            },
        },
        "servers": [
            {"url": PROD_SERVER, "description": "Production"},
        ],
        "tags": [
            {
                "name": "Translation",
                "description": "Translate plain text, i18n batches, HTML and JSON.",
            },
            {
                "name": "Dictionary",
                "description": "Multilingual dictionary lookup for language learners.",
            },
        ],
        "paths": build_paths(),
        "components": {"schemas": build_schemas()},
    }

    return spec


def main() -> None:
    spec = build_spec()
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    OUT_JSON.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Wrote {OUT_YAML}  ({OUT_YAML.stat().st_size:,} bytes)")
    print(f"✓ Wrote {OUT_JSON}  ({OUT_JSON.stat().st_size:,} bytes)")
    print(f"  openapi:  {spec['openapi']}")
    print(f"  paths:    {len(spec['paths'])}")
    print(f"  schemas:  {len(spec['components']['schemas'])}")
    print(f"  servers:  {len(spec['servers'])}")
    # Count examples
    n_req_ex = 0
    n_resp_ex = 0
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            req = op.get("requestBody", {}).get("content", {}).get("application/json", {})
            n_req_ex += len(req.get("examples", {}))
            for code, resp in op.get("responses", {}).items():
                rj = resp.get("content", {}).get("application/json", {})
                n_resp_ex += len(rj.get("examples", {}))
    print(f"  request examples:  {n_req_ex}")
    print(f"  response examples: {n_resp_ex}")


if __name__ == "__main__":
    main()
