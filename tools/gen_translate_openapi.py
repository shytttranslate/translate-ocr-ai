#!/usr/bin/env python3
"""Sinh OpenAPI 3.0.3 spec cho Translate Service — phục vụ upload lên RapidAPI marketplace.

Run:    python3 tools/gen_translate_openapi.py
Output:
    postman/translate-openapi.yaml   (RapidAPI prefer YAML)
    postman/translate-openapi.json   (backup format)

Spec cover 3 endpoints chính:
- POST /v1/translate — single hoặc batch text, response shape khác nhau theo source_lang
- POST /v1/json     — i18n batch array, cùng shape với /v1/translate
- POST /v1/dict     — tra từ điển đa ngôn ngữ (rich camelCase output)

Plus health + models info.

Mọi description/summary trong spec viết tiếng Anh để fit marketplace international;
log/comment trong file generator vẫn tiếng Việt theo quy chuẩn nội bộ.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
OUT_YAML = ROOT / "postman" / "translate-openapi.yaml"
OUT_JSON = ROOT / "postman" / "translate-openapi.json"

OPENAPI_VERSION = "3.0.3"
SPEC_VERSION = "0.3.0"

PROD_SERVER = "https://translate.spacecloud.fit"
RAPIDAPI_SERVER = "https://translate.p.rapidapi.com"


# ----------------------------------------------------------------------------
# Schemas — components.schemas
# ----------------------------------------------------------------------------
def build_schemas() -> dict:
    schemas: dict = {}

    # --- Translate ---------------------------------------------------------
    schemas["TranslateRequest"] = {
        "type": "object",
        "required": ["text", "target_lang"],
        "properties": {
            "text": {
                "description": (
                    "Text to translate. Either a single string or an array (max 100 items, "
                    "each item up to 50 000 characters)."
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
                "example": ["Good morning", "Thank you"],
            },
            "source_lang": {
                "type": "string",
                "default": "auto",
                "pattern": "^(auto|[a-z]{2,3}(-[A-Z]{2})?)$",
                "description": (
                    "`auto` lets the model detect; otherwise an ISO 639-1 code "
                    "(`vi`, `en`, `ja`, `zh`, `ko`, `fr`, `de`, ...)."
                ),
                "example": "en",
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
        "description": "Returned when `source_lang=auto`: each item carries its own detected language.",
        "required": ["translated_text", "detected_source_lang"],
        "properties": {
            "translated_text": {"type": "string", "description": "Translated text."},
            "detected_source_lang": {
                "type": "string",
                "description": "ISO 639-1 code detected from the source text.",
                "example": "en",
            },
        },
    }

    # Response shape khác nhau theo source_lang — dùng oneOf cho field translations
    _translations_field = {
        "description": (
            "Translations in the same order as the input. "
            "When `source_lang=auto`, each item is `{translated_text, detected_source_lang}`. "
            "When an explicit `source_lang` is given, items are plain strings."
        ),
        "oneOf": [
            {
                "type": "array",
                "items": {"$ref": "#/components/schemas/TranslationDetected"},
            },
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

    # --- JSON batch (i18n) -------------------------------------------------
    schemas["JsonTranslateRequest"] = {
        "type": "object",
        "required": ["texts", "target_lang"],
        "properties": {
            "texts": {
                "type": "array",
                "minItems": 1,
                "maxItems": 100,
                "items": {"type": "string", "minLength": 1, "maxLength": 50000},
                "description": "Array of strings to translate (max 100 items, each up to 50 000 chars).",
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
                "description": "Required ISO 639-1 target language.",
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

    # --- Dictionary --------------------------------------------------------
    schemas["DictRequest"] = {
        "type": "object",
        "required": ["word", "targetLang"],
        "properties": {
            "word": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
                "description": "Word or phrase to look up (in `target_lang` — the language being learned).",
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
                "description": "IPA transcription, e.g. `/bʊk/`. Null when not available.",
                "example": "/bʊk/",
            },
            "romanization": {
                "type": "string",
                "nullable": True,
                "description": (
                    "Romanization for non-Latin scripts (pinyin, romaji, revised romanization)."
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
                    "One of: `noun`, `verb`, `adjective`, `adverb`, `preposition`,"
                    " `conjunction`, `interjection`, `pronoun`, `determiner`, `particle`,"
                    " `phrase`, `idiom`."
                ),
                "example": "noun",
            },
            "meaning": {
                "type": "string",
                "description": "Concise definition in `native_lang`.",
                "example": "quyển sách — vật phẩm gồm các trang giấy đóng lại với nhau",
            },
        },
    }

    schemas["DictExample"] = {
        "type": "object",
        "required": ["text", "meaning"],
        "properties": {
            "text": {
                "type": "string",
                "description": "Sentence or phrase in `target_lang`.",
                "example": "I bought a book yesterday.",
            },
            "meaning": {
                "type": "string",
                "description": "Translation into `native_lang`.",
                "example": "Tôi đã mua một quyển sách hôm qua.",
            },
        },
    }

    schemas["DictWordRef"] = {
        "type": "object",
        "required": ["text", "meaning"],
        "properties": {
            "text": {"type": "string", "description": "Reference word in `target_lang`."},
            "meaning": {"type": "string", "description": "Short gloss in `native_lang`."},
        },
    }

    schemas["DictRelated"] = {
        "type": "object",
        "properties": {
            "synonyms": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DictWordRef"},
            },
            "antonyms": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DictWordRef"},
            },
            "relatedWords": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DictWordRef"},
            },
            "memoryTips": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1–3 short mnemonic tips written in `native_lang`.",
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
            "model_used": {
                "type": "string",
                "description": "Identifier of the LLM serving the lookup, e.g. `translator-<8 hex>`.",
            },
            "word": {"type": "string", "description": "Normalized echo of the input word."},
            "nativeLang": {"type": "string", "example": "vi"},
            "targetLang": {"type": "string", "example": "en"},
            "phonetic": {"$ref": "#/components/schemas/DictPhonetic"},
            "shortMeaning": {
                "type": "string",
                "description": "One-line meaning in `native_lang` — suitable for quick display.",
                "example": "quyển sách",
            },
            "definitions": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DictDefinition"},
                "description": "1–2 main senses.",
            },
            "examples": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DictExample"},
                "description": "1–2 representative example sentences.",
            },
            "phrases": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DictExample"},
                "description": "0–4 common phrases / collocations.",
            },
            "related": {"$ref": "#/components/schemas/DictRelated"},
        },
    }

    # --- Health & info -----------------------------------------------------
    schemas["HealthLive"] = {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"type": "string", "enum": ["ok"]},
        },
    }

    schemas["HealthReady"] = {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"type": "string", "enum": ["ok", "degraded", "down"]},
            "components": {
                "type": "object",
                "additionalProperties": True,
                "description": "Per-component health (e.g. vLLM translator backend).",
            },
            "error": {"type": "string"},
        },
    }

    schemas["ModelsInfo"] = {
        "type": "object",
        "required": ["service", "translator"],
        "properties": {
            "service": {"type": "string", "example": "translate"},
            "translator": {
                "type": "object",
                "required": ["served_name", "fingerprint", "url"],
                "properties": {
                    "served_name": {"type": "string"},
                    "fingerprint": {"type": "string"},
                    "url": {"type": "string", "format": "uri"},
                },
            },
        },
    }

    schemas["ErrorResponse"] = {
        "type": "object",
        "required": ["detail"],
        "properties": {
            "detail": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "object"}},
                ],
                "description": "Error detail (string for upstream errors, array of validation errors from FastAPI).",
            },
        },
    }

    return schemas


# ----------------------------------------------------------------------------
# Examples (response samples) — embedded inline per path
# ----------------------------------------------------------------------------
TRANSLATE_REQ_EXAMPLES = {
    "single_auto": {
        "summary": "Single text + auto detect",
        "value": {"text": "Hello world", "target_lang": "vi"},
    },
    "single_explicit": {
        "summary": "Single text + explicit source",
        "value": {"text": "Tôi yêu Việt Nam", "source_lang": "vi", "target_lang": "en"},
    },
    "batch_explicit": {
        "summary": "Batch + explicit source",
        "value": {
            "text": ["Good morning", "Thank you"],
            "source_lang": "en",
            "target_lang": "vi",
        },
    },
    "batch_auto_mixed": {
        "summary": "Batch + auto on mixed-language input",
        "value": {"text": ["こんにちは", "안녕하세요"], "target_lang": "vi"},
    },
}

TRANSLATE_RESP_EXAMPLES = {
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
            "translations": ["Chào buổi sáng", "Cảm ơn"],
        },
    },
}

JSON_REQ_EXAMPLES = {
    "i18n_explicit": {
        "summary": "i18n keys (explicit en→ja)",
        "value": {
            "texts": ["Welcome", "Sign in", "Sign up"],
            "source_lang": "en",
            "target_lang": "ja",
        },
    },
    "i18n_auto": {
        "summary": "Auto detect on mixed batch",
        "value": {"texts": ["こんにちは", "안녕하세요"], "target_lang": "vi"},
    },
}

JSON_RESP_EXAMPLES = {
    "auto_with_detected": {
        "summary": "Auto output",
        "value": {
            "request_id": "263aafcd-ea31-423e-bd4b-1591d2f6cbe3",
            "processing_time_ms": 109,
            "translations": [
                {"translated_text": "Xin chào", "detected_source_lang": "ja"},
                {"translated_text": "Xin chào", "detected_source_lang": "ko"},
            ],
        },
    },
    "explicit_strings": {
        "summary": "Explicit output",
        "value": {
            "request_id": "8a24186d-df3b-4843-8caf-63702cc1e84c",
            "processing_time_ms": 71,
            "translations": ["ようこそ", "サインイン", "サインアップ"],
        },
    },
}

DICT_REQ_EXAMPLES = {
    "vi_learning_en": {
        "summary": "VI speaker learning EN — 'book'",
        "value": {"word": "book", "nativeLang": "vi", "targetLang": "en"},
    },
    "vi_learning_ja": {
        "summary": "VI speaker learning JA — '本'",
        "value": {"word": "本", "nativeLang": "vi", "targetLang": "ja"},
    },
    "en_learning_vi": {
        "summary": "EN speaker learning VI — 'tự do'",
        "value": {"word": "tự do", "nativeLang": "en", "targetLang": "vi"},
    },
}

DICT_RESP_EXAMPLE = {
    "summary": "VI speaker learning EN — 'book'",
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
            {"partOfSpeech": "noun", "meaning": "vật phẩm gồm các trang giấy đóng lại"},
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
}


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
def _error_responses() -> dict:
    return {
        "422": {
            "description": "Validation error (invalid lang code, empty text, exceeds size limits, etc.).",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                },
            },
        },
        "502": {
            "description": "Upstream LLM error (vLLM backend unreachable or returned error).",
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
            "tags": ["translate"],
            "summary": "Translate single text or a batch of strings.",
            "description": (
                "Translate one string or up to 100 strings in one request. The response "
                "shape depends on `source_lang`:\n"
                "- `source_lang=auto` → `translations` is an array of "
                "`{translated_text, detected_source_lang}`.\n"
                "- explicit `source_lang` → `translations` is an array of strings, in the "
                "same order as the input.\n\n"
                "When `text` is a single string, `translations` is still an array (length 1) "
                "for shape consistency."
            ),
            "operationId": "translate",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/TranslateRequest"},
                        "examples": TRANSLATE_REQ_EXAMPLES,
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "Translation result.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/TranslateResponse"},
                            "examples": TRANSLATE_RESP_EXAMPLES,
                        },
                    },
                },
                **_error_responses(),
            },
        },
    }

    paths["/v1/json"] = {
        "post": {
            "tags": ["translate"],
            "summary": "Translate an array of strings (i18n batch).",
            "description": (
                "Same response shape as `/v1/translate`. Use this endpoint when your input "
                "is always an array (e.g. localization keys for a UI). Order is preserved."
            ),
            "operationId": "translateJson",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/JsonTranslateRequest"},
                        "examples": JSON_REQ_EXAMPLES,
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "Translations in the same order as `texts`.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/JsonTranslateResponse"},
                            "examples": JSON_RESP_EXAMPLES,
                        },
                    },
                },
                **_error_responses(),
            },
        },
    }

    paths["/v1/dict"] = {
        "post": {
            "tags": ["dictionary"],
            "summary": "Look up a multilingual dictionary entry.",
            "description": (
                "Multilingual dictionary lookup intended for language learners.\n\n"
                "**Semantics:**\n"
                "- `nativeLang` = learner's mother tongue — meanings and tips are emitted here.\n"
                "- `targetLang` = the language being learned — the language of the input word.\n\n"
                "Output uses **camelCase** keys (`shortMeaning`, `partOfSpeech`, `relatedWords`, "
                "`memoryTips`, ...). `phonetic.romanization` is non-null for ja/zh/ko/ru/th/ar/hi "
                "and null for Latin-script words.\n\n"
                "Latency is typically 2.5–3.5 s per request."
            ),
            "operationId": "dictLookup",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DictRequest"},
                        "examples": DICT_REQ_EXAMPLES,
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "Dictionary entry.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DictResponse"},
                            "examples": {"vi_learning_en_book": DICT_RESP_EXAMPLE},
                        },
                    },
                },
                "404": {
                    "description": "No dictionary entry found for the requested word.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        },
                    },
                },
                **_error_responses(),
            },
        },
    }

    paths["/healthz/live"] = {
        "get": {
            "tags": ["health"],
            "summary": "Liveness probe — process is up.",
            "operationId": "healthLive",
            "responses": {
                "200": {
                    "description": "Process is running.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HealthLive"},
                        },
                    },
                },
            },
        },
    }

    paths["/healthz/ready"] = {
        "get": {
            "tags": ["health"],
            "summary": "Readiness probe — vLLM backend reachable.",
            "description": "Performs a deep check by issuing a minimal inference call to the vLLM translator backend.",
            "operationId": "healthReady",
            "responses": {
                "200": {
                    "description": "Service is ready (backend reachable).",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HealthReady"},
                        },
                    },
                },
                "503": {
                    "description": "Backend is degraded or down.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HealthReady"},
                        },
                    },
                },
            },
        },
    }

    paths["/v1/models"] = {
        "get": {
            "tags": ["info"],
            "summary": "Information about the underlying translator model.",
            "operationId": "modelsInfo",
            "responses": {
                "200": {
                    "description": "Model info.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ModelsInfo"},
                        },
                    },
                },
            },
        },
    }

    return paths


# ----------------------------------------------------------------------------
# Top-level spec assembly
# ----------------------------------------------------------------------------
def build_spec() -> dict:
    description = (
        "**Translate API** — production-grade translation, i18n batch and multilingual "
        "dictionary lookup, powered by **Qwen3-14B-AWQ** served on vLLM.\n\n"
        "## Highlights\n"
        "- `POST /v1/translate` — single string or batch (up to 100 items, each up to 50 000 chars).\n"
        "- `POST /v1/json` — i18n array of strings, order-preserving.\n"
        "- `POST /v1/dict` — rich dictionary entry for language learners (phonetic, "
        "definitions, examples, phrases, synonyms/antonyms, mnemonic tips).\n"
        "- Auto language detection with per-item `detected_source_lang` when "
        "`source_lang=auto`; plain `list[str]` response when an explicit source is given.\n"
        "- Script-purity post-validation with single retry to catch mixed-script outputs.\n\n"
        "## Quick start\n"
        "1. Subscribe on RapidAPI to obtain your `X-RapidAPI-Key`.\n"
        "2. `POST /v1/translate` with `{ \"text\": \"Hello\", \"target_lang\": \"vi\" }`.\n"
        "3. Read `translations[*]` — strings or `{translated_text, detected_source_lang}` "
        "objects depending on `source_lang`."
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
            {"url": PROD_SERVER, "description": "Production server — direct"},
            {"url": RAPIDAPI_SERVER, "description": "RapidAPI gateway"},
        ],
        "tags": [
            {"name": "translate", "description": "Translation endpoints (single/batch + i18n)."},
            {"name": "dictionary", "description": "Multilingual dictionary lookup for learners."},
            {"name": "health", "description": "Liveness and readiness probes."},
            {"name": "info", "description": "Model information."},
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


if __name__ == "__main__":
    main()
