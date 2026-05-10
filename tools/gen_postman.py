#!/usr/bin/env python3
"""Generate Postman collection cho Translate + OCR API.

Run:  python3 tools/gen_postman.py
Output: postman/api.postman_collection.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "postman" / "api.postman_collection.json"


def req_json(name: str, url: str, body: dict | str, description: str = "") -> dict:
    raw = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, indent=2)
    return {
        "name": name,
        "request": {
            "method": "POST",
            "header": [{"key": "Content-Type", "value": "application/json"}],
            "url": {"raw": url, "host": [url.split("/v")[0]], "path": url.split(url.split("/v")[0] + "/")[1].split("/")},
            "description": description,
            "body": {
                "mode": "raw",
                "raw": raw,
                "options": {"raw": {"language": "json"}},
            },
        },
    }


def req_get(name: str, url: str, description: str = "") -> dict:
    parts = (
        url.replace("{{translate_url}}/", "")
        .replace("{{ocr_url}}/", "")
        .replace("{{tts_url}}/", "")
        .split("/")
    )
    base = url.split("/v")[0] if "/v" in url else url.rsplit("/", 1)[0]
    return {
        "name": name,
        "request": {
            "method": "GET",
            "header": [],
            "url": {"raw": url, "host": [base], "path": parts},
            "description": description,
        },
    }


def req_multipart(name: str, url: str, fields: list[tuple], description: str = "") -> dict:
    formdata = []
    for k, v, *rest in fields:
        kind = rest[0] if rest else "text"
        if kind == "file":
            formdata.append({"key": k, "type": "file", "src": v, "description": ""})
        else:
            formdata.append({"key": k, "value": str(v), "type": "text"})
    return {
        "name": name,
        "request": {
            "method": "POST",
            "header": [],
            "url": {"raw": url, "host": [url.split("/v")[0]], "path": url.split(url.split("/v")[0] + "/")[1].split("/")},
            "description": description,
            "body": {"mode": "formdata", "formdata": formdata},
        },
    }


# ============================================================
# Translate folder
# ============================================================
TRANSLATE = [
    req_json(
        "Translate single (auto detect)",
        "{{translate_url}}/v1/translate",
        {"text": "Hello world", "source_lang": "auto", "target_lang": "vi"},
        "Detect source lang tự động, dịch sang tiếng Việt.",
    ),
    req_json(
        "Translate single (explicit lang)",
        "{{translate_url}}/v1/translate",
        {"text": "Tôi yêu Việt Nam", "source_lang": "vi", "target_lang": "en"},
    ),
    req_json(
        "Translate batch (text array)",
        "{{translate_url}}/v1/translate",
        {"text": ["Good morning", "Good night", "Thank you"], "source_lang": "en", "target_lang": "vi"},
    ),
    req_json(
        "JSON batch (i18n)",
        "{{translate_url}}/v1/json",
        {"texts": ["Welcome", "Sign in", "Sign up"], "source_lang": "en", "target_lang": "ja"},
        "Dịch mảng string giữ nguyên thứ tự — phù hợp cho i18n keys.",
    ),
]


# ============================================================
# Dictionary folder — FORMAT MỚI (camelCase, native + target lang)
# ============================================================
DICT_DESC = """Tra từ điển đa ngôn ngữ — phục vụ người học ngoại ngữ.

**Semantic:**
- `native_lang` = mẹ đẻ user (output meanings/giải thích ở đây)
- `target_lang` = ngoại ngữ user đang học (word + examples ở đây)

**Output JSON (camelCase):**
```
{
  "word": "<từ ở target_lang>",
  "phonetic": { "ipa": "/.../", "romanization": "..." | null },
  "shortMeaning": "<nghĩa ngắn 1 dòng ở native_lang>",
  "definitions": [{ "partOfSpeech": "noun|verb|...", "meaning": "..." }],
  "examples":   [{ "text": "<target>", "meaning": "<native>" }],
  "phrases":    [{ "text": "<target>", "meaning": "<native>" }],
  "related": {
    "synonyms":     [{ "text": "<target>", "meaning": "<native>" }],
    "antonyms":     [{ "text": "<target>", "meaning": "<native>" }],
    "relatedWords": [{ "text": "<target>", "meaning": "<native>" }],
    "memoryTips":   ["<mẹo nhớ ở native_lang>"]
  }
}
```

`romanization` non-null cho ja/zh/ko/ru/th/ar/hi (pinyin/romaji/Cyrillic-Latin), null cho Latin script.

Latency ~2.5–3.5s/request.
"""

DICTIONARY = [
    req_json(
        "VN học EN — book",
        "{{translate_url}}/v1/dict",
        {"word": "book", "native_lang": "vi", "target_lang": "en"},
        "Người Việt học tiếng Anh, tra 'book'. Output: phonetic /bʊk/, meanings tiếng Việt.",
    ),
    req_json(
        "VN học JA — 本 (sách)",
        "{{translate_url}}/v1/dict",
        {"word": "本", "native_lang": "vi", "target_lang": "ja"},
        "Người Việt học tiếng Nhật. Romanization='hon' (romaji), meanings tiếng Việt.",
    ),
    req_json(
        "VN học ZH — 学习 (học tập)",
        "{{translate_url}}/v1/dict",
        {"word": "学习", "native_lang": "vi", "target_lang": "zh"},
        "Người Việt học tiếng Trung. Romanization='xuéxí' (pinyin), meanings tiếng Việt.",
    ),
    req_json(
        "VN học KO — 책 (sách)",
        "{{translate_url}}/v1/dict",
        {"word": "책", "native_lang": "vi", "target_lang": "ko"},
        "Người Việt học tiếng Hàn. Romanization='chaek' (revised romanization).",
    ),
    req_json(
        "VN học RU — Минобороны",
        "{{translate_url}}/v1/dict",
        {"word": "Минобороны", "native_lang": "vi", "target_lang": "ru"},
        "Người Việt học tiếng Nga. Romanization='Minoborony' (Cyrillic-Latin), meanings tiếng Việt.",
    ),
    req_json(
        "VN học FR — maison",
        "{{translate_url}}/v1/dict",
        {"word": "maison", "native_lang": "vi", "target_lang": "fr"},
        "Người Việt học tiếng Pháp. IPA /mɛzɔ̃/, romanization=null (Latin), meanings tiếng Việt.",
    ),
    req_json(
        "EN học VI — tự do",
        "{{translate_url}}/v1/dict",
        {"word": "tự do", "native_lang": "en", "target_lang": "vi"},
        "Người Anh học tiếng Việt. Meanings + memoryTips bằng tiếng Anh.",
    ),
]


# ============================================================
# OCR folder
# ============================================================
SAMPLE_IMG_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Newspaper_clipping.png/640px-Newspaper_clipping.png"
SAMPLE_LOCAL_FILE = "/path/to/your/image.png"

OCR = [
    req_get(
        "GET /v1/languages — Supported langs (110 codes, 12 model groups)",
        "{{ocr_url}}/v1/languages",
        "Liệt kê 110 lang codes PaddleOCR PP-OCRv5 + 12 multi-script wrappers.",
    ),
    req_json(
        "JSON image_url (full ML Kit hierarchy)",
        "{{ocr_url}}/v1/ocr",
        {"image_url": SAMPLE_IMG_URL, "lang": "auto"},
        "Response luôn trả full hierarchy: blocks > lines > words.",
    ),
    req_json(
        "JSON image_url + reading_order=rtl",
        "{{ocr_url}}/v1/ocr",
        {"image_url": SAMPLE_IMG_URL, "lang": "japan", "reading_order": "rtl"},
        "Manga JP / Arabic — sort blocks RTL.",
    ),
    req_json(
        "JSON mode=manga (specialized pipeline)",
        "{{ocr_url}}/v1/ocr",
        {"image_url": SAMPLE_IMG_URL, "lang": "japan", "mode": "manga"},
        "Specialized pipeline: PaddleOCR detection + manga-ocr recognition + bubble clustering + RTL.",
    ),
    req_multipart(
        "Multipart upload — full hierarchy",
        "{{ocr_url}}/v1/ocr/upload",
        [("file", [SAMPLE_LOCAL_FILE], "file"), ("lang", "auto")],
        "Upload file + tra full hierarchy. Đổi 'src' file path trong Postman trước khi chạy.",
    ),
    req_multipart(
        "Multipart upload — mode=manga",
        "{{ocr_url}}/v1/ocr/upload",
        [("file", [SAMPLE_LOCAL_FILE], "file"), ("lang", "japan"), ("mode", "manga")],
        "Upload + manga mode (manga-ocr GPU + bubble clustering).",
    ),
    req_json(
        "JSON image base64 — lang=auto (Auto detect)",
        "{{ocr_url}}/v1/ocr",
        {"image": "<BASE64_HERE>", "lang": "auto"},
        "Thay '<BASE64_HERE>' bằng base64 của ảnh.",
    ),
    req_json(
        "JSON image base64 — lang=en (English thuần)",
        "{{ocr_url}}/v1/ocr",
        {"image": "<BASE64_HERE>", "lang": "en"},
    ),
    req_json(
        "JSON image base64 — lang=vi (Vietnamese)",
        "{{ocr_url}}/v1/ocr",
        {"image": "<BASE64_HERE>", "lang": "vi"},
    ),
    req_json(
        "JSON image base64 — lang=ch (Chinese giản thể + EN)",
        "{{ocr_url}}/v1/ocr",
        {"image": "<BASE64_HERE>", "lang": "ch"},
    ),
    req_json(
        "JSON image base64 — lang=japan (Japanese)",
        "{{ocr_url}}/v1/ocr",
        {"image": "<BASE64_HERE>", "lang": "japan"},
    ),
    req_json(
        "JSON image base64 — lang=korean (Hangul)",
        "{{ocr_url}}/v1/ocr",
        {"image": "<BASE64_HERE>", "lang": "korean"},
    ),
    req_json(
        "JSON image base64 — lang=ru (Russian / East Slavic)",
        "{{ocr_url}}/v1/ocr",
        {"image": "<BASE64_HERE>", "lang": "ru"},
    ),
    req_json(
        "JSON image base64 — lang=ar (Arabic)",
        "{{ocr_url}}/v1/ocr",
        {"image": "<BASE64_HERE>", "lang": "ar"},
    ),
    req_json(
        "JSON image base64 — lang=hi (Hindi / Devanagari)",
        "{{ocr_url}}/v1/ocr",
        {"image": "<BASE64_HERE>", "lang": "hi"},
    ),
    req_json(
        "JSON image base64 — lang=th (Thai)",
        "{{ocr_url}}/v1/ocr",
        {"image": "<BASE64_HERE>", "lang": "th"},
    ),
]


# ============================================================
# TTS folder — Chatterbox Multilingual 0.5B
# ============================================================
TTS_DESC = """Text-to-Speech với Chatterbox Multilingual 0.5B (ResembleAI, MIT) — port 9004.

**23 ngôn ngữ:** ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh.
**KHÔNG có Vietnamese (vi).** Pydantic Literal sẽ trả 422 nếu request `vi`.

**Output:** JSON với `audio_base64` (WAV PCM 16-bit, 24kHz mono). Decode bằng `base64 -d > out.wav`.

**Voice (7):** preset profile từ server (`tts_service/voices/voices.json`):
- `default` — giọng nội bộ Chatterbox (no audio prompt)
- 3 nam: `male_deep` (Bass 96Hz), `male_warm` (Baritone 141Hz), `male_bright` (Tenor 157Hz)
- 3 nữ: `female_warm` (Alto 186Hz), `female_clear` (Mezzo 220Hz), `female_bright` (Soprano 245Hz)

Voice prompts từ LibriTTS-R (CC-BY-4.0), tone phân biệt rõ theo F0 (pitch).

**Text normalization (auto):** Chatterbox không có normalizer built-in nên server tự xử lý:
- `1000 dollars` → `one thousand dollars` (num2words)
- `$1234.56` → `1234.56 dollars` → `one thousand two hundred thirty-four point five six dollars`
- `50%` → `50 percent`
- French `1000€` → `mille euros` (suffix style)

Tắt qua env `TTS_NORMALIZE_NUMBERS=false` nếu cần raw text.

**Tempo control:** Chatterbox không có speed param. Dùng `cfg_weight`:
- `cfg_weight=0.3` → model "tự do", có xu hướng nói **chậm** hơn, đều hơn
- `cfg_weight=0.5` (default) → balanced
- `cfg_weight=0.7` → bám tempo voice prompt

(Đã thử post-process librosa time_stretch — pitch không tự nhiên với speech, đã bỏ.)

**Watermark Perth:** mọi audio đều có watermark imperceptible (responsible AI, không gỡ được).
"""

TTS = [
    req_get(
        "GET /v1/voices — Preset voices",
        "{{tts_url}}/v1/voices",
        "Liệt kê voice profile preset trên server (id, gender, language_hint, has_audio_prompt).",
    ),
    req_get(
        "GET /v1/languages — 23 supported languages",
        "{{tts_url}}/v1/languages",
        "Chatterbox Multilingual 0.5B hỗ trợ 23 ngôn ngữ. Note kèm cảnh báo VI chưa support.",
    ),
    req_json(
        "Synthesize EN — default voice",
        "{{tts_url}}/v1/tts",
        {"text": "Hello world, this is a TTS test from Chatterbox.",
         "language_id": "en", "voice_id": "default"},
        "Demo voice mặc định, không audio prompt. Decode audio_base64 → out.wav.",
    ),
    req_json(
        "Synthesize EN — full params",
        "{{tts_url}}/v1/tts",
        {"text": "The quick brown fox jumps over the lazy dog.",
         "language_id": "en", "voice_id": "default",
         "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
        "Đầy đủ generation parameters. Tăng exaggeration để dramatic hơn.",
    ),
    req_json(
        "Voice — Bass (male_deep, F0~96Hz)",
        "{{tts_url}}/v1/tts",
        {"text": "The quick brown fox jumps over the lazy dog.",
         "language_id": "en", "voice_id": "male_deep", "seed": 42},
        "Giọng nam trầm sâu — LibriTTS-R speaker 5536.",
    ),
    req_json(
        "Voice — Baritone (male_warm, F0~141Hz)",
        "{{tts_url}}/v1/tts",
        {"text": "The quick brown fox jumps over the lazy dog.",
         "language_id": "en", "voice_id": "male_warm", "seed": 42},
        "Giọng nam ấm trung — LibriTTS-R speaker 3170.",
    ),
    req_json(
        "Voice — Tenor (male_bright, F0~157Hz)",
        "{{tts_url}}/v1/tts",
        {"text": "The quick brown fox jumps over the lazy dog.",
         "language_id": "en", "voice_id": "male_bright", "seed": 42},
        "Giọng nam sáng cao — LibriTTS-R speaker 174.",
    ),
    req_json(
        "Voice — Alto (female_warm, F0~186Hz)",
        "{{tts_url}}/v1/tts",
        {"text": "The quick brown fox jumps over the lazy dog.",
         "language_id": "en", "voice_id": "female_warm", "seed": 42},
        "Giọng nữ ấm thấp — LibriTTS-R speaker 8842.",
    ),
    req_json(
        "Voice — Mezzo (female_clear, F0~220Hz)",
        "{{tts_url}}/v1/tts",
        {"text": "The quick brown fox jumps over the lazy dog.",
         "language_id": "en", "voice_id": "female_clear", "seed": 42},
        "Giọng nữ trong trung — LibriTTS-R speaker 6313.",
    ),
    req_json(
        "Voice — Soprano (female_bright, F0~245Hz)",
        "{{tts_url}}/v1/tts",
        {"text": "The quick brown fox jumps over the lazy dog.",
         "language_id": "en", "voice_id": "female_bright", "seed": 42},
        "Giọng nữ sáng cao — LibriTTS-R speaker 2035.",
    ),
    req_json(
        "Synthesize Chinese (Mandarin)",
        "{{tts_url}}/v1/tts",
        {"text": "你好，世界。这是中文语音合成测试。",
         "language_id": "zh", "voice_id": "default"},
        "Multilingual zh — Chatterbox dùng pkuseg để segment.",
    ),
    req_json(
        "Synthesize Japanese",
        "{{tts_url}}/v1/tts",
        {"text": "こんにちは、世界。これは日本語の音声合成テストです。",
         "language_id": "ja", "voice_id": "default"},
        "Multilingual ja — Chatterbox dùng pykakasi để romanize.",
    ),
    req_json(
        "Synthesize Korean",
        "{{tts_url}}/v1/tts",
        {"text": "안녕하세요, 세계. 한국어 음성 합성 테스트입니다.",
         "language_id": "ko", "voice_id": "default"},
    ),
    req_json(
        "Synthesize Spanish",
        "{{tts_url}}/v1/tts",
        {"text": "Hola mundo. Esta es una prueba de síntesis de voz.",
         "language_id": "es", "voice_id": "default"},
    ),
    req_json(
        "Synthesize French",
        "{{tts_url}}/v1/tts",
        {"text": "Bonjour le monde. Ceci est un test de synthèse vocale.",
         "language_id": "fr", "voice_id": "default"},
    ),
    req_json(
        "Synthesize Hindi (Devanagari)",
        "{{tts_url}}/v1/tts",
        {"text": "नमस्ते दुनिया। यह एक हिंदी वाणी संश्लेषण परीक्षण है।",
         "language_id": "hi", "voice_id": "default"},
    ),
    req_json(
        "Synthesize Arabic",
        "{{tts_url}}/v1/tts",
        {"text": "مرحبا بالعالم. هذا اختبار لتوليد الصوت.",
         "language_id": "ar", "voice_id": "default"},
    ),
    req_json(
        "Synthesize với seed (reproducible)",
        "{{tts_url}}/v1/tts",
        {"text": "Reproducible output test", "language_id": "en",
         "voice_id": "default", "seed": 42},
        "Cùng seed + text + voice + lang → audio_base64 identical (verify bằng MD5).",
    ),
    req_json(
        "Synthesize text dài (chunk + concat)",
        "{{tts_url}}/v1/tts",
        {"text": ("This is a longer text that will be chunked into multiple pieces. "
                  "The chunker splits sentences by punctuation. Each chunk is generated "
                  "independently, then concatenated with 50ms of silence between chunks. "
                  "The result should be smooth and natural to listen to."),
         "language_id": "en", "voice_id": "default"},
        "Test chunking + concat. Response chunk_count >= 2.",
    ),
    req_json(
        "Validation: language_id=vi → 422",
        "{{tts_url}}/v1/tts",
        {"text": "Tiếng Việt", "language_id": "vi", "voice_id": "default"},
        "Expect 422 — Chatterbox KHÔNG hỗ trợ Vietnamese.",
    ),
    req_json(
        "Validation: voice_id không tồn tại → 422",
        "{{tts_url}}/v1/tts",
        {"text": "Hello", "language_id": "en", "voice_id": "ghost_voice"},
        "Expect 422 với danh sách voice có sẵn.",
    ),
    req_json(
        "Number normalization: '1000 dollars'",
        "{{tts_url}}/v1/tts",
        {"text": "I have 1000 dollars in my wallet.",
         "language_id": "en", "voice_id": "male_warm", "seed": 42},
        "Server tự normalize '1000' → 'one thousand' (num2words). KHÔNG cần manual.",
    ),
    req_json(
        "Number normalization: currency + percent + decimal",
        "{{tts_url}}/v1/tts",
        {"text": "You get 50% off, total is $1234.56 only.",
         "language_id": "en", "voice_id": "female_clear", "seed": 42},
        "$ → dollars, 50% → fifty percent, 1234.56 → one thousand two hundred...",
    ),
    req_json(
        "Number normalization: French currency suffix",
        "{{tts_url}}/v1/tts",
        {"text": "Je gagne 1000€ par mois.",
         "language_id": "fr", "voice_id": "female_warm", "seed": 42},
        "Suffix style FR/DE/IT/ES: 1000€ → mille euros.",
    ),
    req_json(
        "Number normalization: year (cardinal)",
        "{{tts_url}}/v1/tts",
        {"text": "In year 2026, AI changes everything.",
         "language_id": "en", "voice_id": "male_bright", "seed": 42},
        "2026 → two thousand and twenty-six (cardinal mode).",
    ),
    req_json(
        "Tempo: cfg_weight thấp (chậm hơn, tự do)",
        "{{tts_url}}/v1/tts",
        {"text": "The quick brown fox jumps over the lazy dog.",
         "language_id": "en", "voice_id": "male_warm", "seed": 42,
         "cfg_weight": 0.3, "exaggeration": 0.5, "temperature": 0.7},
        "cfg_weight=0.3 → model 'tự do' hơn, có xu hướng nói chậm/đều. Recommend cho audiobook.",
    ),
    req_json(
        "Tempo: cfg_weight cao (bám voice prompt)",
        "{{tts_url}}/v1/tts",
        {"text": "The quick brown fox jumps over the lazy dog.",
         "language_id": "en", "voice_id": "male_warm", "seed": 42,
         "cfg_weight": 0.7, "exaggeration": 0.5, "temperature": 0.7},
        "cfg_weight=0.7 → bám tempo voice prompt. So sánh với cfg_weight=0.3 ở request trên.",
    ),
]


# ============================================================
# Health folder
# ============================================================
HEALTH = [
    req_get("Translate — liveness", "{{translate_url}}/healthz/live"),
    req_get("Translate — readiness (deep)", "{{translate_url}}/healthz/ready"),
    req_get("Translate — models info", "{{translate_url}}/v1/models",
            "Liệt kê model vLLM đang serve (Qwen3-14B-AWQ)."),
    req_get("OCR — liveness", "{{ocr_url}}/healthz/live"),
    req_get("OCR — readiness", "{{ocr_url}}/healthz/ready"),
    req_get("TTS — liveness", "{{tts_url}}/healthz/live"),
    req_get("TTS — readiness (deep — engine ready)", "{{tts_url}}/healthz/ready"),
]


COLLECTION = {
    "info": {
        "name": "Translate & OCR & TTS API",
        "_postman_id": "translate-ocr-tts-2026",
        "description": (
            "AI server — Translate + Dictionary + OCR + TTS.\n\n"
            "**Endpoints:**\n"
            "- `{{translate_url}}/v1/translate` — dịch single/batch\n"
            "- `{{translate_url}}/v1/json` — dịch i18n array\n"
            "- `{{translate_url}}/v1/dict` — tra từ điển đa ngôn ngữ\n"
            "- `{{ocr_url}}/v1/ocr` — OCR JSON (image_url hoặc base64)\n"
            "- `{{ocr_url}}/v1/ocr/upload` — OCR multipart upload\n"
            "- `{{ocr_url}}/v1/languages` — list lang codes\n"
            "- `{{tts_url}}/v1/tts` — synthesize speech (Chatterbox 0.5B, 23 lang, KHÔNG có VI)\n"
            "- `{{tts_url}}/v1/voices` — list preset voices\n"
            "- `{{tts_url}}/v1/languages` — list 23 supported languages\n\n"
            "Set biến `translate_url`, `ocr_url`, `tts_url` ở Variables tab."
        ),
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
    },
    "variable": [
        {"key": "translate_url", "value": "https://translate.spacecloud.fit", "type": "string"},
        {"key": "ocr_url", "value": "https://ocr.spacecloud.fit", "type": "string"},
        {"key": "tts_url", "value": "http://localhost:9004", "type": "string"},
    ],
    "item": [
        {"name": "Translate", "description": "Dịch single/batch — port 9002.", "item": TRANSLATE},
        {"name": "Dictionary", "description": DICT_DESC, "item": DICTIONARY},
        {"name": "OCR", "description": "OCR PaddleOCR PP-OCRv5 + manga-ocr (mode=manga) — port 9003.", "item": OCR},
        {"name": "TTS", "description": TTS_DESC, "item": TTS},
        {"name": "Health", "description": "Health checks cho 4 service.", "item": HEALTH},
    ],
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(COLLECTION, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {f["name"]: len(f["item"]) for f in COLLECTION["item"]}
    print(f"✓ Wrote {OUT}")
    print(f"  size: {OUT.stat().st_size:,} bytes")
    for name, n in counts.items():
        print(f"  {name}: {n} items")


if __name__ == "__main__":
    main()
