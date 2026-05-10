# Translate API — Quick Start

Translate plain text, i18n strings, **HTML**, and **JSON** while preserving structure. Plus a multilingual **dictionary** for language learners. Powered by Qwen3-14B-AWQ on vLLM.

---

## Authentication

When you call through RapidAPI's gateway, the only headers you need:

```http
X-RapidAPI-Key:  <YOUR_RAPIDAPI_KEY>
X-RapidAPI-Host: <RAPIDAPI_HOST>
Content-Type:    application/json
```

Both headers are pre-filled in the **Code Snippets** tab — copy from there.

---

## Try it (60 seconds)

The most common call: translate a string with auto-detect.

```bash
curl -X POST 'https://<RAPIDAPI_HOST>/v1/translate' \
  -H 'Content-Type: application/json' \
  -H 'X-RapidAPI-Key: <YOUR_RAPIDAPI_KEY>' \
  -H 'X-RapidAPI-Host: <RAPIDAPI_HOST>' \
  -d '{"text":"Hello world","target_lang":"vi"}'
```

Response (~120 ms):

```json
{
  "request_id": "03f0ece3-9ad8-4b52-ad0c-40da09487722",
  "processing_time_ms": 126,
  "translations": [
    { "translated_text": "Xin chào thế giới", "detected_source_lang": "en" }
  ]
}
```

**Two response shapes** depending on `source_lang`:

| `source_lang` | Shape of `translations` |
|---|---|
| `"auto"` (default) | `[ { translated_text, detected_source_lang } ]` |
| explicit (e.g. `"en"`) | `[ "string", "string", … ]` |

---

## Endpoints at a glance

| Endpoint | Use it for | Per-request limits |
|---|---|---|
| `POST /v1/translate` | Single string or up to 100 strings | 100 items × 50 000 chars |
| `POST /v1/json` | i18n key arrays | 100 items × 50 000 chars |
| `POST /v1/translate-html` | HTML — articles, product pages, emails | 20 MB input |
| `POST /v1/translate-json` | Nested JSON — catalogs, CMS exports | 20 000 strings, depth 200 |
| `POST /v1/dict` | Dictionary lookup for learners | 200 chars per word |

**Supported languages (16+):** `vi`, `en`, `ja`, `zh`, `zh-TW`, `ko`, `fr`, `de`, `es`, `ru`, `th`, `id`, `pt`, `it`, `ar`, `hi`, and most other ISO 639-1 codes.

> **HTML and JSON endpoints have powerful exclusion options** — `ignore_terms` to keep brand names verbatim, `paths_to_exclude` (with `*` wildcard) and `common_keys_to_exclude` for JSON. See the full request schemas in the API spec.

---

## Throughput & timeouts

Pick a client HTTP timeout based on input size:

| Input size | Recommended client timeout |
|---|---|
| Single string / batch ≤ 100 items | **30 s** |
| HTML ≤ 500 segments / JSON ≤ 1 000 strings | **60 s** |
| HTML ≤ 5 000 segments / JSON ≤ 5 000 strings | **3 min** |
| Maximum input (20 MB / 20 000 strings) | **10+ min** |

Throughput on the GPU: ~80 strings/sec for JSON, ~40 segments/sec for HTML, single text ~120 ms.

---

## Status codes

| Code | Meaning | What to do |
|---|---|---|
| **200** | Success | Use `translations` / `html` / `json_data` from the response. |
| **400** | Malformed JSON body | Check that your request body is valid JSON. |
| **401** | Missing/invalid `X-RapidAPI-Key` | Verify the key in the RapidAPI dashboard. |
| **403** | Subscription quota exceeded | Upgrade tier on RapidAPI or wait for reset. |
| **413** | JSON too large (only `/v1/translate-json`) | Input has > 20 000 strings or depth > 200. Split into chunks. |
| **422** | Validation error | Read `detail`. Common: invalid `target_lang`, exceeded length, `target_lang: "auto"`. |
| **422** (`html_too_malformed`) | HTML too broken (only `/v1/translate-html`) | Run input through `tidy` / `prettier --parser html` and retry. See structured detail in response. |
| **429** | Rate limit exceeded | Backoff and retry. |
| **502** | Upstream LLM error | Retry with exponential backoff (1 s, 3 s, 9 s, max 3 retries). |

### `422` validation example

```json
{
  "detail": [
    { "loc": ["body", "target_lang"], "msg": "string does not match regex …", "type": "value_error.str.regex" }
  ]
}
```

### `422` HTML-too-malformed example

```json
{
  "detail": {
    "error": "html_too_malformed",
    "health": "severe",
    "metrics": { "error_rate": 0.75, "errors_total": 6, "fatals_total": 0 },
    "errors_sample": [
      { "line": 1, "column": 77, "severity": "error", "message": "Opening and ending tag mismatch: b and i" }
    ],
    "suggestion": "Run HTML through `tidy -q -m -ashtml` or `html-minifier-terser` before retry."
  }
}
```

---

## Tips

- **Auto-detect costs nothing extra.** Use `source_lang: "auto"` for user-generated or mixed-language content.
- **Always use `ignore_terms` / `words_not_to_translate` for brand names.** Word-boundary, default case-sensitive — `Apple` ≠ `apples`.
- **For JSON, exclude by `paths_to_exclude` when you can.** It's faster and 100% deterministic vs. relying on the auto-skip filter.
- **Don't pre-strip HTML tags.** Send raw HTML to `/v1/translate-html` — the API preserves tags, attributes, and inline formatting (including reordering inline tags around translated words).

Need the full request/response schemas? See the OpenAPI spec on this listing.
