# Translate API — Quick Start Tutorial for RapidAPI Users

> **TL;DR** — One API, five purpose-built endpoints: plain text, i18n string arrays, **HTML with preserved structure**, **JSON objects with smart skip rules**, and a multilingual dictionary for language learners. Powered by Qwen3-14B-AWQ on vLLM.

This tutorial walks you from "I just subscribed on RapidAPI" to "I'm shipping translated content in production" in ~15 minutes.

---

## Table of contents

1. [What you can do with this API](#1-what-you-can-do-with-this-api)
2. [Prerequisites](#2-prerequisites)
3. [Authentication — the RapidAPI way](#3-authentication--the-rapidapi-way)
4. [Your first call (60 seconds)](#4-your-first-call-60-seconds)
5. [Endpoint walkthroughs](#5-endpoint-walkthroughs)
   - [5.1 `POST /v1/translate` — plain text & batches](#51-post-v1translate--plain-text--batches)
   - [5.2 `POST /v1/json` — i18n string arrays](#52-post-v1json--i18n-string-arrays)
   - [5.3 `POST /v1/translate-html` — HTML with preserved structure](#53-post-v1translate-html--html-with-preserved-structure)
   - [5.4 `POST /v1/translate-json` — JSON objects with smart skip rules](#54-post-v1translate-json--json-objects-with-smart-skip-rules)
   - [5.5 `POST /v1/dict` — multilingual dictionary](#55-post-v1dict--multilingual-dictionary)
6. [Common patterns](#6-common-patterns)
7. [Tips & best practices](#7-tips--best-practices)
8. [Errors & troubleshooting](#8-errors--troubleshooting)
9. [Limits, throughput & timeouts](#9-limits-throughput--timeouts)
10. [FAQ](#10-faq)
11. [Support](#11-support)

---

## 1. What you can do with this API

| If your input is… | Use this endpoint | Highlights |
|---|---|---|
| A sentence or up to 100 strings | `POST /v1/translate` | Auto-detect or explicit source language |
| An i18n key array (`["Welcome", "Sign in", …]`) | `POST /v1/json` | Order-preserving batch for localization |
| HTML — articles, product pages, emails | `POST /v1/translate-html` | Tags, attributes, inline formatting **all preserved**; brand-name `ignore_terms`; auto-fix minor parse issues |
| A nested JSON object (catalog, CMS export) | `POST /v1/translate-json` | Smart skip filters; per-path/per-key exclusion rules; word-boundary `words_not_to_translate` |
| A single word for a learner app | `POST /v1/dict` | Phonetic (IPA + romanization), definitions, examples, phrases, synonyms/antonyms, mnemonic tips |

**16+ languages** out of the box: Vietnamese (`vi`), English (`en`), Japanese (`ja`), Chinese Simplified/Traditional (`zh`, `zh-TW`), Korean (`ko`), French (`fr`), German (`de`), Spanish (`es`), Russian (`ru`), Thai (`th`), Indonesian (`id`), Portuguese (`pt`), Italian (`it`), Arabic (`ar`), Hindi (`hi`), and most other ISO 639-1 codes.

---

## 2. Prerequisites

1. **A RapidAPI account** — free to create at [rapidapi.com](https://rapidapi.com).
2. **A subscription** to **Translate API** in the RapidAPI Hub. Pick a tier that matches your expected monthly volume.
3. **Your RapidAPI key** — visible in the API page header once you're subscribed. It looks like `abc123def4567890abcdef1234567890abcdef1234567890`.

That's it. No OAuth, no separate service account.

---

## 3. Authentication — the RapidAPI way

When you call the API **through RapidAPI's gateway**, you only need two headers:

```http
X-RapidAPI-Key:  <YOUR_RAPIDAPI_KEY>
X-RapidAPI-Host: <RAPIDAPI_HOST>
```

> Both headers are pre-filled in the RapidAPI **Code Snippets** tab — copy from there for any language.

You **do not** need to manage any other credential. The gateway authenticates you, forwards the request, and meters usage automatically.

---

## 4. Your first call (60 seconds)

Translate `"Hello world"` to Vietnamese.

### cURL

```bash
curl -X POST 'https://<RAPIDAPI_HOST>/v1/translate' \
  -H 'Content-Type: application/json' \
  -H 'X-RapidAPI-Key: <YOUR_RAPIDAPI_KEY>' \
  -H 'X-RapidAPI-Host: <RAPIDAPI_HOST>' \
  -d '{
    "text": "Hello world",
    "target_lang": "vi"
  }'
```

### Response

```json
{
  "request_id": "03f0ece3-9ad8-4b52-ad0c-40da09487722",
  "processing_time_ms": 126,
  "translations": [
    {
      "translated_text": "Xin chào thế giới",
      "detected_source_lang": "en"
    }
  ]
}
```

That's the entire roundtrip: ~120 ms, source language auto-detected, target text returned.

> **Why is `translations` an array even for one input?** So your client never has to branch on input shape. When `text` is a single string, you get a length-1 array. When it's an array, you get an aligned array of the same length. Same code path either way.

---

## 5. Endpoint walkthroughs

### 5.1 `POST /v1/translate` — plain text & batches

**Use it for:** chat messages, sentences, paragraphs, or up to **100 strings** in one call.

#### Single string with auto-detect

```json
{ "text": "Tôi yêu Việt Nam", "target_lang": "en" }
```

```json
{
  "translations": [
    { "translated_text": "I love Vietnam", "detected_source_lang": "vi" }
  ]
}
```

#### Batch with explicit source language

```json
{
  "text": ["Good morning", "Thank you", "How are you?"],
  "source_lang": "en",
  "target_lang": "vi"
}
```

When you provide `source_lang` (not `auto`), the response `translations` is a **plain `list[str]`** — perfect for tight integrations:

```json
{
  "translations": ["Chào buổi sáng", "Cảm ơn", "Bạn khỏe không?"]
}
```

#### Batch with mixed-language input + `auto`

```json
{
  "text": ["こんにちは", "안녕하세요", "Bonjour"],
  "target_lang": "vi"
}
```

```json
{
  "translations": [
    { "translated_text": "Xin chào", "detected_source_lang": "ja" },
    { "translated_text": "Xin chào", "detected_source_lang": "ko" },
    { "translated_text": "Xin chào", "detected_source_lang": "fr" }
  ]
}
```

Each item carries its **own** detected source language — handy for chat threads or user-generated content where languages mix freely.

---

### 5.2 `POST /v1/json` — i18n string arrays

**Use it for:** localization key bundles. Same response shape as `/v1/translate`, but the request **always** uses an array, matching how i18n libraries (i18next, FormatJS, ICU, …) store keys.

```json
{
  "texts": ["Welcome", "Sign in", "Sign up", "Forgot password?"],
  "source_lang": "en",
  "target_lang": "ja"
}
```

```json
{
  "translations": ["ようこそ", "サインイン", "サインアップ", "パスワードをお忘れですか？"]
}
```

> **Order is preserved**: `translations[i]` corresponds to `texts[i]`. Use the same index lookup you already have for English keys.

---

### 5.3 `POST /v1/translate-html` — HTML with preserved structure

**Use it for:** articles, product descriptions, emails, CMS pages — anywhere tags, attributes, and inline formatting must survive intact.

#### Simple example — inline tag preserved

```json
{
  "html": "<p>Hello <b>red</b> car!</p>",
  "target_lang": "vi"
}
```

```json
{
  "html": "<p>Chào xe <b>đỏ</b>!</p>",
  "detected_source_lang": "en",
  "health": { "health": "clean", "parse_tier": "fragment_wrap" /* … */ },
  "segments_translated": 1,
  "warnings": []
}
```

Notice the inline `<b>` ended up on a different word in Vietnamese (`đỏ` = red, comes after the noun) — but the tag is still there, wrapping the right word. That's because the API uses **inline-aware segmentation** with XLIFF-style placeholders behind the scenes.

#### Brand names with `ignore_terms`

Stop your translator from mangling **Apple**, **MacBook Pro**, **iPhone**, etc.:

```json
{
  "html": "<p>Apple makes the new MacBook Pro and iPhone. Buy now!</p>",
  "source_lang": "en",
  "target_lang": "vi",
  "ignore_terms": ["Apple", "MacBook Pro", "iPhone"]
}
```

```json
{
  "html": "<p>Apple tạo ra MacBook Pro mới và iPhone. Mua ngay!</p>"
}
```

> **Word-boundary match, default case-sensitive.** `Apple` won't match `apples`, and `apple` (lowercase) is treated as a different term unless you set `"ignore_case": true`.

#### Attributes are translated too

`alt`, `title`, `aria-label`, `placeholder`, and `<meta name="description">` content — translated by default. URLs (`href`, `src`), classes, IDs, and data-attributes are **never** touched.

```json
{
  "html": "<img src=\"cat.jpg\" alt=\"A picture of a cat\" title=\"Click to enlarge\"><p>Image gallery</p>",
  "source_lang": "en",
  "target_lang": "vi"
}
```

```json
{
  "html": "<img src=\"cat.jpg\" alt=\"Một bức tranh về một con mèo\" title=\"Nháy để phóng to\"><p>Thư viện hình ảnh</p>"
}
```

Set `"translate_attributes": false` if you want text nodes only.

#### What's skipped

`<script>`, `<style>`, `<noscript>`, `<svg>`, `<math>`, `<template>`, `<textarea>`, `<pre>`, `<code>`, `<kbd>`, `<samp>`, `<var>` — entire subtree skipped. So this:

```html
<p>Use <code>print()</code> to debug.</p><script>console.log("skip")</script>
```

becomes:

```html
<p>Sử dụng <code>print()</code> để gỡ lỗi.</p><script>console.log("skip")</script>
```

Code stays code. JavaScript stays JavaScript. Translator only translates prose.

#### When the HTML is broken

The API auto-fixes **minor** parse issues (unclosed tags, attribute quirks, vendor markup from Word/Outlook) and returns `health: "clean" | "minor" | "moderate"`. **Severely** malformed HTML returns a **422** with diagnostic info — see [Errors & troubleshooting](#8-errors--troubleshooting).

---

### 5.4 `POST /v1/translate-json` — JSON objects with smart skip rules

**Use it for:** product catalogs, CMS exports, structured documents — any nested JSON where you want every text field translated but identifiers, URLs, prices, and codes left untouched.

#### Basic call

```json
{
  "json_data": {
    "title": "Welcome",
    "description": "A great product"
  },
  "target_lang": "vi"
}
```

```json
{
  "json_data": {
    "title": "Chào mừng",
    "description": "Một sản phẩm tuyệt vời"
  },
  "stats": { "strings_translated": 2, "strings_skipped": 0, "chars_translated": 22 }
}
```

#### The killer feature: three exclusion options

You can mix and match these to keep your data clean:

| Option | What it does | Format |
|---|---|---|
| `words_not_to_translate` | Keep words/phrases verbatim **inside** translated text | `";"`-separated string OR `list[str]` |
| `paths_to_exclude` | Skip entire JSON paths (with `*` wildcard for array indices) | `";"`-separated string OR `list[str]` |
| `common_keys_to_exclude` | Skip key names at **any** nesting depth | `";"`-separated string OR `list[str]` |

#### Real-world example — e-commerce catalog

```json
{
  "json_data": {
    "title": "Premium Wireless Earbuds",
    "price": 99.99,
    "product": {
      "name": "Earbuds Pro",
      "description": "Best quality from New York",
      "media": {
        "img_desc": "Detailed product photo",
        "title": "Main image"
      }
    },
    "items": [
      { "name": "Item A", "image_url": "https://x.com/a.jpg", "desc": "Item A description" },
      { "name": "Item B", "image_url": "https://x.com/b.jpg", "desc": "Item B description" }
    ]
  },
  "source_lang": "en",
  "target_lang": "vi",
  "words_not_to_translate": "Earbuds; New York",
  "paths_to_exclude": "product.media.img_desc; items.*.image_url",
  "common_keys_to_exclude": "name; price"
}
```

```json
{
  "json_data": {
    "title": "Không dây cao cấp Earbuds",
    "price": 99.99,
    "product": {
      "name": "Earbuds Pro",
      "description": "Chất lượng tốt nhất từ New York",
      "media": {
        "img_desc": "Detailed product photo",
        "title": "Hình ảnh chính"
      }
    },
    "items": [
      { "name": "Item A", "image_url": "https://x.com/a.jpg", "desc": "Mô tả mục A" },
      { "name": "Item B", "image_url": "https://x.com/b.jpg", "desc": "Mô tả mục B" }
    ]
  },
  "stats": { "strings_translated": 5 /* … */ }
}
```

What just happened?

- **`Earbuds`** stays inside the translated title (`"Không dây cao cấp Earbuds"`) — it's a brand term.
- **`New York`** stays inside the translated description.
- **`product.media.img_desc`** is left as `"Detailed product photo"` — that path is excluded.
- **`items.*.image_url`** for both items keeps the URL — wildcard matched both indices.
- **`name`** and **`price`** are skipped *everywhere* (top-level `price`, `product.name`, every `items[*].name`).
- Everything else is translated.

#### Auto-skip filter (on by default)

Strings that are clearly **not** human prose are skipped automatically. You don't have to list them:

| Type | Example |
|---|---|
| Numbers | `99`, `1,234.56`, `+42` |
| Currency | `$99.99`, `€1.234,56` |
| Percent | `50%`, `12.5‰` |
| URLs | `https://example.com`, `mailto:x@y.com` |
| Emails | `support@example.com` |
| UUIDs | `550e8400-e29b-41d4-a716-446655440000` |
| Hashes | MD5, SHA1, SHA256 (32–128 hex chars) |
| ISO dates | `2026-01-15`, `2026-01-15T10:30:00Z` |
| Code/IDs | `PROD-123-X`, `SKU-0001` |

Set `"skip_non_text": false` to translate **everything** (including IDs and dates) — useful for tests or unusual data.

---

### 5.5 `POST /v1/dict` — multilingual dictionary

**Use it for:** language-learning apps, browser extensions, vocabulary builders. Returns a rich entry with phonetic, definitions, examples, common phrases, synonyms/antonyms, and mnemonic memory tips — all in the learner's native language.

```json
{
  "word": "book",
  "nativeLang": "vi",
  "targetLang": "en"
}
```

```json
{
  "word": "book",
  "phonetic": { "ipa": "/bʊk/", "romanization": null },
  "shortMeaning": "quyển sách",
  "definitions": [
    { "partOfSpeech": "noun", "meaning": "vật phẩm gồm các trang giấy đóng lại với nhau, dùng để đọc" },
    { "partOfSpeech": "verb", "meaning": "đặt chỗ trước (nhà hàng, vé, phòng)" }
  ],
  "examples": [
    { "text": "I bought a book yesterday.", "meaning": "Tôi đã mua một quyển sách hôm qua." },
    { "text": "Please book a table for two.", "meaning": "Làm ơn đặt bàn cho hai người." }
  ],
  "phrases": [
    { "text": "by the book", "meaning": "đúng theo quy tắc" },
    { "text": "book club", "meaning": "câu lạc bộ đọc sách" }
  ],
  "related": {
    "synonyms": [{ "text": "volume", "meaning": "tập sách" }],
    "antonyms": [],
    "relatedWords": [{ "text": "novel", "meaning": "tiểu thuyết" }],
    "memoryTips": ["'book' nghe gần với 'búc' — 'búc cuốn sách'"]
  }
}
```

> **Semantics:** `nativeLang` is the **learner's mother tongue** (where meanings/tips are written). `targetLang` is the **language being learned** (which the input `word` is in). For a Japanese kanji like `本`, `phonetic.romanization` will be `"hon"` (romaji); for Mandarin, it'll be pinyin; for Korean, revised romanization.

---

## 6. Common patterns

### 6.1 Translate every i18n key in your app (en → ja)

```python
import requests, json

# Load your en bundle
with open("locales/en.json") as f:
    keys = json.load(f)  # {"welcome": "Welcome", "signin": "Sign in", ...}

# Order-preserving batch
order = list(keys.keys())
texts = [keys[k] for k in order]

resp = requests.post(
    "https://<RAPIDAPI_HOST>/v1/json",
    headers={
        "X-RapidAPI-Key":  "<YOUR_KEY>",
        "X-RapidAPI-Host": "<RAPIDAPI_HOST>",
        "Content-Type":    "application/json",
    },
    json={"texts": texts, "source_lang": "en", "target_lang": "ja"},
).json()

ja_keys = dict(zip(order, resp["translations"]))
with open("locales/ja.json", "w") as f:
    json.dump(ja_keys, f, ensure_ascii=False, indent=2)
```

### 6.2 Translate a blog post HTML, keeping brand names intact

```python
html = """<article>
  <h1>How Apple changed the music industry</h1>
  <p>The iPod launched in 2001…</p>
</article>"""

resp = requests.post(
    "https://<RAPIDAPI_HOST>/v1/translate-html",
    headers={...},  # same as before
    json={
        "html": html,
        "source_lang": "en",
        "target_lang": "vi",
        "ignore_terms": ["Apple", "iPod"],
    },
    timeout=120,  # longer for HTML
).json()

print(resp["html"])
print(f"{resp['segments_translated']} segments, health={resp['health']['health']}")
```

### 6.3 Translate an e-commerce product catalog

```python
catalog = json.load(open("catalog.json"))

resp = requests.post(
    "https://<RAPIDAPI_HOST>/v1/translate-json",
    headers={...},
    json={
        "json_data": catalog,
        "source_lang": "en",
        "target_lang": "vi",
        "common_keys_to_exclude": "sku; barcode; price; currency; created_at",
        "paths_to_exclude": "metadata.*; images.*.url",
        "words_not_to_translate": ["Pro", "Max", "Plus", "Lite"],
    },
    timeout=300,  # large catalog → long wall-clock
).json()

with open("catalog.vi.json", "w") as f:
    json.dump(resp["json_data"], f, ensure_ascii=False, indent=2)
```

### 6.4 Build a vocabulary card from a single word

```python
resp = requests.post(
    "https://<RAPIDAPI_HOST>/v1/dict",
    headers={...},
    json={"word": "freedom", "nativeLang": "vi", "targetLang": "en"},
).json()

card = {
    "word": resp["word"],
    "ipa":  resp["phonetic"]["ipa"],
    "short": resp["shortMeaning"],
    "examples": [e["text"] + "  →  " + e["meaning"] for e in resp["examples"]],
    "tip": resp["related"]["memoryTips"][0] if resp["related"]["memoryTips"] else "",
}
```

---

## 7. Tips & best practices

### Pick the right endpoint

- **One sentence?** → `/v1/translate` with a `text` string.
- **Up to 100 sentences?** → `/v1/translate` with a `text` array (single round-trip, batched on GPU).
- **i18n key bundle?** → `/v1/json`.
- **Anything that has tags?** → `/v1/translate-html`. Don't pre-strip tags; you'll lose formatting.
- **Anything that has structure (objects, arrays)?** → `/v1/translate-json`. Don't flatten to strings; you'll lose paths.

### Use `ignore_terms` / `words_not_to_translate` aggressively

Brand names, product names, place names, technical jargon — anything you want to read identically in every language. Word-boundary match means you won't accidentally hit substrings.

### Use `paths_to_exclude` for predictable structure

If your JSON has fields that are **always** non-translatable (`*.sku`, `metadata.*`, `id`, `image_url`), exclude them by path rather than relying on the auto-skip filter. Faster and 100% deterministic.

### Set HTTP timeouts based on input size

| Input | Recommended client timeout |
|---|---|
| Single text or small batch (< 100 strings) | 30 s |
| HTML up to ~500 segments / JSON up to 1 000 strings | 60 s |
| HTML up to ~5 000 segments / JSON up to 5 000 strings | 3 minutes |
| Maximum size (20 MB HTML / 20 000 strings) | 10+ minutes |

### Use auto-detect for user-generated content

If you don't *know* what language a string is in (chat, comments, social media), let `source_lang: "auto"` decide. Per-item detection means a mixed-language batch is one call, not five.

### Use explicit `source_lang` when you can

It's slightly faster (skips the detection step) and makes the response shape simpler (`list[str]` instead of `list[{translated_text, detected_source_lang}]`).

---

## 8. Errors & troubleshooting

### `422 Unprocessable Entity`

You sent something the API can't validate.

- **Generic** — body looks like `{"detail": [{"loc": [...], "msg": "..."}]}`. Read `msg` and the field path. Common causes: missing `target_lang`, invalid language code, exceeded length.
- **`html_too_malformed`** (only for `/v1/translate-html`) — your HTML had >10% mismatched tags or fatal markers (encoding errors, deeply broken structure):

  ```json
  {
    "detail": {
      "error": "html_too_malformed",
      "health": "severe",
      "metrics": { "error_rate": 0.75, "errors_total": 6 /* … */ },
      "errors_sample": [
        { "line": 1, "column": 77, "severity": "error",
          "message": "Opening and ending tag mismatch: b and i" }
      ],
      "suggestion": "Run HTML through `tidy -q -m -ashtml` or `html-minifier-terser` before retry."
    }
  }
  ```

  Run your HTML through a sanitizer (e.g. `tidy`, `html-minifier-terser`, `prettier --parser html`) and retry.

### `413 Payload Too Large` (only on `/v1/translate-json`)

Your JSON has more than 20 000 translatable strings or nests deeper than 200 levels. Split into chunks and re-merge on your side.

### `502 Bad Gateway`

The upstream model server is briefly unavailable or returned an error. Retry with exponential backoff (start at 1 s, max 3 retries).

### Translation looks weird / mixes scripts

Rare — the API does script-purity validation and a single retry pass internally. If it still happens, check the `warnings` field in the response. For HTML/JSON endpoints, warnings flag `inline_placeholders_lost`, `ignore_terms_lost_in_translation`, `post_translate_tag_mismatch`, etc.

### Brand name still got translated

- Check case sensitivity: `Apple` and `apple` are different terms by default.
- Multi-word phrases must match exactly: `"MacBook Pro"` (1 space) — `"MacBook  Pro"` (2 spaces) won't match.
- If your text has the term inside a longer word (`appletree`), it won't match — that's word-boundary behavior.
- Set `"ignore_case": true` if you want case-insensitive match.

---

## 9. Limits, throughput & timeouts

| Endpoint | Per-request limit | Throughput |
|---|---|---|
| `/v1/translate` | 100 items, 50 000 chars/item | ~80–200 ms per item |
| `/v1/json` | 100 items, 50 000 chars/item | Same as above |
| `/v1/translate-html` | 20 MB input | ~40 segments/s |
| `/v1/translate-json` | 20 000 translatable strings, depth 200 | ~78 strings/s |
| `/v1/dict` | 200 chars per word | ~2.5–3.5 s per request |

**Wall-clock estimates for `/v1/translate-json`:**

| Strings | Time |
|---|---|
| 100 | ~3 s |
| 500 | ~8 s |
| 1 000 | ~15 s |
| 5 000 | ~70 s |
| 10 000 | ~2 min |
| 20 000 | ~4–5 min |

**Per-month quotas** are managed by your RapidAPI subscription tier. Upgrade in the RapidAPI dashboard if you hit your monthly cap.

---

## 10. FAQ

**Q: What model powers the API?**
A: Qwen3-14B-AWQ on vLLM, with idiom-aware prompting and script-purity validation.

**Q: Can I get streaming responses?**
A: Not currently — every endpoint returns the full result in one response. Plan around the wall-clock figures in section 9.

**Q: Are my inputs logged?**
A: Only request metadata (size, language pair, timing) for monitoring. Content is not persisted.

**Q: Can I translate from Vietnamese to English (and back)?**
A: Yes — every supported language pair works in both directions.

**Q: Why is `translations` an array even for one input on `/v1/translate`?**
A: To keep client code simple — you never branch on input shape.

**Q: Why does the response sometimes include `detected_source_lang` per item, and sometimes not?**
A: When `source_lang=auto`, each item is auto-detected, so the API tells you what it found. With explicit `source_lang`, that's redundant — so you get `list[str]` instead.

**Q: What happens if my HTML has a `<script>` block I want translated?**
A: It's not. `<script>`, `<style>`, `<code>`, `<pre>`, `<kbd>`, `<samp>`, `<var>` are always skipped to avoid breaking your code/styles. If you want to translate string literals inside JS, extract them first and use `/v1/translate` or `/v1/json`.

**Q: Can I send Markdown?**
A: Markdown is plain text, so `/v1/translate` works — but inline syntax (`**bold**`, `[link](url)`) may be reordered or split. For best fidelity, render to HTML first and use `/v1/translate-html`.

**Q: Does `paths_to_exclude` support negative array indices?**
A: No — only positive indices and `*`. If you want "all except index 0", list each path explicitly or use a `common_keys_to_exclude` strategy.

---

## 11. Support

- **Docs & live spec:** the OpenAPI spec is published in the RapidAPI playground.
- **Issues, feature requests:** contact the API provider on the RapidAPI listing page.
- **Status & latency:** if you see persistent `502`s, please report — there's no scheduled maintenance window otherwise.

Happy translating! 🌍
