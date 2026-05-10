"""Logic translation: build prompt, gọi vLLM, parse response.

Theo chỉ đạo anh Thịnh:
- Hỗ trợ source_lang=auto (model tự detect) hoặc explicit code
- KHÔNG cache, KHÔNG domain awareness
- Output JSON structured qua prompt + json-repair fallback
- Validation script purity post-process: catch case model output mix script
  (vd dịch vi nhưng output có chữ Trung) → retry 1 lần với prompt mạnh hơn
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import httpx
from json_repair import repair_json

from services.vllm_client import VllmEndpoint
from utils.logging import get_logger

log = get_logger(__name__)

LANG_NAMES = {
    "auto": "automatically detected",
    "en": "English",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "zh": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "th": "Thai",
    "id": "Indonesian",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "hi": "Hindi",
}

# Threshold tỷ lệ char đúng script tối thiểu để coi output OK
SCRIPT_PURITY_THRESHOLD = 0.85


@dataclass
class TranslationResult:
    translated_text: str
    detected_source_lang: str
    warnings: list[str] = field(default_factory=list)


def _lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


_CORE_RULES = (
    "Rules:\n"
    "- Output 100% in {target} native script (romanize foreign words if needed).\n"
    "- Translate idioms by meaning, never literally.\n"
    "- Preserve names, numbers, code, URLs, emails, HTML/MD tags, register."
)


@lru_cache(maxsize=256)
def _build_system_prompt(source_lang: str, target_lang: str, retry: bool = False) -> str:
    """Prompt yêu cầu model detect (nếu auto) + translate + output JSON.

    Cache theo (source_lang, target_lang, retry) — prompt không đổi với cùng input.
    """
    target_name = _lang_name(target_lang)
    rules = _CORE_RULES.format(target=target_name)
    retry_line = (
        f"\nNOTE: previous output mixed scripts — output MUST be 100% {target_name} script."
        if retry
        else ""
    )

    if source_lang == "auto":
        return (
            f"Translate the message to {target_name} ({target_lang}). "
            f"Detect source language as ISO 639-1.\n"
            f"{rules}\n"
            f'Output JSON only: {{"detected_lang":"<ISO>","translation":"<text>"}}'
            f"{retry_line}"
        )

    source_name = _lang_name(source_lang)
    return (
        f"Translate from {source_name} ({source_lang}) to {target_name} ({target_lang}).\n"
        f"{rules}\n"
        f'Output JSON only: {{"translation":"<text>"}}'
        f"{retry_line}"
    )


# Regex pattern để check script. Chấp nhận chung: digit, whitespace, ASCII punct/symbol.
_NEUTRAL_RE = re.compile(r"[\d\s -/:-@[-`{-~ -¿]")
# Vietnamese: Latin + diacritics (Latin-1 supplement, Latin Extended A/B, Vietnamese specifics)
_LATIN_RE = re.compile(
    r"[A-Za-zÀ-ɏḀ-ỿ]"
)
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")  # CJK Unified
_HIRAGANA_RE = re.compile(r"[぀-ゟ]")
_KATAKANA_RE = re.compile(r"[゠-ヿｦ-ﾟ]")
_HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")
_THAI_RE = re.compile(r"[฀-๿]")
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
# Japanese expected: CJK + hiragana + katakana
_JA_EXPECTED_RE = re.compile("[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3040-\u309F\u30A0-\u30FF\uFF66-\uFF9F]")
# JSON markdown fence stripping
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _check_script_purity(text: str, target_lang: str) -> tuple[bool, float, str]:
    """Tính tỷ lệ char đúng script target. Trả (ok, ratio, foreign_sample).

    Char neutral (digit, space, ASCII punct) không tính (cả expected lẫn foreign).
    """
    if not text:
        return True, 1.0, ""

    expected_re: re.Pattern[str] | None
    if target_lang in ("vi", "en", "fr", "de", "es", "pt", "it", "id"):
        expected_re = _LATIN_RE
        foreign_res = [
            ("CJK", _CJK_RE),
            ("Hangul", _HANGUL_RE),
            ("Hiragana", _HIRAGANA_RE),
            ("Katakana", _KATAKANA_RE),
            ("Arabic", _ARABIC_RE),
            ("Thai", _THAI_RE),
        ]
    elif target_lang in ("zh", "zh-TW"):
        expected_re = _CJK_RE
        foreign_res = [
            ("Hangul", _HANGUL_RE),
            ("Hiragana", _HIRAGANA_RE),
            ("Katakana", _KATAKANA_RE),
            ("Arabic", _ARABIC_RE),
            ("Thai", _THAI_RE),
        ]
    elif target_lang == "ja":
        expected_re = _JA_EXPECTED_RE
        foreign_res = [
            ("Hangul", _HANGUL_RE),
            ("Arabic", _ARABIC_RE),
            ("Thai", _THAI_RE),
        ]
    elif target_lang == "ko":
        expected_re = _HANGUL_RE
        foreign_res = [
            ("CJK_only", _CJK_RE),
            ("Hiragana", _HIRAGANA_RE),
            ("Katakana", _KATAKANA_RE),
            ("Arabic", _ARABIC_RE),
            ("Thai", _THAI_RE),
        ]
    elif target_lang == "th":
        expected_re = _THAI_RE
        foreign_res = [("CJK", _CJK_RE), ("Hangul", _HANGUL_RE), ("Arabic", _ARABIC_RE)]
    elif target_lang == "ar":
        expected_re = _ARABIC_RE
        foreign_res = [("CJK", _CJK_RE), ("Hangul", _HANGUL_RE), ("Thai", _THAI_RE)]
    elif target_lang == "hi":
        expected_re = _DEVANAGARI_RE
        foreign_res = [("CJK", _CJK_RE), ("Hangul", _HANGUL_RE), ("Arabic", _ARABIC_RE)]
    elif target_lang == "ru":
        expected_re = _CYRILLIC_RE
        foreign_res = [("CJK", _CJK_RE), ("Hangul", _HANGUL_RE), ("Thai", _THAI_RE)]
    else:
        # Lang khác — không check, pass-through
        return True, 1.0, ""

    expected = 0
    foreign_chars: list[str] = []
    foreign_total = 0
    for ch in text:
        if _NEUTRAL_RE.match(ch):
            continue
        if expected_re.match(ch):
            expected += 1
            continue
        for _, fr in foreign_res:
            if fr.match(ch):
                foreign_total += 1
                if len(foreign_chars) < 12:
                    foreign_chars.append(ch)
                break

    total = expected + foreign_total
    if total == 0:
        return True, 1.0, ""

    ratio = expected / total
    ok = ratio >= SCRIPT_PURITY_THRESHOLD
    sample = "".join(foreign_chars)
    return ok, ratio, sample


def _parse_response(raw: str, source_lang: str) -> tuple[str, str]:
    """Parse JSON output của model, trả về (translation, detected_lang).

    Robust với:
    - JSON valid → trực tiếp parse
    - JSON kèm markdown fence ```json...``` → strip
    - JSON broken → json-repair
    - Plain text fallback → trả raw
    """
    cleaned = raw.strip()

    # Strip markdown code fence nếu có
    fence = _JSON_FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    parsed: dict | None = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(cleaned)
            parsed = json.loads(repaired) if repaired else None
        except (json.JSONDecodeError, ValueError):
            parsed = None

    if not isinstance(parsed, dict):
        log.warning("translation_json_parse_failed", raw_preview=cleaned[:200])
        return cleaned, source_lang if source_lang != "auto" else "unknown"

    translation = str(parsed.get("translation") or parsed.get("translated_text") or "").strip()
    if not translation:
        translation = cleaned

    if source_lang == "auto":
        detected = str(parsed.get("detected_lang") or parsed.get("source_lang") or "unknown").strip().lower()[:5]
    else:
        detected = source_lang

    return translation, detected


async def _call_vllm_translate(
    *,
    endpoint: VllmEndpoint,
    text: str,
    source_lang: str,
    target_lang: str,
    retry: bool,
) -> tuple[str, str]:
    """1 lần call vLLM, parse, trả (translation, detected_lang)."""
    system = _build_system_prompt(source_lang, target_lang, retry=retry)
    payload = {
        "model": endpoint.served_model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "max_tokens": min(4096, max(256, len(text) * 4)),
        "temperature": 0.05,  # near-greedy, ổn định hơn 0.1
        # vLLM/OpenAI guided JSON — model bị ép output JSON hợp lệ ngay,
        # giảm parse-fail + json-repair fallback path.
        "response_format": {"type": "json_object"},
        # Qwen3 default ON "thinking mode" → output có <think>...</think>
        # blocks, làm tăng latency + break JSON parse. Disable.
        # Field này yêu cầu vLLM 0.7+; với vLLM cũ sẽ bị ignore (silently).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = await endpoint.chat_completion(payload)
    raw = resp["choices"][0]["message"]["content"]
    return _parse_response(raw, source_lang)


async def translate_one(
    *,
    endpoint: VllmEndpoint,
    text: str,
    source_lang: str,
    target_lang: str,
) -> TranslationResult:
    """Dịch 1 text. Pass 1 + retry 1 nếu output mix script."""
    warnings: list[str] = []

    try:
        translated, detected = await _call_vllm_translate(
            endpoint=endpoint,
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            retry=False,
        )
    except httpx.HTTPError as exc:
        log.error("vllm_call_failed", error=str(exc), text_len=len(text))
        raise

    ok, ratio, foreign_sample = _check_script_purity(translated, target_lang)
    if not ok:
        log.warning(
            "translation_script_mixed",
            target_lang=target_lang,
            ratio=round(ratio, 3),
            foreign_sample=foreign_sample,
            text_len=len(text),
        )
        warnings.append(f"script_mixed_pass1: ratio={ratio:.2f} foreign={foreign_sample!r}")

        try:
            translated_retry, detected_retry = await _call_vllm_translate(
                endpoint=endpoint,
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                retry=True,
            )
        except httpx.HTTPError as exc:
            log.warning("vllm_retry_failed", error=str(exc))
        else:
            ok2, ratio2, foreign2 = _check_script_purity(translated_retry, target_lang)
            if ok2 or ratio2 > ratio:
                # Retry tốt hơn → dùng output retry
                translated = translated_retry
                detected = detected_retry
                if ok2:
                    warnings.append("script_fixed_on_retry")
                else:
                    warnings.append(
                        f"script_still_mixed_after_retry: ratio={ratio2:.2f}"
                    )
            else:
                warnings.append("retry_no_improvement")

    return TranslationResult(
        translated_text=translated,
        detected_source_lang=detected,
        warnings=warnings,
    )


async def translate_batch(
    *,
    endpoint: VllmEndpoint,
    texts: list[str],
    source_lang: str,
    target_lang: str,
    max_concurrency: int = 16,
) -> list[TranslationResult]:
    """Translate nhiều text song song. vLLM continuous batching tự gộp request.

    Concurrency 16 — vLLM continuous batching gom request đồng thời thành 1
    forward pass GPU, concurrency cao hơn → batch GPU lớn hơn → throughput tăng.
    Pool httpx 200 connection (xem VllmEndpoint) đủ buffer cho ~12 concurrent client.
    Nếu 1 text fail → trả TranslationResult với warnings, không fail toàn batch.
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def _bounded(t: str) -> TranslationResult:
        async with sem:
            try:
                return await translate_one(
                    endpoint=endpoint,
                    text=t,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            except Exception as exc:  # noqa: BLE001 — partial success cho batch
                err_kind = type(exc).__name__
                err_msg = str(exc) or err_kind
                log.warning("translate_batch_item_failed", error=err_msg, kind=err_kind, text_len=len(t))
                return TranslationResult(
                    translated_text="",
                    detected_source_lang=source_lang if source_lang != "auto" else "unknown",
                    warnings=[f"item_failed: {err_kind}: {err_msg}"[:200]],
                )

    return await asyncio.gather(*[_bounded(t) for t in texts])
