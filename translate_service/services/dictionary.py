"""Dictionary lookup đa ngôn ngữ — build prompt → vLLM → parse JSON.

Format output (camelCase JSON) gồm: phonetic (ipa+romanization), shortMeaning,
definitions, examples, phrases, related (synonyms/antonyms/relatedWords/memoryTips).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx
from json_repair import repair_json

from services.translator import LANG_NAMES
from services.vllm_client import VllmEndpoint
from utils.logging import get_logger

log = get_logger(__name__)

# Ngôn ngữ non-Latin → cần romanization. Latin (en, fr, de, es, it, pt, vi, id) → null.
_NON_LATIN_LANGS = {"ja", "zh", "zh-TW", "ko", "ru", "th", "ar", "hi"}

_VALID_POS = {
    "noun", "verb", "adjective", "adverb", "preposition", "conjunction",
    "interjection", "pronoun", "determiner", "particle", "phrase", "idiom",
}


@dataclass
class DictResult:
    word: str = ""
    phonetic_ipa: str | None = None
    phonetic_romanization: str | None = None
    short_meaning: str = ""
    definitions: list[dict] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    phrases: list[dict] = field(default_factory=list)
    synonyms: list[dict] = field(default_factory=list)
    antonyms: list[dict] = field(default_factory=list)
    related_words: list[dict] = field(default_factory=list)
    memory_tips: list[str] = field(default_factory=list)


def _build_dict_prompt(native_lang: str, target_lang: str) -> str:
    """Build prompt ngắn gọn cho người học ngoại ngữ.

    - native_lang = mẹ đẻ (output meanings ở đây)
    - target_lang = ngoại ngữ đang học (word + examples ở đây)
    """
    native_name = LANG_NAMES.get(native_lang, native_lang)
    target_name = LANG_NAMES.get(target_lang, target_lang)
    needs_roman = target_lang in _NON_LATIN_LANGS
    roman_rule = (
        f"romanization REQUIRED for {target_name} (pinyin/romaji/etc)"
        if needs_roman
        else "romanization=null (Latin script)"
    )

    return (
        f"You are a bilingual dictionary for a {native_name} learner studying {target_name}. "
        f"User gives a word in {target_name}; you return JSON.\n\n"
        f"=== LANGUAGE OF EACH FIELD (CRITICAL — DO NOT MIX UP) ===\n"
        f"Fields written in {target_name} ({target_lang}):\n"
        f"  • word\n"
        f"  • examples[].text\n"
        f"  • phrases[].text\n"
        f"  • related.synonyms[].text, related.antonyms[].text, related.relatedWords[].text\n"
        f"Fields written in {native_name} ({native_lang}) — ALWAYS the learner's native language:\n"
        f"  • shortMeaning\n"
        f"  • definitions[].meaning\n"
        f"  • examples[].meaning\n"
        f"  • phrases[].meaning\n"
        f"  • related.synonyms[].meaning, related.antonyms[].meaning, related.relatedWords[].meaning\n"
        f"  • related.memoryTips (each item)\n"
        f"Even if the user input is in {target_name}, every meaning/explanation MUST be in {native_name}.\n\n"
        f"Counts: definitions 1-2, examples 1-2, phrases 1-2, synonyms 1-2, "
        f"antonyms 0-1, relatedWords 1-2, memoryTips 1.\n"
        f"Keep every meaning/text under 15 words.\n"
        f"phonetic.ipa MUST use slashes (e.g. /bʊk/). {roman_rule}.\n"
        f"partOfSpeech lowercase from: noun|verb|adjective|adverb|preposition|"
        f"conjunction|interjection|pronoun|determiner|particle|phrase|idiom.\n"
        f"shortMeaning ≤80 chars, semicolons between senses. Preserve diacritics & native scripts. "
        f"NO markdown fence, NO preamble — pure JSON."
    )


def _build_json_schema() -> dict:
    """JSON schema cho vLLM guided decoding — match DictResponse output (camelCase)."""
    pos_enum = [
        "noun", "verb", "adjective", "adverb", "preposition", "conjunction",
        "interjection", "pronoun", "determiner", "particle", "phrase", "idiom",
    ]
    pair = {
        "type": "object",
        "properties": {"text": {"type": "string"}, "meaning": {"type": "string"}},
        "required": ["text", "meaning"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "word": {"type": "string"},
            "phonetic": {
                "type": "object",
                "properties": {
                    "ipa": {"type": ["string", "null"]},
                    "romanization": {"type": ["string", "null"]},
                },
                "required": ["ipa", "romanization"],
                "additionalProperties": False,
            },
            "shortMeaning": {"type": "string"},
            "definitions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "partOfSpeech": {"type": "string", "enum": pos_enum},
                        "meaning": {"type": "string"},
                    },
                    "required": ["partOfSpeech", "meaning"],
                    "additionalProperties": False,
                },
            },
            "examples": {"type": "array", "minItems": 1, "maxItems": 2, "items": pair},
            "phrases": {"type": "array", "minItems": 0, "maxItems": 2, "items": pair},
            "related": {
                "type": "object",
                "properties": {
                    "synonyms": {"type": "array", "minItems": 0, "maxItems": 2, "items": pair},
                    "antonyms": {"type": "array", "minItems": 0, "maxItems": 1, "items": pair},
                    "relatedWords": {"type": "array", "minItems": 0, "maxItems": 2, "items": pair},
                    "memoryTips": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"type": "string"},
                    },
                },
                "required": ["synonyms", "antonyms", "relatedWords", "memoryTips"],
                "additionalProperties": False,
            },
        },
        "required": [
            "word", "phonetic", "shortMeaning", "definitions",
            "examples", "phrases", "related",
        ],
        "additionalProperties": False,
    }


_DICT_JSON_SCHEMA = _build_json_schema()


def _coerce_str(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _coerce_pair_list(raw: object, field_name: str) -> list[dict]:
    """Parse list of {text, meaning} (skip invalid entries)."""
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = _coerce_str(item.get("text"))
        meaning = _coerce_str(item.get("meaning"))
        if not text:
            continue
        out.append({"text": text, "meaning": meaning})
    return out


def _parse_dict_response(raw: str) -> DictResult:
    """Parse JSON response. Robust với markdown fence + JSON broken."""
    cleaned = raw.strip()

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
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
        log.warning("dict_json_parse_failed", raw_preview=cleaned[:300])
        return DictResult()

    word = _coerce_str(parsed.get("word"))

    phonetic = parsed.get("phonetic") or {}
    if isinstance(phonetic, dict):
        ipa = _coerce_str(phonetic.get("ipa")) or None
        roman = _coerce_str(phonetic.get("romanization")) or None
    else:
        ipa = None
        roman = None

    short_meaning = _coerce_str(parsed.get("shortMeaning") or parsed.get("short_meaning"))

    raw_defs = parsed.get("definitions") or []
    definitions: list[dict] = []
    if isinstance(raw_defs, list):
        for d in raw_defs:
            if not isinstance(d, dict):
                continue
            pos = _coerce_str(d.get("partOfSpeech") or d.get("part_of_speech")).lower()
            if pos not in _VALID_POS:
                pos = "noun"
            meaning = _coerce_str(d.get("meaning"))
            if not meaning:
                continue
            definitions.append({"part_of_speech": pos, "meaning": meaning})

    examples = _coerce_pair_list(parsed.get("examples"), "examples")
    phrases = _coerce_pair_list(parsed.get("phrases"), "phrases")

    related = parsed.get("related") or {}
    if not isinstance(related, dict):
        related = {}
    synonyms = _coerce_pair_list(related.get("synonyms"), "synonyms")
    antonyms = _coerce_pair_list(related.get("antonyms"), "antonyms")
    related_words = _coerce_pair_list(
        related.get("relatedWords") or related.get("related_words"), "relatedWords"
    )
    raw_tips = related.get("memoryTips") or related.get("memory_tips") or []
    memory_tips: list[str] = []
    if isinstance(raw_tips, list):
        memory_tips = [_coerce_str(t) for t in raw_tips if _coerce_str(t)]

    return DictResult(
        word=word,
        phonetic_ipa=ipa,
        phonetic_romanization=roman,
        short_meaning=short_meaning,
        definitions=definitions,
        examples=examples,
        phrases=phrases,
        synonyms=synonyms,
        antonyms=antonyms,
        related_words=related_words,
        memory_tips=memory_tips,
    )


async def dict_lookup(
    *,
    endpoint: VllmEndpoint,
    word: str,
    native_lang: str,
    target_lang: str,
) -> DictResult:
    system = _build_dict_prompt(native_lang, target_lang)
    payload = {
        "model": endpoint.served_model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": word.strip()},
        ],
        "max_tokens": 600,
        "temperature": 0.0,
        "top_p": 1.0,
        # Qwen3 disable thinking mode (giống translator). vLLM cũ sẽ ignore.
        "chat_template_kwargs": {"enable_thinking": False},
        # vLLM guided JSON: enforce schema → skip token vô nghĩa, giảm 10-20% latency.
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "dict_entry", "schema": _DICT_JSON_SCHEMA},
        },
    }

    try:
        resp = await endpoint.chat_completion(payload)
    except httpx.HTTPError as exc:
        log.error("dict_vllm_call_failed", word=word, error=str(exc))
        raise

    raw = resp["choices"][0]["message"]["content"]
    return _parse_dict_response(raw)
