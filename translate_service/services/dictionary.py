"""Logic dictionary lookup: build prompt Cambridge-style, gọi vLLM, parse JSON."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx
from json_repair import repair_json

from services.translator import LANG_NAMES
from services.vllm_client import VllmEndpoint
from utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DictResult:
    headword: str
    ipa: str
    definitions: list[dict]


def _build_dict_prompt(native_lang: str, target_lang: str) -> str:
    native = LANG_NAMES.get(native_lang, native_lang)
    target = LANG_NAMES.get(target_lang, target_lang)

    return (
        f"You are a professional bilingual dictionary engine. The user types a word in "
        f"{native} ({native_lang}). Look up its equivalent in {target} ({target_lang}) "
        f"and provide a Cambridge-style dictionary entry.\n\n"
        f"Output ONLY a valid JSON object with EXACTLY this schema:\n"
        f"{{\n"
        f'  "headword": "<the equivalent word/phrase in {target}>",\n'
        f'  "ipa": "<IPA phonetic transcription wrapped in slashes, e.g. /ˈfriː.dəm/>",\n'
        f'  "definitions": [\n'
        f"    {{\n"
        f'      "part_of_speech": "<noun|verb|adjective|adverb|preposition|conjunction|interjection|phrase>",\n'
        f'      "definition_target": "<concise Cambridge-style definition in {target}>",\n'
        f'      "definition_native": "<accurate translation of the definition into {native}>",\n'
        f'      "examples": ["<example sentence 1 in {target}>", "<example sentence 2>"]\n'
        f"    }}\n"
        f"  ]\n"
        f"}}\n\n"
        f"Rules:\n"
        f"- Provide 1–5 definitions covering the most common meanings.\n"
        f"- 1–3 natural example sentences per definition (full sentences, real usage).\n"
        f"- IPA must use standard International Phonetic Alphabet, wrapped in slashes.\n"
        f"- For Vietnamese in any field: preserve all diacritics.\n"
        f"- For CJK languages: use the appropriate native script.\n"
        f"- NO markdown fence, NO preamble, NO trailing text — output PURE JSON only.\n\n"
        f'Example (input "tự do", vi → en):\n'
        f'{{"headword":"freedom","ipa":"/ˈfriː.dəm/","definitions":[{{"part_of_speech":"noun","definition_target":"the right to act, speak, or think as one wants without hindrance or restraint","definition_native":"quyền được hành động, nói, hoặc suy nghĩ theo ý muốn mà không bị cản trở","examples":["We must defend our freedom of speech.","The people fought for their freedom from oppression."]}}]}}'
    )


def _parse_dict_response(raw: str) -> DictResult:
    """Parse JSON, trả DictResult. Robust với markdown fence + JSON broken."""
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
        return DictResult(headword="", ipa="", definitions=[])

    headword = str(parsed.get("headword", "")).strip()
    ipa = str(parsed.get("ipa", "")).strip()
    raw_defs = parsed.get("definitions") or []
    definitions: list[dict] = []

    if isinstance(raw_defs, list):
        for d in raw_defs:
            if not isinstance(d, dict):
                continue
            examples = d.get("examples") or []
            if not isinstance(examples, list):
                examples = []
            definitions.append(
                {
                    "part_of_speech": str(d.get("part_of_speech", "")).strip().lower(),
                    "definition_target": str(d.get("definition_target", "")).strip(),
                    "definition_native": str(d.get("definition_native", "")).strip(),
                    "examples": [str(e).strip() for e in examples if str(e).strip()],
                }
            )

    return DictResult(headword=headword, ipa=ipa, definitions=definitions)


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
            {"role": "user", "content": word},
        ],
        "max_tokens": 2048,
        "temperature": 0.2,
        # Qwen3 disable thinking mode (giống translator). vLLM cũ sẽ ignore.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        resp = await endpoint.chat_completion(payload)
    except httpx.HTTPError as exc:
        log.error("dict_vllm_call_failed", word=word, error=str(exc))
        raise

    raw = resp["choices"][0]["message"]["content"]
    return _parse_dict_response(raw)
