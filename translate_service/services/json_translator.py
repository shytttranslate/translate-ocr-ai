"""JSON-aware translation: walk tree → collect strings → batch translate → write back.

Pipeline:
1. Walk JSON tree recursively, mỗi string value lưu (parent_ref, key, path).
2. Skip filter: numbers, URLs, emails, UUID, codes, strings < 2 chars.
3. Skip theo paths_to_exclude (dot-notation + wildcard `*`).
4. Skip theo common_keys_to_exclude (match key bất kỳ depth).
5. Apply words_not_to_translate (mask qua placeholder, restore sau).
6. Batch translate qua existing translate_batch.
7. Write back vào parent_ref[key].

Path syntax:
- "product.media.img_desc"      — exact match (cũng match prefix subtree).
- "product.items.*.image_url"   — wildcard cho array index.
- Prefix subtree: "product.media" skip cả con cháu.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from services.html_translator import apply_ignore_terms, restore_ignore_terms
from services.translator import TranslationResult, translate_batch
from services.vllm_client import VllmEndpoint
from utils.logging import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Skip filters: detect strings không phải "human text"
# ----------------------------------------------------------------------------

# Pure number / currency / percent (cho phép thousand separator + decimal)
_RE_PURE_NUMBER = re.compile(
    r"^\s*[+-]?[\d,.\s]*\d[\d,.\s]*\s*[%‰]?\s*$"
)
# URL/URI
_RE_URL = re.compile(
    r"^\s*(?:https?|ftp|file|data|mailto|tel|sms|geo|magnet)://?",
    re.IGNORECASE,
)
# Email
_RE_EMAIL = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)
# UUID
_RE_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
# Hash MD5/SHA1/SHA256 etc (32-128 hex chars, no dash)
_RE_HASH = re.compile(r"^[0-9a-f]{32,128}$", re.IGNORECASE)
# Code/ID: A-Z, 0-9, _, -, dot — phải có ≥ 4 chars
_RE_CODE_ID = re.compile(r"^[A-Z0-9][A-Z0-9_\-.]{3,}$")
# Currency với symbol leading: $99, €1,234.56
_RE_CURRENCY = re.compile(
    r"^\s*[$€£¥₫₹₽¢]\s*[\d,.\s]+\s*$"
)
# ISO date / datetime
_RE_DATE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
# Pure punctuation/symbols
_RE_PUNCT_ONLY = re.compile(r"^[\W\d_\s]+$")


def is_translatable_string(s: str) -> bool:
    """Return True nếu string là human text đáng dịch."""
    if not s or not s.strip():
        return False
    stripped = s.strip()
    if len(stripped) < 2:
        return False
    if _RE_PURE_NUMBER.match(stripped):
        return False
    if _RE_CURRENCY.match(stripped):
        return False
    if _RE_URL.match(stripped):
        return False
    if _RE_EMAIL.match(stripped):
        return False
    if _RE_UUID.match(stripped):
        return False
    if _RE_HASH.match(stripped):
        return False
    if _RE_DATE.match(stripped):
        return False
    if _RE_CODE_ID.match(stripped):
        return False
    # Pure punct/symbol — chỉ skip khi có ≥ 1 letter để dịch
    if not re.search(r"[A-Za-zÀ-ỹ぀-ヿ一-鿿가-힯]", stripped):
        return False
    return True


# ----------------------------------------------------------------------------
# Path matching
# ----------------------------------------------------------------------------

def _normalize_dotpath(path: str) -> tuple[str, ...]:
    """'a.b.c' → ('a', 'b', 'c'). Strip whitespace."""
    return tuple(p.strip() for p in path.split(".") if p.strip())


def path_matches_excluded(
    path: tuple[str, ...], excluded_patterns: list[tuple[str, ...]]
) -> bool:
    """Check current path against excluded patterns (with `*` wildcard).

    Pattern khớp khi pattern là PREFIX của path (item phù hợp với segment),
    hoặc pattern khớp đúng path. Vd:
    - pattern ('product','media','img_desc') khớp path ('product','media','img_desc')
      VÀ khớp path ('product','media','img_desc','sub','x') (subtree).
    - pattern ('items','*','name') khớp ('items','0','name'), ('items','1','name').
    """
    for pat in excluded_patterns:
        if len(pat) > len(path):
            continue
        ok = True
        for i, seg in enumerate(pat):
            if seg == "*":
                continue
            if path[i] != seg:
                ok = False
                break
        if ok:
            return True
    return False


# ----------------------------------------------------------------------------
# Walk + collect
# ----------------------------------------------------------------------------

@dataclass
class _StringRef:
    parent: Any  # dict hoặc list
    key: Any  # str (dict key) hoặc int (list index)
    text: str
    path: tuple[str, ...]
    keep_map: dict[int, str] = field(default_factory=dict)


@dataclass
class JsonTranslationResult:
    json_data: Any
    detected_source_lang: str | None
    strings_translated: int
    strings_skipped: int
    chars_translated: int
    warnings: list[str] = field(default_factory=list)


def collect_string_refs(
    obj: Any,
    excluded_paths: list[tuple[str, ...]],
    excluded_keys: set[str],
    skip_filter: bool = True,
) -> tuple[list[_StringRef], int]:
    """Walk JSON tree, return (refs_to_translate, skipped_count)."""
    refs: list[_StringRef] = []
    skipped = 0

    def walk(node: Any, path: tuple[str, ...]) -> None:
        nonlocal skipped
        if isinstance(node, dict):
            for k, v in node.items():
                if not isinstance(k, str):
                    continue
                new_path = path + (k,)
                # common_keys_to_exclude: skip toàn subtree khi key match
                if k in excluded_keys:
                    continue
                # paths_to_exclude
                if path_matches_excluded(new_path, excluded_paths):
                    continue
                if isinstance(v, str):
                    if skip_filter and not is_translatable_string(v):
                        skipped += 1
                        continue
                    refs.append(_StringRef(parent=node, key=k, text=v, path=new_path))
                elif isinstance(v, (dict, list)):
                    walk(v, new_path)
                else:
                    # number, bool, null — skip
                    pass
        elif isinstance(node, list):
            for i, v in enumerate(node):
                new_path = path + (str(i),)
                if path_matches_excluded(new_path, excluded_paths):
                    continue
                if isinstance(v, str):
                    if skip_filter and not is_translatable_string(v):
                        skipped += 1
                        continue
                    refs.append(_StringRef(parent=node, key=i, text=v, path=new_path))
                elif isinstance(v, (dict, list)):
                    walk(v, new_path)

    walk(obj, ())
    return refs, skipped


# ----------------------------------------------------------------------------
# Parse 3 option fields (string với `;` separator HOẶC list)
# ----------------------------------------------------------------------------

def parse_string_or_list(value: Any, sep: str = ";") -> list[str]:
    """Accept str (split by `sep`) hoặc list[str]. Return clean list."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [p.strip() for p in value.split(sep)]
    elif isinstance(value, list):
        items = [str(p).strip() for p in value]
    else:
        return []
    return [p for p in items if p]


def parse_excluded_paths(value: Any) -> list[tuple[str, ...]]:
    """Parse paths_to_exclude → list of tuple segments."""
    raw = parse_string_or_list(value)
    return [_normalize_dotpath(p) for p in raw if p]


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------

# Limits
MAX_JSON_STRINGS = 2000  # số string max trong 1 JSON
MAX_JSON_DEPTH = 50  # nesting depth tối đa


class JsonTooLarge(Exception):
    """JSON quá lớn / quá sâu."""


def _max_depth(obj: Any, current: int = 0) -> int:
    if isinstance(obj, dict):
        if not obj:
            return current
        return max(_max_depth(v, current + 1) for v in obj.values())
    if isinstance(obj, list):
        if not obj:
            return current
        return max(_max_depth(v, current + 1) for v in obj)
    return current


async def translate_json(
    *,
    json_data: Any,
    source_lang: str,
    target_lang: str,
    endpoint: VllmEndpoint,
    words_not_to_translate: list[str] | None = None,
    paths_to_exclude: list[tuple[str, ...]] | None = None,
    common_keys_to_exclude: list[str] | None = None,
    ignore_case: bool = False,
    skip_non_text: bool = True,
) -> JsonTranslationResult:
    """Translate string values trong JSON, preserve structure + skip rules.

    json_data: dict, list hoặc string. Mutated in-place đồng thời trả ra.
    """
    words_not_to_translate = words_not_to_translate or []
    paths_to_exclude = paths_to_exclude or []
    common_keys_to_exclude = common_keys_to_exclude or []
    warnings: list[str] = []

    # Sanity
    depth = _max_depth(json_data)
    if depth > MAX_JSON_DEPTH:
        raise JsonTooLarge(f"JSON nesting depth {depth} > {MAX_JSON_DEPTH}")

    # Single string fast-path
    if isinstance(json_data, str):
        if skip_non_text and not is_translatable_string(json_data):
            return JsonTranslationResult(
                json_data=json_data,
                detected_source_lang=None,
                strings_translated=0,
                strings_skipped=1,
                chars_translated=0,
            )
        masked, kmap = apply_ignore_terms(json_data, words_not_to_translate, ignore_case)
        results = await translate_batch(
            endpoint=endpoint,
            texts=[masked],
            source_lang=source_lang,
            target_lang=target_lang,
        )
        translated, w = restore_ignore_terms(results[0].translated_text, kmap)
        warnings.extend(w)
        return JsonTranslationResult(
            json_data=translated,
            detected_source_lang=results[0].detected_source_lang
            if source_lang == "auto" else None,
            strings_translated=1,
            strings_skipped=0,
            chars_translated=len(json_data),
            warnings=warnings,
        )

    # Walk tree, collect refs
    excluded_keys_set = set(common_keys_to_exclude)
    refs, skipped = collect_string_refs(
        json_data,
        excluded_paths=paths_to_exclude,
        excluded_keys=excluded_keys_set,
        skip_filter=skip_non_text,
    )

    if len(refs) > MAX_JSON_STRINGS:
        raise JsonTooLarge(
            f"JSON có {len(refs)} translatable strings > {MAX_JSON_STRINGS}"
        )

    if not refs:
        return JsonTranslationResult(
            json_data=json_data,
            detected_source_lang=None,
            strings_translated=0,
            strings_skipped=skipped,
            chars_translated=0,
        )

    # Apply words_not_to_translate per ref
    masked_texts: list[str] = []
    for ref in refs:
        masked, kmap = apply_ignore_terms(ref.text, words_not_to_translate, ignore_case)
        masked_texts.append(masked)
        ref.keep_map = kmap

    chars_total = sum(len(t) for t in masked_texts)

    # Batch translate
    results: list[TranslationResult] = await translate_batch(
        endpoint=endpoint,
        texts=masked_texts,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    # Write back + collect detected langs
    detected_langs: list[str] = []
    for ref, res in zip(refs, results):
        translated, w = restore_ignore_terms(res.translated_text, ref.keep_map)
        if w:
            warnings.extend(f"{'.'.join(ref.path)}: {x}" for x in w)
        # Write back vào parent
        if isinstance(ref.parent, dict):
            ref.parent[ref.key] = translated
        else:  # list
            ref.parent[ref.key] = translated
        if res.detected_source_lang and res.detected_source_lang != "unknown":
            detected_langs.append(res.detected_source_lang)

    detected = (
        max(set(detected_langs), key=detected_langs.count)
        if detected_langs and source_lang == "auto"
        else (None if source_lang == "auto" else source_lang)
    )

    return JsonTranslationResult(
        json_data=json_data,
        detected_source_lang=detected,
        strings_translated=len(refs),
        strings_skipped=skipped,
        chars_translated=chars_total,
        warnings=warnings,
    )
