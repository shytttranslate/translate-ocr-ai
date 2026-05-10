"""HTML-aware translation: parse → score → DOM walk → batch translate → reinsert.

Pipeline:
1. Parse HTML qua lxml.html (recover=True). Fallback html5lib khi error nặng.
2. Score health: error_rate + structure_diff. Severe → reject 422.
3. Walk DOM, build inline segments tại "leaf block" (block chỉ có inline children).
   Inline children được serialize với XLIFF-style `<g id=N>` placeholder.
4. Apply ignore_terms: replace exact match → `\\u2060KEEP\\u2060N\\u2060` placeholder.
5. Batch translate qua existing translate_batch (concurrency=16).
6. Restore ignore_terms → original. Restore inline placeholders → original tags.
7. Reinsert text vào DOM. Serialize lại HTML.
8. Verify post-tree: tag count match, no broken structure.

Skip subtree: script, style, noscript, svg, math, template, textarea, code, pre, kbd, samp, var.
Translate attributes (whitelist): alt, title, aria-label, placeholder, content (meta description).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from lxml import etree, html as lxml_html

from services.translator import TranslationResult, translate_batch
from services.vllm_client import VllmEndpoint
from utils.logging import get_logger

log = get_logger(__name__)

# ----------------------------------------------------------------------------
# Tag classifications
# ----------------------------------------------------------------------------

# Inline tags: text inside dịch chung với parent's segment, tag preserve qua placeholder.
INLINE_TAGS: frozenset[str] = frozenset({
    "a", "abbr", "acronym", "b", "bdi", "bdo", "br", "cite", "del", "dfn",
    "em", "i", "ins", "mark", "q", "rp", "rt", "ruby", "s", "small", "span",
    "strong", "sub", "sup", "u", "wbr", "img", "font", "tt",
})

# Opaque inline: tag giữ trong segment NHƯNG text bên trong KHÔNG dịch (vd: code, time).
OPAQUE_INLINE_TAGS: frozenset[str] = frozenset({
    "code", "kbd", "samp", "var", "time", "data",
})

# Skip toàn subtree — không walk, không dịch.
SKIP_SUBTREE_TAGS: frozenset[str] = frozenset({
    "script", "style", "noscript", "svg", "math", "template", "textarea",
    "pre",  # pre giữ formatting nguyên
})

# Whitelist attribute được dịch.
TRANSLATABLE_ATTRS: frozenset[str] = frozenset({
    "alt", "title", "aria-label", "placeholder",
})

# Tag mà attribute "content" được dịch (chỉ meta[name=description|keywords]).
META_TRANSLATE_NAMES: frozenset[str] = frozenset({"description", "keywords"})

# Placeholder cho ignore_terms — HTML void tag, model SOTA luôn preserve tags.
_KEEP_RE = re.compile(r'<x-keep\s+id="(\d+)"\s*/>', re.IGNORECASE)


def _keep_placeholder(idx: int) -> str:
    return f'<x-keep id="{idx}"/>'

# XLIFF-style inline placeholder. Model thấy là HTML tag → giữ.
_GID_OPEN_RE = re.compile(r'<g\s+id="(\d+)">', re.IGNORECASE)
_GID_CLOSE_RE = re.compile(r"</g>", re.IGNORECASE)
_GID_PAIR_RE = re.compile(r'<g\s+id="(\d+)">(.*?)</g>', re.IGNORECASE | re.DOTALL)
_GID_VOID_RE = re.compile(r'<g\s+id="(\d+)"\s*/>', re.IGNORECASE)

# Threshold health
HEALTH_MINOR_ERR = 0.01
HEALTH_MINOR_DIFF = 0.01
HEALTH_MODERATE_ERR = 0.10
HEALTH_MODERATE_DIFF = 0.05
FATAL_NESTING_DEPTH = 50
FATAL_TAG_DIFF_RATIO = 0.30


HealthLevel = Literal["clean", "minor", "moderate", "severe"]


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------

@dataclass
class HtmlHealth:
    health: HealthLevel
    error_rate: float
    structure_diff: float
    errors_total: int
    fatals_total: int
    parse_tier: Literal["lxml", "html5lib", "fragment_wrap"]
    sample_errors: list[dict[str, Any]] = field(default_factory=list)
    fatal_markers: list[str] = field(default_factory=list)


@dataclass
class _Segment:
    """1 đơn vị dịch — text node, inline-block, hoặc attribute value."""
    kind: Literal["inline_block", "attr"]
    text: str
    elem: etree._Element  # element gốc để write back
    attr: str | None = None  # với kind="attr"
    inline_map: list[dict[str, Any]] = field(default_factory=list)  # placeholder map


@dataclass
class HtmlTranslationResult:
    html: str
    health: HtmlHealth
    detected_source_lang: str | None
    segments_translated: int
    chars_translated: int
    warnings: list[str] = field(default_factory=list)


class HtmlTooMalformed(Exception):
    """Health=severe — caller phải sửa HTML trước."""

    def __init__(self, health: HtmlHealth) -> None:
        super().__init__(f"HTML too malformed: health={health.health}")
        self.health = health


# ----------------------------------------------------------------------------
# Parse + Health scoring
# ----------------------------------------------------------------------------

def _count_raw_tag_open(raw_html: str) -> int:
    """Đếm số tag open trong raw HTML (rough approximation)."""
    return len(re.findall(r"<[a-zA-Z][^>]*?>", raw_html))


def _adjusted_parsed_count(tree: etree._Element, raw_html: str) -> int:
    """Tag count trong tree trừ wrapper html/head/body/meta lxml auto-thêm."""
    parsed_total = sum(1 for _ in tree.iter() if isinstance(_.tag, str))
    auto_added = 0
    for tag in ("html", "head", "body", "meta"):
        in_tree = sum(
            1 for e in tree.iter()
            if isinstance(e.tag, str) and e.tag.lower() == tag
        )
        in_raw = len(re.findall(rf"<{tag}\b", raw_html, re.IGNORECASE))
        auto_added += max(0, in_tree - in_raw)
    return max(0, parsed_total - auto_added)


def _has_fatal_markers(raw_html: str, parse_errors: list, tree: etree._Element) -> list[str]:
    """Detect fatal markers — return list lý do."""
    fatals: list[str] = []

    fatal_count = sum(1 for e in parse_errors if e.level >= 3)
    if fatal_count > 0:
        fatals.append(f"libxml2_fatal: {fatal_count} fatal-level errors")

    if "\x00" in raw_html or raw_html.count("�") > 5:
        fatals.append("encoding_undecodable")

    max_depth = 0
    for elem in tree.iter():
        depth = 0
        node = elem
        while node.getparent() is not None:
            depth += 1
            node = node.getparent()
        max_depth = max(max_depth, depth)
    if max_depth > FATAL_NESTING_DEPTH:
        fatals.append(f"nesting_too_deep: {max_depth} > {FATAL_NESTING_DEPTH}")

    raw_open = _count_raw_tag_open(raw_html)
    parsed_count = _adjusted_parsed_count(tree, raw_html)
    if raw_open > 0:
        diff_ratio = abs(raw_open - parsed_count) / raw_open
        if diff_ratio > FATAL_TAG_DIFF_RATIO:
            fatals.append(
                f"tag_diff_too_high: raw_open={raw_open} parsed={parsed_count} "
                f"diff={diff_ratio:.2%}"
            )

    return fatals


def _score_health(
    parse_errors: list,
    tree: etree._Element,
    raw_html: str,
    parse_tier: Literal["lxml", "html5lib", "fragment_wrap"],
) -> HtmlHealth:
    """Compute health level từ parse_errors + structure metrics."""
    parsed_tag_count = _adjusted_parsed_count(tree, raw_html)
    raw_tag_count = _count_raw_tag_open(raw_html)

    errors_total = sum(1 for e in parse_errors if e.level == 2)
    fatals_total = sum(1 for e in parse_errors if e.level >= 3)
    weighted = errors_total + fatals_total * 5
    error_rate = weighted / max(parsed_tag_count, 1)

    if raw_tag_count > 0:
        structure_diff = abs(raw_tag_count - parsed_tag_count) / raw_tag_count
    else:
        structure_diff = 0.0

    fatal_markers = _has_fatal_markers(raw_html, parse_errors, tree)

    sample = [
        {
            "line": e.line,
            "column": e.column,
            "severity": ("warning", "error", "fatal")[min(e.level, 3) - 1] if e.level >= 1 else "info",
            "message": e.message[:200],
        }
        for e in parse_errors[:8]
    ]

    if fatal_markers or error_rate >= HEALTH_MODERATE_ERR or structure_diff >= HEALTH_MODERATE_DIFF:
        level: HealthLevel = "severe"
    elif error_rate >= HEALTH_MINOR_ERR or structure_diff >= HEALTH_MINOR_DIFF:
        level = "moderate"
    elif errors_total > 0 or fatals_total > 0:
        level = "minor"
    else:
        level = "clean"

    return HtmlHealth(
        health=level,
        error_rate=round(error_rate, 4),
        structure_diff=round(structure_diff, 4),
        errors_total=errors_total,
        fatals_total=fatals_total,
        parse_tier=parse_tier,
        sample_errors=sample,
        fatal_markers=fatal_markers,
    )


def parse_html(raw: str) -> tuple[etree._Element, HtmlHealth]:
    """Parse HTML với 2-tier strategy. Trả tree + health."""
    if not raw or not raw.strip():
        raise HtmlTooMalformed(
            HtmlHealth(
                health="severe",
                error_rate=1.0,
                structure_diff=1.0,
                errors_total=0,
                fatals_total=1,
                parse_tier="lxml",
                fatal_markers=["empty_input"],
            )
        )

    # Strip nul bytes (lxml refuse)
    raw_clean = raw.replace("\x00", "")

    # Detect fragment: nếu không có <html> hoặc <body> → wrap
    has_root = bool(re.search(r"<html\b", raw_clean, re.IGNORECASE))
    parse_tier: Literal["lxml", "html5lib", "fragment_wrap"] = "lxml"
    if not has_root:
        wrapped = (
            f'<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
            f'<body>{raw_clean}</body></html>'
        )
        parse_tier = "fragment_wrap"
    else:
        wrapped = raw_clean

    # Tier 1: lxml recover mode
    parser = lxml_html.HTMLParser(recover=True, encoding="utf-8")
    try:
        tree = lxml_html.fromstring(wrapped.encode("utf-8"), parser=parser)
    except (etree.ParserError, ValueError) as exc:
        log.warning("lxml_parse_failed", error=str(exc)[:200])
        # Tier 2: html5lib fallback
        try:
            import html5lib

            tree = html5lib.parse(wrapped, treebuilder="lxml", namespaceHTMLElements=False)
            # html5lib trả ElementTree, lấy root
            tree = tree.getroot() if hasattr(tree, "getroot") else tree
            parse_tier = "html5lib"
        except Exception as exc2:
            raise HtmlTooMalformed(
                HtmlHealth(
                    health="severe",
                    error_rate=1.0,
                    structure_diff=1.0,
                    errors_total=0,
                    fatals_total=1,
                    parse_tier="html5lib",
                    fatal_markers=[f"both_parsers_failed: {exc2!s}"[:200]],
                )
            ) from exc2

    # Score health từ tier-1 errors
    health = _score_health(parser.error_log, tree, raw_clean, parse_tier)

    # Nếu tier-1 ra severe nhưng chưa thử tier-2 → retry
    if health.health == "severe" and parse_tier == "lxml":
        try:
            import html5lib

            tree2 = html5lib.parse(wrapped, treebuilder="lxml", namespaceHTMLElements=False)
            tree2 = tree2.getroot() if hasattr(tree2, "getroot") else tree2
            # Re-score với tree mới (html5lib không expose error log → gọi lại _score_health với empty)
            health2 = _score_health([], tree2, raw_clean, "html5lib")
            if health2.health != "severe":
                tree = tree2
                health = health2
                log.info("html_tier2_recovered", from_health="severe", to_health=health.health)
        except Exception:
            pass  # giữ tier-1 kết quả

    if health.health == "severe":
        raise HtmlTooMalformed(health)

    return tree, health


# ----------------------------------------------------------------------------
# DOM walk + segment collection
# ----------------------------------------------------------------------------

def _tag_lower(elem: etree._Element) -> str:
    return elem.tag.lower() if isinstance(elem.tag, str) else ""


def _children_all_inlinish(elem: etree._Element) -> bool:
    """Element là 'leaf block' nếu mọi con đều inline/opaque/skip."""
    for c in elem:
        if not isinstance(c.tag, str):
            continue
        t = c.tag.lower()
        if t not in INLINE_TAGS and t not in OPAQUE_INLINE_TAGS and t not in SKIP_SUBTREE_TAGS:
            return False
    return True


def _serialize_inline_subtree(
    elem: etree._Element, inline_map: list[dict[str, Any]]
) -> str:
    """Convert subtree thành text với <g id=N> placeholder cho inline tags.

    inline_map (output): list[{name, attrs, opaque_html?}].
    """
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)

    for child in elem:
        if not isinstance(child.tag, str):
            if child.tail:
                parts.append(child.tail)
            continue

        ctag = child.tag.lower()

        if ctag in SKIP_SUBTREE_TAGS or ctag in OPAQUE_INLINE_TAGS:
            # Opaque — giữ nguyên HTML, void placeholder
            opaque = lxml_html.tostring(child, encoding="unicode", with_tail=False)
            idx = len(inline_map)
            inline_map.append({"opaque": opaque})
            parts.append(f'<g id="{idx}"/>')
        elif ctag in INLINE_TAGS:
            # Transparent inline — recurse text bên trong
            idx = len(inline_map)
            inline_map.append({"name": ctag, "attrs": dict(child.attrib)})
            inner = _serialize_inline_subtree(child, inline_map)
            # Self-closing tags như <br>, <img>, <wbr> không có text bên trong
            if ctag in ("br", "img", "wbr") and not inner:
                parts.append(f'<g id="{idx}"/>')
            else:
                parts.append(f'<g id="{idx}">{inner}</g>')
        else:
            # Block child trong leaf block — không nên xảy ra (đã check _children_all_inlinish)
            opaque = lxml_html.tostring(child, encoding="unicode", with_tail=False)
            idx = len(inline_map)
            inline_map.append({"opaque": opaque})
            parts.append(f'<g id="{idx}"/>')

        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def _has_skip_ancestor(elem: etree._Element) -> bool:
    p = elem.getparent()
    while p is not None:
        if isinstance(p.tag, str) and p.tag.lower() in SKIP_SUBTREE_TAGS:
            return True
        p = p.getparent()
    return False


def collect_segments(
    root: etree._Element, translate_attributes: bool
) -> list[_Segment]:
    """Walk DOM, collect translatable segments."""
    segments: list[_Segment] = []
    visited_for_inline: set[int] = set()

    def visit(elem: etree._Element) -> None:
        if not isinstance(elem.tag, str):
            return
        tag = elem.tag.lower()

        if tag in SKIP_SUBTREE_TAGS:
            return

        # Attributes
        if translate_attributes:
            for attr in TRANSLATABLE_ATTRS:
                if attr in elem.attrib:
                    val = elem.attrib[attr].strip()
                    if val:
                        segments.append(_Segment(kind="attr", text=val, elem=elem, attr=attr))
            # meta[name=description|keywords] content
            if tag == "meta":
                name = elem.attrib.get("name", "").lower()
                if name in META_TRANSLATE_NAMES and elem.attrib.get("content", "").strip():
                    segments.append(
                        _Segment(kind="attr", text=elem.attrib["content"], elem=elem, attr="content")
                    )

        # Inline-block check: element là leaf block → build 1 segment cho cả subtree
        is_leaf_block = (
            tag not in INLINE_TAGS
            and tag not in OPAQUE_INLINE_TAGS
            and _children_all_inlinish(elem)
            and (elem.text or len(elem) > 0)
        )
        if is_leaf_block:
            inline_map: list[dict[str, Any]] = []
            text = _serialize_inline_subtree(elem, inline_map)
            if text.strip():
                segments.append(
                    _Segment(
                        kind="inline_block",
                        text=text,
                        elem=elem,
                        inline_map=inline_map,
                    )
                )
            visited_for_inline.add(id(elem))
            # Vẫn recurse children (để pick attributes của inline children)
            for child in elem:
                visit(child)
            return

        # Recurse children
        for child in elem:
            visit(child)

    visit(root)
    return segments


# ----------------------------------------------------------------------------
# Inline placeholder restore (after translation)
# ----------------------------------------------------------------------------

def restore_inline_placeholders(
    translated: str, inline_map: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    """Replace `<g id=N>...</g>` → original tag. Stack-based parser handle nested.

    Returns (html_fragment, warnings).
    """
    warnings: list[str] = []
    used_ids: set[int] = set()
    out: list[str] = []
    stack: list[int] = []  # idx của <g> đang mở

    i = 0
    n = len(translated)
    while i < n:
        # Void: <g id="N"/>
        m = _GID_VOID_RE.match(translated, i)
        if m:
            idx = int(m.group(1))
            used_ids.add(idx)
            if idx < len(inline_map):
                info = inline_map[idx]
                if "opaque" in info:
                    out.append(info["opaque"])
                else:
                    attrs = "".join(
                        f' {k}="{_esc_attr(v)}"' for k, v in info.get("attrs", {}).items()
                    )
                    out.append(f'<{info["name"]}{attrs}/>')
            else:
                warnings.append(f"unknown_placeholder_id: {idx}")
            i = m.end()
            continue

        # Open: <g id="N">
        m = _GID_OPEN_RE.match(translated, i)
        if m:
            idx = int(m.group(1))
            used_ids.add(idx)
            if idx < len(inline_map):
                info = inline_map[idx]
                if "opaque" in info:
                    # Opaque = self-contained (script/code/...) — không có cặp close
                    out.append(info["opaque"])
                else:
                    attrs = "".join(
                        f' {k}="{_esc_attr(v)}"' for k, v in info.get("attrs", {}).items()
                    )
                    out.append(f'<{info["name"]}{attrs}>')
                    stack.append(idx)
            else:
                warnings.append(f"unknown_placeholder_id: {idx}")
                stack.append(-1)  # unknown, vẫn track close để không lệch
            i = m.end()
            continue

        # Close: </g>
        m = _GID_CLOSE_RE.match(translated, i)
        if m:
            if stack:
                idx = stack.pop()
                if idx >= 0 and idx < len(inline_map):
                    info = inline_map[idx]
                    if "opaque" not in info:
                        out.append(f'</{info["name"]}>')
            else:
                warnings.append("unmatched_g_close")
            i = m.end()
            continue

        out.append(translated[i])
        i += 1

    # Close unclosed
    while stack:
        idx = stack.pop()
        if idx >= 0 and idx < len(inline_map):
            info = inline_map[idx]
            if "opaque" not in info:
                out.append(f'</{info["name"]}>')
                warnings.append(f"auto_closed_unclosed_g: id={idx}")

    # Detect missing
    expected = set(range(len(inline_map)))
    missing = expected - used_ids
    if missing:
        warnings.append(f"inline_placeholders_lost: {sorted(missing)}")

    return "".join(out), warnings


def _esc_attr(v: str) -> str:
    return v.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


# ----------------------------------------------------------------------------
# ignore_terms: pre-mask + post-restore
# ----------------------------------------------------------------------------

def apply_ignore_terms(
    text: str, terms: list[str], ignore_case: bool
) -> tuple[str, dict[int, str]]:
    """Replace exact term match → `\\u2060KEEP\\u2060N\\u2060` placeholder.

    Returns (masked_text, {placeholder_id: original_term}).
    Match longest-first để tránh partial overlap.
    """
    if not terms:
        return text, {}

    # Sort longest first
    sorted_terms = sorted({t for t in terms if t}, key=len, reverse=True)
    flags = re.IGNORECASE if ignore_case else 0
    mapping: dict[int, str] = {}
    out = text
    next_id = 0

    for term in sorted_terms:
        pat = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags)

        def _sub(m: re.Match[str]) -> str:
            nonlocal next_id
            idx = next_id
            next_id += 1
            mapping[idx] = m.group(0)
            return _keep_placeholder(idx)

        out = pat.sub(_sub, out)

    return out, mapping


def restore_ignore_terms(
    text: str, mapping: dict[int, str]
) -> tuple[str, list[str]]:
    """Restore placeholder → original term. Cảnh báo nếu mất term."""
    if not mapping:
        return text, []
    warnings: list[str] = []
    used: set[int] = set()

    def _sub(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        used.add(idx)
        if idx not in mapping:
            warnings.append(f"unknown_keep_id: {idx}")
            return ""
        return mapping[idx]

    out = _KEEP_RE.sub(_sub, text)
    missing = set(mapping.keys()) - used
    if missing:
        terms_lost = [mapping[i] for i in sorted(missing)]
        warnings.append(f"ignore_terms_lost_in_translation: {terms_lost}")
        # Append at end để không mất hoàn toàn
        out = out + " " + " ".join(terms_lost)

    return out, warnings


# ----------------------------------------------------------------------------
# Reinsert translated segments back into DOM
# ----------------------------------------------------------------------------

def _replace_inline_block(elem: etree._Element, new_html_fragment: str) -> None:
    """Replace inner content của elem với HTML fragment đã translate.

    Giữ nguyên elem.tag + attributes; chỉ đổi text + children + tail (children only).
    """
    # Wrap fragment trong dummy để parse
    wrapped = f"<x>{new_html_fragment}</x>"
    try:
        new_elem = lxml_html.fragment_fromstring(wrapped, create_parent=False)
    except (etree.ParserError, etree.XMLSyntaxError):
        # Fragment broken → fallback: set as text only
        elem.text = re.sub(r"<[^>]+>", "", new_html_fragment)
        for child in list(elem):
            elem.remove(child)
        return

    # Drop existing children + text
    for child in list(elem):
        elem.remove(child)
    elem.text = new_elem.text or ""

    # Move children of new_elem vào elem
    for child in new_elem:
        elem.append(child)


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------

async def translate_html(
    *,
    raw_html: str,
    source_lang: str,
    target_lang: str,
    endpoint: VllmEndpoint,
    ignore_terms: list[str] | None = None,
    ignore_case: bool = False,
    translate_attributes: bool = True,
) -> HtmlTranslationResult:
    """Translate HTML preserve-structure. Raise HtmlTooMalformed nếu severe."""
    ignore_terms = ignore_terms or []
    warnings: list[str] = []

    # 1. Parse + score
    tree, health = parse_html(raw_html)

    # 2. Collect segments
    segments = collect_segments(tree, translate_attributes=translate_attributes)
    if not segments:
        # Không có gì để dịch — return original
        return HtmlTranslationResult(
            html=raw_html,
            health=health,
            detected_source_lang=None,
            segments_translated=0,
            chars_translated=0,
            warnings=["no_translatable_content"],
        )

    # 3. Apply ignore_terms per segment
    keep_maps: list[dict[int, str]] = []
    masked_texts: list[str] = []
    for seg in segments:
        masked, mp = apply_ignore_terms(seg.text, ignore_terms, ignore_case)
        masked_texts.append(masked)
        keep_maps.append(mp)

    # 4. Batch translate
    chars_translated = sum(len(t) for t in masked_texts)
    results: list[TranslationResult] = await translate_batch(
        endpoint=endpoint,
        texts=masked_texts,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    # 5. Restore ignore_terms + inline placeholders, write back
    detected_langs: list[str] = []
    for seg, res, keep_map in zip(segments, results, keep_maps):
        translated = res.translated_text
        if res.detected_source_lang and res.detected_source_lang != "unknown":
            detected_langs.append(res.detected_source_lang)

        # Restore ignore_terms
        translated, w_keep = restore_ignore_terms(translated, keep_map)
        warnings.extend(w_keep)

        # Restore inline placeholders + write back
        if seg.kind == "inline_block":
            html_frag, w_inline = restore_inline_placeholders(translated, seg.inline_map)
            warnings.extend(w_inline)
            _replace_inline_block(seg.elem, html_frag)
        else:  # attr
            seg.elem.set(seg.attr, translated)  # type: ignore[arg-type]

    # 6. Detected source lang dominant của batch
    detected = (
        max(set(detected_langs), key=detected_langs.count)
        if detected_langs and source_lang == "auto"
        else (None if source_lang == "auto" else source_lang)
    )

    # 7. Serialize lại
    serialized = lxml_html.tostring(tree, encoding="unicode", method="html")

    # 8. Bỏ wrapper nếu input là fragment
    if health.parse_tier == "fragment_wrap":
        # Extract body content
        m = re.search(
            r"<body[^>]*>(.*)</body>", serialized, re.DOTALL | re.IGNORECASE
        )
        if m:
            serialized = m.group(1)

    # 9. Verify post-translation — so sánh tag count đã adjust (không tính wrapper)
    try:
        verify_parser = lxml_html.HTMLParser(recover=True, encoding="utf-8")
        re_tree = lxml_html.fromstring(serialized.encode("utf-8"), parser=verify_parser)
        re_count = _adjusted_parsed_count(re_tree, serialized)
        orig_count = _adjusted_parsed_count(tree, raw_html)
        if orig_count > 0 and abs(re_count - orig_count) > max(2, orig_count * 0.05):
            warnings.append(
                f"post_translate_tag_mismatch: orig={orig_count} after={re_count}"
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"post_translate_verify_failed: {exc!s}"[:200])

    return HtmlTranslationResult(
        html=serialized,
        health=health,
        detected_source_lang=detected,
        segments_translated=len(segments),
        chars_translated=chars_translated,
        warnings=warnings,
    )
