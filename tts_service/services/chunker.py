"""Tách text dài thành chunk vừa với context của Chatterbox.

Regex tách câu cover Latin + CJK + Devanagari (.!? 。！？ ।).
Câu liền kề được gộp lại tới khi gần `max_chunk_chars`.
Câu đơn dài hơn `max_chunk_chars` → hard-split theo space/comma.
"""
from __future__ import annotations

import re

# Tách sau dấu câu kết thúc + whitespace (Latin + CJK + Devanagari).
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？।])\s+")

# Hard-split fallback: ưu tiên cắt ở khoảng trắng, dấu phẩy, dấu chấm phẩy.
_HARD_SPLIT_RE = re.compile(r"[,;，；、]\s*|\s+")


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Cắt câu dài >max_chars thành nhiều phần, ưu tiên ranh giới từ."""
    if len(sentence) <= max_chars:
        return [sentence]

    parts: list[str] = []
    cur = ""
    tokens = _HARD_SPLIT_RE.split(sentence)
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if len(cur) + len(token) + 1 <= max_chars:
            cur = f"{cur} {token}".strip()
        else:
            if cur:
                parts.append(cur)
            # Token đơn lẻ vẫn quá dài (vd URL khổng lồ) → cắt cứng theo char.
            if len(token) > max_chars:
                for i in range(0, len(token), max_chars):
                    parts.append(token[i:i + max_chars])
                cur = ""
            else:
                cur = token
    if cur:
        parts.append(cur)
    return parts


def split_text_for_tts(text: str, max_chunk_chars: int) -> list[str]:
    """Trả list chunk có len <= max_chunk_chars (best effort).

    Nếu text gốc đã ngắn hơn max_chunk_chars → trả list 1 phần tử.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chunk_chars:
        return [text]

    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    cur = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Câu đơn quá dài → flush cur, hard-split sentence riêng.
        if len(sentence) > max_chunk_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_hard_split(sentence, max_chunk_chars))
            continue
        # Gộp câu vào cur nếu vẫn fit.
        candidate = f"{cur} {sentence}".strip() if cur else sentence
        if len(candidate) <= max_chunk_chars:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            cur = sentence
    if cur:
        chunks.append(cur)
    return chunks
