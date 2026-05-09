"""Heuristic gộp các OCR text-line liền kề thành paragraph (1 speech bubble / khối văn).

Vấn đề: PaddleOCR detect text ở mức line. Một speech bubble manga hoặc 1 đoạn văn
nhiều dòng sẽ ra nhiều block riêng lẻ:

    "There are so"
    "many kinds of"
    "gradation tones."

Module này gộp lại thành 1 paragraph: "There are so many kinds of gradation tones."

Thuật toán:
  1. Đo `line_height` trung vị từ chiều cao bbox.
  2. Cặp 2 block thuộc cùng paragraph nếu:
       - x-overlap ratio (theo box hẹp hơn) >= `x_overlap_threshold`
       - y-gap dọc (top của block dưới - bottom của block trên) <= `line_height * line_gap_ratio`
       - chênh chiều cao tương đối <= `height_diff_ratio` (tránh gộp tiêu đề + body)
  3. Union-Find gom cluster.
  4. Trong cluster, sort theo reading order rồi nối text:
       - Nếu cluster chứa ký tự CJK (Trung/Nhật/Hàn) → nối "" (không space).
       - Còn lại → nối " " (giả định wrap, không phải câu mới).
  5. Sort các paragraph theo reading order toàn ảnh:
       - "ltr": row-first ascending x
       - "rtl": row-first descending x (manga JP)
       - "auto": detect dựa vào aspect ratio + tỉ lệ ký tự CJK

Module thuần pure function — không phụ thuộc PaddleOCR runtime, dễ unit test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Literal

from engine import OcrLine


ReadingOrder = Literal["ltr", "rtl", "auto"]


@dataclass
class OcrParagraph:
    text: str
    bbox: list[list[int]]
    block_indices: list[int] = field(default_factory=list)
    avg_confidence: float = 0.0
    line_count: int = 0


def _bbox_extents(bbox: list[list[int]]) -> tuple[int, int, int, int]:
    """Trả về (xmin, ymin, xmax, ymax) — axis-aligned bound."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_ratio(a_min: int, a_max: int, b_min: int, b_max: int) -> float:
    """Overlap chia cho size của box hẹp hơn."""
    span_a = max(1, a_max - a_min)
    span_b = max(1, b_max - b_min)
    overlap = max(0, min(a_max, b_max) - max(a_min, b_min))
    return overlap / min(span_a, span_b)


def _has_cjk(text: str) -> bool:
    """True nếu text chứa ký tự CJK / Kana / Hangul (không cần space khi nối)."""
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:  # Hiragana, Katakana
            return True
        if 0x4E00 <= cp <= 0x9FFF:  # CJK Unified Ideographs
            return True
        if 0x3400 <= cp <= 0x4DBF:  # CJK Extension A
            return True
        if 0xAC00 <= cp <= 0xD7AF:  # Hangul Syllables
            return True
        if 0xF900 <= cp <= 0xFAFF:  # CJK Compatibility Ideographs
            return True
    return False


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def detect_reading_order(blocks: list[OcrLine]) -> ReadingOrder:
    """Public helper — wrap `_detect_reading_order_from_extents` để client không phải tự build extents."""
    if not blocks:
        return "ltr"
    extents = [_bbox_extents(b.bbox) for b in blocks]
    texts = [b.text for b in blocks]
    return _detect_reading_order_from_extents(extents, texts)


def _detect_reading_order_from_extents(
    extents: list[tuple[int, int, int, int]], texts: list[str],
) -> ReadingOrder:
    """RTL nếu ảnh portrait (H/W >= 1.2) VÀ >= 30% block có CJK. Còn lại LTR."""
    if not extents:
        return "ltr"
    page_xmin = min(e[0] for e in extents)
    page_ymin = min(e[1] for e in extents)
    page_xmax = max(e[2] for e in extents)
    page_ymax = max(e[3] for e in extents)
    width = max(1, page_xmax - page_xmin)
    height = max(1, page_ymax - page_ymin)
    portrait = (height / width) >= 1.2
    cjk_ratio = sum(1 for t in texts if _has_cjk(t)) / max(1, len(texts))
    if portrait and cjk_ratio >= 0.3:
        return "rtl"
    return "ltr"


def merge_blocks_into_paragraphs(
    blocks: list[OcrLine],
    reading_order: ReadingOrder = "auto",
    x_overlap_threshold: float = 0.3,
    line_gap_ratio: float = 1.5,
    height_diff_ratio: float = 0.7,
) -> list[OcrParagraph]:
    """Gộp các OCR block liền kề thành paragraph.

    Args:
        blocks: list block gốc từ PaddleOCR.
        reading_order: "ltr" | "rtl" | "auto" (default auto detect).
        x_overlap_threshold: 2 block phải chồng theo trục x ít nhất 30% (theo box hẹp hơn).
        line_gap_ratio: y-gap tối đa giữa 2 dòng cùng paragraph, tính bằng `line_height`.
        height_diff_ratio: chênh chiều cao tối đa cho phép — tránh gộp tiêu đề khác cỡ chữ.

    Returns:
        List paragraph đã sort theo reading order. Block không match nào → 1 paragraph riêng.
    """
    n = len(blocks)
    if n == 0:
        return []
    if n == 1:
        b = blocks[0]
        return [
            OcrParagraph(
                text=b.text,
                bbox=b.bbox,
                block_indices=[0],
                avg_confidence=b.confidence,
                line_count=1,
            )
        ]

    extents = [_bbox_extents(b.bbox) for b in blocks]
    heights = [(e[3] - e[1]) for e in extents]
    line_height = max(1.0, median(heights))

    if reading_order == "auto":
        reading_order = _detect_reading_order_from_extents(
            extents, [b.text for b in blocks],
        )

    # Build cluster bằng union-find. O(n^2) — n thường <= 200 nên không lo.
    uf = _UnionFind(n)
    for i in range(n):
        xi_min, yi_min, xi_max, yi_max = extents[i]
        hi = max(1, yi_max - yi_min)
        for j in range(i + 1, n):
            xj_min, yj_min, xj_max, yj_max = extents[j]
            hj = max(1, yj_max - yj_min)

            # Chênh chiều cao quá lớn → khả năng khác cỡ chữ / khác vai trò
            ratio = abs(hi - hj) / max(hi, hj)
            if ratio > height_diff_ratio:
                continue

            # Phải chồng đáng kể theo trục x
            x_overlap = _overlap_ratio(xi_min, xi_max, xj_min, xj_max)
            if x_overlap < x_overlap_threshold:
                continue

            # Y-gap (block trên trước, block dưới sau)
            top_block_bottom = min(yi_max, yj_max)
            bottom_block_top = max(yi_min, yj_min)
            y_gap = bottom_block_top - top_block_bottom
            # y_gap < 0 nghĩa là 2 box overlap dọc → vẫn coi là cùng paragraph
            if y_gap > line_height * line_gap_ratio:
                continue

            uf.union(i, j)

    # Gom indices theo cluster root
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    paragraphs: list[OcrParagraph] = []
    direction = -1 if reading_order == "rtl" else 1

    for indices in clusters.values():
        # Sort indices trong cluster theo (y_center, x_center * direction).
        # Trong 1 paragraph reading order vẫn là top→bottom. RTL chỉ ảnh hưởng
        # khi 2 line cùng y (vertical text manga column → đọc phải sang trái).
        indices.sort(
            key=lambda idx: (
                (extents[idx][1] + extents[idx][3]) / 2,
                ((extents[idx][0] + extents[idx][2]) / 2) * direction,
            )
        )

        merged_text_parts: list[str] = []
        any_cjk = False
        for idx in indices:
            t = blocks[idx].text
            if _has_cjk(t):
                any_cjk = True
            merged_text_parts.append(t)
        joiner = "" if any_cjk else " "
        merged_text = joiner.join(merged_text_parts)

        # Bbox bao quanh cluster — axis-aligned
        cluster_xmin = min(extents[i][0] for i in indices)
        cluster_ymin = min(extents[i][1] for i in indices)
        cluster_xmax = max(extents[i][2] for i in indices)
        cluster_ymax = max(extents[i][3] for i in indices)
        merged_bbox = [
            [cluster_xmin, cluster_ymin],
            [cluster_xmax, cluster_ymin],
            [cluster_xmax, cluster_ymax],
            [cluster_xmin, cluster_ymax],
        ]

        avg_conf = sum(blocks[i].confidence for i in indices) / len(indices)

        paragraphs.append(
            OcrParagraph(
                text=merged_text,
                bbox=merged_bbox,
                block_indices=indices,
                avg_confidence=avg_conf,
                line_count=len(indices),
            )
        )

    # Sort paragraph toàn ảnh theo row-first reading order.
    # Hai paragraph cùng "row" nếu y_center chênh < line_height * 2.
    band = max(1.0, line_height * 2.0)

    def _sort_key(p: OcrParagraph) -> tuple[int, float]:
        x1, y1, x2, y2 = _bbox_extents(p.bbox)
        y_center = (y1 + y2) / 2
        x_center = (x1 + x2) / 2
        return (int(y_center / band), x_center * direction)

    paragraphs.sort(key=_sort_key)
    return paragraphs


def paragraphs_to_full_text(paragraphs: Iterable[OcrParagraph]) -> str:
    """Nối paragraph bằng \\n\\n — quy ước "đoạn văn" trong plain text."""
    return "\n\n".join(p.text for p in paragraphs)
