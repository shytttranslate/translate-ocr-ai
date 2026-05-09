"""Heuristic gộp các OCR text-line liền kề thành paragraph (1 speech bubble / khối văn).

Vấn đề: PaddleOCR detect text ở mức line. Một speech bubble manga hoặc 1 đoạn văn
nhiều dòng sẽ ra nhiều block riêng lẻ:

    "There are so"
    "many kinds of"
    "gradation tones."

Module này gộp lại thành 1 paragraph: "There are so many kinds of gradation tones."

Thuật toán **2 pass**:

  Pass 1 — cluster theo geometry (gap + x-overlap), height_diff loose:
    Gom các line gần nhau theo y, cùng cột x.

  Pass 2 — split cluster theo style break (height jump OR boldness jump):
    Trong cùng card UI, header (bold, h cao) + body (regular, h thấp) thường có
    geometry rất gần nhau (gap 7-15px) → Pass 1 gộp chúng. Pass 2 phát hiện
    style transition (font weight + size khác) để tách thành paragraph riêng.

Tham số trọng yếu:
  - line_gap_ratio: y-gap tối đa giữa 2 dòng cùng paragraph (× line_height).
  - style_height_jump: ratio chênh height giữa 2 line liền kề báo hiệu style break.
  - style_boldness_jump: ratio chênh boldness báo hiệu style break.

Module pure function (image_gray optional). Unit test friendly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Literal, Optional

from engine import OcrLine

try:
    import numpy as np
    _NumpyArr = "np.ndarray"
except ImportError:  # numpy luôn có khi paddleocr install, nhưng safety import
    np = None  # type: ignore[assignment]
    _NumpyArr = "object"


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


def is_meaningless_text(text: str) -> bool:
    """Detect text vô nghĩa từ OCR misread (icon/decoration/artifact).

    Conservative filter — chỉ bắt 2 pattern:
    1. Chứa backslash `\\`: PaddleOCR đọc icon stylized (vd tên tab `文A`)
       thành LaTeX-like `$\\a$` hoặc `\\A`. Backslash KHÔNG xuất hiện trong text
       Latin/Cyrillic/CJK thực tế → 100% là garbage.
    2. Toàn ký hiệu/dấu câu, length ≥ 2: không có alphanumeric AND không có CJK.
       Length 1 (vd `?`, `!`) giữ lại vì có thể là bubble exclamation legitimate.
    """
    s = text.strip()
    if not s:
        return True
    if "\\" in s:
        return True
    if any(c.isalnum() for c in s) or _has_cjk(s):
        return False
    return len(s) >= 2


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


def _bbox_boldness(image_gray, bbox: list[list[int]], dark_threshold: int = 100) -> float:
    """Tỷ lệ pixel tối trong bbox — proxy cho font weight (bold vs regular).

    Bold text: stroke dày → nhiều pixel tối → ratio cao.
    Regular text: stroke mảnh → ratio thấp.
    Same paragraph thường có boldness ratio chênh < 30%.
    """
    if image_gray is None:
        return 0.0
    x1, y1, x2, y2 = _bbox_extents(bbox)
    H, W = image_gray.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = image_gray[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float((crop < dark_threshold).mean())


def _split_cluster_by_style(
    indices: list[int],
    blocks: list[OcrLine],
    image_gray,
    height_jump: float = 0.40,
    boldness_jump: float = 0.40,
) -> list[list[int]]:
    """Pass 2: split cluster theo style break — OR logic với threshold cao.

    Sort lines by y_center, scan adjacent pairs. Split nếu:
    - Height ratio > 0.40 (font size khác RÕ RỆT, vd 32px vs 48px = ratio 0.33 KHÔNG split) OR
    - Boldness ratio > 0.40 (font weight khác RÕ RỆT, vd bold→regular)

    Threshold cao + OR logic → chỉ trigger khi signal MẠNH:
    - Trong cùng paragraph: h variance ~0.10-0.35 (do descender/ascender + short
      lines như "dim" có h cao hơn full line do PaddleOCR padding), b stable
    - Header → body cùng size khác weight: b > 0.40 trigger
    - Header → body khác size: h > 0.40 trigger

    Khi image_gray=None: chỉ dùng height jump.
    """
    if len(indices) <= 1:
        return [list(indices)]

    extents = [_bbox_extents(blocks[i].bbox) for i in indices]
    heights = [max(1, e[3] - e[1]) for e in extents]
    boldnesses = [_bbox_boldness(image_gray, blocks[i].bbox) for i in indices]
    y_centers = [(e[1] + e[3]) / 2 for e in extents]

    order = sorted(range(len(indices)), key=lambda k: y_centers[k])

    sub_clusters: list[list[int]] = []
    current: list[int] = [indices[order[0]]]
    for pos in range(1, len(order)):
        prev_k = order[pos - 1]
        cur_k = order[pos]
        h_diff = abs(heights[cur_k] - heights[prev_k]) / max(heights[cur_k], heights[prev_k])
        split = h_diff > height_jump
        if not split and image_gray is not None:
            b_max = max(boldnesses[cur_k], boldnesses[prev_k], 0.01)
            b_diff = abs(boldnesses[cur_k] - boldnesses[prev_k]) / b_max
            split = b_diff > boldness_jump
        if split:
            sub_clusters.append(current)
            current = [indices[cur_k]]
        else:
            current.append(indices[cur_k])
    sub_clusters.append(current)
    return sub_clusters


def merge_blocks_into_paragraphs(
    blocks: list[OcrLine],
    reading_order: ReadingOrder = "auto",
    x_overlap_threshold: float = 0.3,
    line_gap_ratio: float = 0.8,
    image_gray=None,
) -> list[OcrParagraph]:
    """Gộp các OCR block thành paragraph qua 2-pass clustering.

    Args:
        blocks: list block gốc từ PaddleOCR.
        reading_order: "ltr" | "rtl" | "auto".
        x_overlap_threshold: 2 block phải chồng theo trục x ít nhất 30%.
        line_gap_ratio: y-gap tối đa giữa 2 dòng cùng paragraph, × line_height.
        image_gray: optional grayscale ndarray cho boldness detection. Nếu None,
            split chỉ dựa height jump (kém chính xác hơn).

    Pass 1: gom cluster theo gap + x_overlap (loose, no height check).
    Pass 2: trong mỗi cluster, split theo style break (height + boldness jump).

    Returns: List paragraph sort theo reading order.
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

    # Pass 1: cluster bằng union-find theo gap + x_overlap (no height filter).
    uf = _UnionFind(n)
    for i in range(n):
        xi_min, yi_min, xi_max, yi_max = extents[i]
        for j in range(i + 1, n):
            xj_min, yj_min, xj_max, yj_max = extents[j]

            x_overlap = _overlap_ratio(xi_min, xi_max, xj_min, xj_max)
            if x_overlap < x_overlap_threshold:
                continue

            top_block_bottom = min(yi_max, yj_max)
            bottom_block_top = max(yi_min, yj_min)
            y_gap = bottom_block_top - top_block_bottom
            if y_gap > line_height * line_gap_ratio:
                continue

            uf.union(i, j)

    # Pass 2: split clusters theo style break (height + boldness jump)
    raw_clusters: dict[int, list[int]] = {}
    for i in range(n):
        raw_clusters.setdefault(uf.find(i), []).append(i)

    clusters: dict[int, list[int]] = {}
    next_id = 0
    for cluster_indices in raw_clusters.values():
        sub_clusters = _split_cluster_by_style(cluster_indices, blocks, image_gray)
        for sub in sub_clusters:
            clusters[next_id] = sub
            next_id += 1

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
