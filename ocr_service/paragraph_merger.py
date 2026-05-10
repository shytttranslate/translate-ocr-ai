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
    """Quyết định reading order page-level.

    RTL khi:
      - Hebrew/Arabic/Persian: tỷ lệ ký tự RTL >= 30% (đặc trưng rõ, không lẫn).
      - CJK manga vertical: ảnh portrait (H/W >= 1.2) VÀ tỷ lệ char CJK/kana >= 0.5
        (50% — strict để tránh false positive khi ảnh Latin có vài char CJK rác do
        OCR lỗi đọc).
    Còn lại LTR.

    Đếm theo CHARACTER thay vì block — robust hơn khi 1 block có 1 char CJK rác
    nhưng cả block là Latin.
    """
    if not extents:
        return "ltr"
    page_xmin = min(e[0] for e in extents)
    page_ymin = min(e[1] for e in extents)
    page_xmax = max(e[2] for e in extents)
    page_ymax = max(e[3] for e in extents)
    width = max(1, page_xmax - page_xmin)
    height = max(1, page_ymax - page_ymin)
    portrait = (height / width) >= 1.2

    cjk_chars = 0
    rtl_chars = 0
    other_alpha = 0
    for t in texts:
        for c in t:
            cp = ord(c)
            if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) \
                    or (0x3040 <= cp <= 0x30FF) or (0xAC00 <= cp <= 0xD7AF):
                cjk_chars += 1
            elif (0x0590 <= cp <= 0x05FF) or (0x0600 <= cp <= 0x06FF) \
                    or (0x0750 <= cp <= 0x077F):
                rtl_chars += 1
            elif c.isalpha():
                other_alpha += 1
    total = cjk_chars + rtl_chars + other_alpha
    if total == 0:
        return "ltr"
    if rtl_chars / total >= 0.30:
        return "rtl"
    if portrait and cjk_chars / total >= 0.50:
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
    y_overlap_threshold: float = 0.3,
    line_gap_ratio: float = 0.8,
    inline_gap_ratio: float = 1.5,
    image_gray=None,
) -> list[OcrParagraph]:
    """Gộp các OCR block thành paragraph qua 2-pass clustering.

    Args:
        blocks: list block gốc từ PaddleOCR.
        reading_order: "ltr" | "rtl" | "auto".
        x_overlap_threshold: vertical merge — 2 block cùng cột phải chồng x ≥ 30%.
        y_overlap_threshold: horizontal merge — 2 block cùng dòng phải chồng y ≥ 30%.
        line_gap_ratio: y-gap tối đa giữa 2 dòng cùng paragraph (× line_height).
        inline_gap_ratio: x-gap tối đa giữa 2 đoạn cùng dòng (× line_height) —
            handle trường hợp PaddleOCR tách "SO   GIVE YOUR   TIME" thành 3 box.
        image_gray: optional grayscale ndarray cho boldness detection. Nếu None,
            split chỉ dựa height jump (kém chính xác hơn).

    Pass 1: gom cluster qua union-find. 2 cách merge:
        - Vertical (cùng cột): x_overlap ≥ threshold + y_gap ≤ line_gap_ratio.
        - Horizontal (cùng dòng): y_overlap ≥ threshold + x_gap ≤ inline_gap_ratio.
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

    # Pass 1: cluster bằng union-find. 2 hướng merge (vertical OR horizontal).
    uf = _UnionFind(n)
    for i in range(n):
        xi_min, yi_min, xi_max, yi_max = extents[i]
        for j in range(i + 1, n):
            xj_min, yj_min, xj_max, yj_max = extents[j]

            # Vertical merge: 2 block cùng cột (x_overlap đủ), gap y nhỏ → cùng paragraph.
            x_overlap = _overlap_ratio(xi_min, xi_max, xj_min, xj_max)
            if x_overlap >= x_overlap_threshold:
                y_gap = max(yi_min, yj_min) - min(yi_max, yj_max)
                if y_gap <= line_height * line_gap_ratio:
                    uf.union(i, j)
                    continue

            # Horizontal merge: 2 block cùng dòng (y_overlap đủ), gap x nhỏ → cùng line.
            # Handle case PaddleOCR tách "SO   GIVE YOUR   TIME" thành 3 box do gap word
            # lớn (handwritten / spaced text) — các box không x-overlap nhau nhưng cùng
            # 1 dòng đọc.
            y_overlap = _overlap_ratio(yi_min, yi_max, yj_min, yj_max)
            if y_overlap >= y_overlap_threshold:
                x_gap = max(xi_min, xj_min) - min(xi_max, xj_max)
                if x_gap <= line_height * inline_gap_ratio:
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

    # Bin y_center theo band ~0.7 × line_height — block cùng 1 dòng (y_center
    # chênh < 0.7 lh do PaddleOCR pad bbox khác nhau) gom cùng band → sort thứ
    # cấp theo x. Tránh case sort primary theo raw y → đảo ngược thứ tự dòng.
    intra_band = max(1.0, line_height * 0.7)

    for indices in clusters.values():
        # Sort theo (band(y_center), x_center * direction). Trong cùng band x đi
        # theo direction (LTR: trái→phải; RTL: phải→trái).
        indices.sort(
            key=lambda idx: (
                int(((extents[idx][1] + extents[idx][3]) / 2) / intra_band),
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
