"""Manga pipeline — specialized cho comic/manga JP với vertical text + speech bubbles.

Khác general OCR pipeline ở 3 điểm:
1. Recognition: manga-ocr (kha-white/manga-ocr-base) thay PaddleOCR rec.
   Specialized cho Japanese vertical text trong manga, accuracy cao hơn rõ rệt.
2. Bubble clustering: group text-line theo bubble (proximity + orientation similarity)
   thay vì gộp theo distance đơn thuần.
3. Reading order: phải→trái + trên→dưới (manga JP standard) — sort bubble theo cột RTL.

Detection vẫn dùng PaddleOCR (`PP-OCRv5_server_det`) vì model det handle text mọi script.
"""
from __future__ import annotations

import asyncio
import io
import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from engine import OcrBlock, OcrEngine, OcrWord

log = logging.getLogger("ocr_service.manga")


@dataclass
class MangaBubble:
    """1 speech bubble = group of text-lines."""
    text: str
    bbox: list[list[int]]  # axis-aligned bbox bao quanh bubble
    block_indices: list[int]  # index trong text_blocks
    line_count: int
    avg_confidence: float


class MangaOcrWrapper:
    """Singleton wrapper cho manga-ocr (lazy init)."""

    _instance: "MangaOcrWrapper | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._mocr: Any = None
        self._init_lock = threading.Lock()

    @classmethod
    def get(cls) -> "MangaOcrWrapper":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_init(self) -> None:
        if self._mocr is not None:
            return
        with self._init_lock:
            if self._mocr is not None:
                return
            log.info("manga_ocr_initializing")
            from manga_ocr import MangaOcr  # type: ignore[import-not-found]
            # GPU mode — torch 2.11+cu130 work tốt trên Blackwell SM 12.0 (khác PaddleOCR cu126).
            # Per-line inference 13-24ms GPU vs 50-200ms CPU (5-10x speedup), VRAM ~430MB.
            # Override bằng env PADDLEOCR_MANGAOCR_CPU=1 nếu cần fallback.
            import os
            force_cpu = os.environ.get("PADDLEOCR_MANGAOCR_CPU", "0") == "1"
            self._mocr = MangaOcr(force_cpu=force_cpu)
            log.info("manga_ocr_ready force_cpu=%s", force_cpu)

    def recognize(self, pil_image: Image.Image) -> str:
        """Nhận diện text trong ảnh đã crop. Input: PIL.Image."""
        self._ensure_init()
        return self._mocr(pil_image)


def _bbox_axis_aligned(poly: list[list[int]]) -> tuple[int, int, int, int]:
    """Convert 4-corner polygon → (xmin, ymin, xmax, ymax)."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_center(poly: list[list[int]]) -> tuple[float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _bbox_dim(poly: list[list[int]]) -> tuple[int, int]:
    """Trả (width, height) axis-aligned."""
    x1, y1, x2, y2 = _bbox_axis_aligned(poly)
    return x2 - x1, y2 - y1


def _is_vertical(poly: list[list[int]]) -> bool:
    """Vertical text: height > width × 1.3."""
    w, h = _bbox_dim(poly)
    return h > w * 1.3


def _bbox_distance(p1: list[list[int]], p2: list[list[int]]) -> float:
    """Khoảng cách Euclidean giữa 2 bbox center."""
    c1 = _bbox_center(p1)
    c2 = _bbox_center(p2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def _bbox_overlap_axis(b1: list[list[int]], b2: list[list[int]], axis: str) -> float:
    """Tỷ lệ overlap giữa 2 bbox theo trục `x` hoặc `y`. 0 = không overlap, 1 = trùng hoàn toàn."""
    x1a, y1a, x2a, y2a = _bbox_axis_aligned(b1)
    x1b, y1b, x2b, y2b = _bbox_axis_aligned(b2)
    if axis == "x":
        a1, a2, b1_, b2_ = x1a, x2a, x1b, x2b
    else:
        a1, a2, b1_, b2_ = y1a, y2a, y1b, y2b
    inter = max(0, min(a2, b2_) - max(a1, b1_))
    union = max(a2, b2_) - min(a1, b1_)
    return inter / union if union > 0 else 0.0


def _gap_axis(b1: list[list[int]], b2: list[list[int]], axis: str) -> int:
    """Khoảng gap dương giữa 2 bbox theo trục `x` hoặc `y`. <0 nếu overlap."""
    x1a, y1a, x2a, y2a = _bbox_axis_aligned(b1)
    x1b, y1b, x2b, y2b = _bbox_axis_aligned(b2)
    if axis == "x":
        a1, a2, b1_, b2_ = x1a, x2a, x1b, x2b
    else:
        a1, a2, b1_, b2_ = y1a, y2a, y1b, y2b
    return max(b1_ - a2, a1 - b2_)


def _filter_valid_bubbles(
    raw: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.6,
) -> list[tuple[int, int, int, int]]:
    """Filter contour bbox theo area + aspect + size."""
    img_area = img_w * img_h
    valid: list[tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2) in raw:
        cw, ch = x2 - x1, y2 - y1
        area = cw * ch
        if area < img_area * min_area_ratio:
            continue
        if area > img_area * max_area_ratio:
            continue
        if cw < 20 or ch < 20:
            continue
        ar = cw / ch if ch > 0 else 0
        if ar < 0.1 or ar > 8.0:
            continue
        valid.append((x1, y1, x2, y2))
    return valid


def _dedupe_bubbles(
    bubbles: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.7,
) -> list[tuple[int, int, int, int]]:
    """Dedupe bubbles có IoU > threshold (giữ cái area nhỏ hơn = chính xác hơn)."""
    sorted_b = sorted(bubbles, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    kept: list[tuple[int, int, int, int]] = []
    for b in sorted_b:
        x1, y1, x2, y2 = b
        is_dup = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ua = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - inter
            iou = inter / ua if ua > 0 else 0
            if iou > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(b)
    return kept


def detect_speech_bubbles_cv(
    image_bytes: bytes,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    """Detect speech bubble bằng MULTI-STRATEGY OpenCV.

    Manga có nhiều style bubble: pure white border đen (Yotsuba), gradient,
    hand-drawn outline, screen tone interior. 1 threshold không cover hết.

    **3 strategy chạy song song, merge + dedupe IoU:**

    Strategy 1 — Fixed threshold INTERIOR:
        threshold > 240 = trắng → contours của bubble interior.
        Tốt với manga pure white (Yotsuba style).

    Strategy 2 — Adaptive threshold:
        cv2.adaptiveThreshold (Gaussian, block 51, C=5) handle anti-aliasing border,
        gradient bubble.

    Strategy 3 — BORDER-based (inverse):
        threshold ngược (đen = border) + dilate để close gap → find contours của
        ENCLOSED region. Hữu ích khi bubble border bị broken/dashed.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        log.warning("cv2_not_available — fallback heuristic clustering")
        return []

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape
    all_raw: list[tuple[int, int, int, int]] = []
    counts = {"s1": 0, "s2": 0, "s3": 0}

    # ===== Strategy 1: fixed threshold interior =====
    _, bin1 = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin1 = cv2.morphologyEx(bin1, cv2.MORPH_CLOSE, kernel5, iterations=2)
    cnt1, _ = cv2.findContours(bin1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s1_raw = [tuple(cv2.boundingRect(c)) for c in cnt1]
    s1_raw = [(x, y, x + cw, y + ch) for (x, y, cw, ch) in s1_raw]
    s1_valid = _filter_valid_bubbles(s1_raw, w, h)
    counts["s1"] = len(s1_valid)
    all_raw.extend(s1_valid)

    # ===== Strategy 2: adaptive threshold =====
    bin2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5,
    )
    bin2 = cv2.morphologyEx(bin2, cv2.MORPH_CLOSE, kernel5, iterations=2)
    cnt2, _ = cv2.findContours(bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s2_raw = [tuple(cv2.boundingRect(c)) for c in cnt2]
    s2_raw = [(x, y, x + cw, y + ch) for (x, y, cw, ch) in s2_raw]
    s2_valid = _filter_valid_bubbles(s2_raw, w, h)
    counts["s2"] = len(s2_valid)
    all_raw.extend(s2_valid)

    # ===== Strategy 3: border-based (inverse) =====
    # Bubble = ENCLOSED region by black border. Tìm border đen rồi flood fill region inside.
    _, bin3_inv = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)
    # Dilate để close gaps trong border (border đứt nét, dashed)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin3_inv = cv2.dilate(bin3_inv, kernel3, iterations=2)
    # Invert lại để có WHITE = inside bubble, BLACK = border
    bin3 = cv2.bitwise_not(bin3_inv)
    # Flood fill từ corner để mark background
    flood = bin3.copy()
    cv2.floodFill(flood, None, (0, 0), 0)
    cv2.floodFill(flood, None, (w - 1, 0), 0)
    cv2.floodFill(flood, None, (0, h - 1), 0)
    cv2.floodFill(flood, None, (w - 1, h - 1), 0)
    # Connected components của vùng còn lại = bubble candidates
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flood, connectivity=8)
    s3_raw: list[tuple[int, int, int, int]] = []
    for label_id in range(1, num_labels):  # skip 0 = background
        x, y, cw, ch, _area = stats[label_id]
        s3_raw.append((x, y, x + cw, y + ch))
    s3_valid = _filter_valid_bubbles(s3_raw, w, h)
    counts["s3"] = len(s3_valid)
    all_raw.extend(s3_valid)

    # Dedupe IoU
    bubbles = _dedupe_bubbles(all_raw, iou_threshold=0.6)

    log.info(
        "bubble_detection_cv s1=%d s2=%d s3=%d merged=%d image=%dx%d",
        counts["s1"], counts["s2"], counts["s3"], len(bubbles), w, h,
    )
    return bubbles


def _block_in_bubble(block_bbox: list[list[int]], bubble: tuple[int, int, int, int]) -> bool:
    """Block center có nằm trong bubble bbox không."""
    cx, cy = _bbox_center(block_bbox)
    x1, y1, x2, y2 = bubble
    return x1 <= cx <= x2 and y1 <= cy <= y2


def cluster_into_bubbles(
    blocks: list[OcrBlock],
    image_width: int,
    image_height: int,
    image_bytes: bytes | None = None,
) -> list[list[int]]:
    """Group text-lines thành bubbles — ưu tiên CV bubble detection, fallback heuristic.

    **Step 1**: Detect speech bubble bằng cv2.findContours (manga có border đen + interior trắng).
    Mỗi contour = 1 bubble. Group blocks có center nằm trong bubble bbox.

    **Step 2 (fallback)**: Blocks không nằm trong bubble nào → cluster bằng orientation heuristic.

    Khắc phục case 2 bubble cạnh nhau (cùng độ cao + gần nhau) bị heuristic cũ gộp sai.
    """
    n = len(blocks)
    if n == 0:
        return []

    # Step 1: CV bubble detection
    bubbles_cv: list[tuple[int, int, int, int]] = []
    if image_bytes is not None:
        try:
            bubbles_cv = detect_speech_bubbles_cv(image_bytes, image_width, image_height)
        except Exception as exc:  # noqa: BLE001 — không được fail request vì CV
            log.warning("bubble_cv_failed error=%s", exc)
            bubbles_cv = []

    block_to_bubble: list[int] = [-1] * n
    for i, blk in enumerate(blocks):
        # Match block với bubble nhỏ nhất chứa nó (tránh nested contours)
        best_bubble = -1
        best_area = float("inf")
        for bi, bub in enumerate(bubbles_cv):
            if _block_in_bubble(blk.bbox, bub):
                area = (bub[2] - bub[0]) * (bub[3] - bub[1])
                if area < best_area:
                    best_area = area
                    best_bubble = bi
        block_to_bubble[i] = best_bubble

    cv_clusters: dict[int, list[int]] = {}
    fallback_indices: list[int] = []
    for i, bi in enumerate(block_to_bubble):
        if bi == -1:
            fallback_indices.append(i)
        else:
            cv_clusters.setdefault(bi, []).append(i)

    result: list[list[int]] = list(cv_clusters.values())

    # Step 2: fallback heuristic cho blocks không thuộc bubble nào
    if fallback_indices:
        fb_clusters = _cluster_heuristic(
            [blocks[i] for i in fallback_indices], image_width, image_height,
        )
        for fc in fb_clusters:
            result.append([fallback_indices[k] for k in fc])

    log.info(
        "cluster_done cv_bubbles=%d cv_grouped=%d fallback=%d total_clusters=%d",
        len(bubbles_cv), n - len(fallback_indices), len(fallback_indices), len(result),
    )
    return result


def _cluster_heuristic(
    blocks: list[OcrBlock],
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    """Fallback heuristic clustering — dùng khi CV không detect được bubble.

    **Vertical text (manga JP standard):**
    - 1 bubble = nhiều CỘT vertical kề nhau (đọc phải→trái)
    - 2 vertical blocks thuộc cùng bubble nếu:
      * y_range overlap > 40% (cùng độ cao)
      * x_gap < line_width × 2.5 (cột kề nhau, không quá xa)

    **Horizontal text:**
    - 1 bubble = nhiều DÒNG ngang xếp chồng (đọc trên→dưới)
    - 2 horizontal blocks thuộc cùng bubble nếu:
      * x_range overlap > 40% (cùng cột text)
      * y_gap < line_height × 2.0 (dòng kề nhau)

    **Khác orientation (vertical vs horizontal)** = bubble riêng.
    Trường hợp "先生" chữ nhỏ horizontal lẫn vào các bubble vertical sẽ tự thành block riêng.

    Trả list[cluster] với mỗi cluster là list index của blocks.
    """
    n = len(blocks)
    if n == 0:
        return []

    # Union-find
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        bi = blocks[i].bbox
        wi, hi = _bbox_dim(bi)
        vi = _is_vertical(bi)
        for j in range(i + 1, n):
            bj = blocks[j].bbox
            wj, hj = _bbox_dim(bj)
            vj = _is_vertical(bj)
            # Khác orientation → bubble riêng
            if vi != vj:
                continue

            if vi:
                # Vertical: cột phải kề nhau, cùng độ cao
                # 1) y_range overlap đủ lớn (cùng level cao)
                y_ovl = _bbox_overlap_axis(bi, bj, "y")
                if y_ovl < 0.4:
                    continue
                # 2) x_gap nhỏ (kề cột)
                x_gap = _gap_axis(bi, bj, "x")
                line_w = max(wi, wj)
                if x_gap > line_w * 2.5:
                    continue
            else:
                # Horizontal: dòng kề chồng, cùng cột text
                x_ovl = _bbox_overlap_axis(bi, bj, "x")
                if x_ovl < 0.4:
                    continue
                y_gap = _gap_axis(bi, bj, "y")
                line_h = max(hi, hj)
                if y_gap > line_h * 2.0:
                    continue
            union(i, j)

    # Group theo root
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())


def sort_bubbles_rtl(bubbles: list[MangaBubble]) -> list[MangaBubble]:
    """Sort bubbles theo manga JP reading order: phải→trái + trên→dưới.

    Heuristic: grid-based — chia ảnh thành các cột (right→left), trong mỗi cột sort top→bottom.
    """
    if not bubbles:
        return bubbles
    # Sort theo (column index right-to-left, y_min top-to-bottom)
    # Column = x_center // (image_width / 3) → bucket 3 cột
    # Hoặc simple: sort theo (-x_center, y_min)
    def key(b: MangaBubble) -> tuple[int, int]:
        x1, y1, x2, _ = _bbox_axis_aligned(b.bbox)
        x_center = (x1 + x2) // 2
        return (-x_center, y1)  # x giảm dần (phải→trái), y tăng (trên→dưới)
    return sorted(bubbles, key=key)


async def run_manga_pipeline(
    image_bytes: bytes,
    engine: OcrEngine,
    use_manga_ocr_for_recognition: bool = True,
) -> tuple[list[OcrBlock], list[MangaBubble], int, int]:
    """Manga pipeline đầy đủ.

    1. PaddleOCR detection (model 'japan' để cover JP text trong default rec).
       Đồng thời lấy text PaddleOCR rec làm fallback.
    2. (Optional) Re-recognize từng line bằng manga-ocr cho accuracy JP cao hơn.
    3. Cluster lines thành bubbles.
    4. Sort RTL.

    Returns: (text_blocks, bubbles, image_width, image_height)
    """
    # Step 1: PaddleOCR detection + fallback recognition (lang="japan" cover hiragana/katakana/kanji)
    blocks, width, height = await engine.ocr(image_bytes, "japan", return_word_box=False)
    if not blocks:
        return [], [], width, height

    # Step 2: Re-recognize bằng manga-ocr
    if use_manga_ocr_for_recognition:
        try:
            mocr = MangaOcrWrapper.get()
            # Decode ảnh gốc 1 lần
            pil_full = Image.open(io.BytesIO(image_bytes))
            if pil_full.mode != "RGB":
                pil_full = pil_full.convert("RGB")

            def _re_recognize(blk: OcrBlock) -> str:
                x1, y1, x2, y2 = _bbox_axis_aligned(blk.bbox)
                # Padding nhỏ để không cắt sát chữ
                pad = 4
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(pil_full.width, x2 + pad), min(pil_full.height, y2 + pad)
                if x2 <= x1 or y2 <= y1:
                    return blk.text
                crop = pil_full.crop((x1, y1, x2, y2))
                try:
                    return mocr.recognize(crop)
                except Exception as exc:  # noqa: BLE001
                    log.warning("manga_ocr_recognize_failed bbox=%s error=%s", (x1, y1, x2, y2), exc)
                    return blk.text  # fallback paddle text

            # Run trong thread pool để không block event loop
            new_texts = await asyncio.gather(
                *[asyncio.to_thread(_re_recognize, b) for b in blocks]
            )
            for blk, new_text in zip(blocks, new_texts):
                if new_text and new_text.strip():
                    blk.text = new_text.strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("manga_ocr_pipeline_skipped error=%s", exc)

    # Step 3: Cluster thành bubbles
    clusters = cluster_into_bubbles(blocks, width, height, image_bytes=image_bytes)
    bubbles: list[MangaBubble] = []
    for indices in clusters:
        cluster_blocks = [blocks[i] for i in indices]
        # Sort lines trong bubble: vertical → top-to-bottom theo y; nếu nhiều cột → right-to-left
        is_vertical = _is_vertical(cluster_blocks[0].bbox) if cluster_blocks else False
        if is_vertical:
            # Vertical lines: cột phải đọc trước (manga JP), trong cột top→bottom
            cluster_blocks_sorted = sorted(
                cluster_blocks,
                key=lambda b: (
                    -((_bbox_axis_aligned(b.bbox)[0] + _bbox_axis_aligned(b.bbox)[2]) // 2),
                    _bbox_axis_aligned(b.bbox)[1],
                ),
            )
        else:
            # Horizontal: top→bottom theo y, trong dòng left→right
            cluster_blocks_sorted = sorted(
                cluster_blocks,
                key=lambda b: (_bbox_axis_aligned(b.bbox)[1], _bbox_axis_aligned(b.bbox)[0]),
            )
        merged_text = "".join(b.text for b in cluster_blocks_sorted) if is_vertical else " ".join(b.text for b in cluster_blocks_sorted)
        # Bbox bao quanh cluster
        all_xs = [_bbox_axis_aligned(b.bbox)[0] for b in cluster_blocks] + \
                 [_bbox_axis_aligned(b.bbox)[2] for b in cluster_blocks]
        all_ys = [_bbox_axis_aligned(b.bbox)[1] for b in cluster_blocks] + \
                 [_bbox_axis_aligned(b.bbox)[3] for b in cluster_blocks]
        x1, y1, x2, y2 = min(all_xs), min(all_ys), max(all_xs), max(all_ys)
        avg_conf = sum(b.confidence for b in cluster_blocks) / len(cluster_blocks)
        bubbles.append(MangaBubble(
            text=merged_text,
            bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            block_indices=indices,
            line_count=len(cluster_blocks),
            avg_confidence=avg_conf,
        ))

    # Step 4: Sort RTL
    bubbles = sort_bubbles_rtl(bubbles)

    return blocks, bubbles, width, height
