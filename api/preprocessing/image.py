"""Image preprocessing với guard chống decompression bomb + SSRF.

Áp dụng các fix từ #18 Bảo mật:
- MAX_IMAGE_PIXELS=25M (giảm từ default 89M của Pillow)
- Pre-check dimension qua header trước khi decode full
- Reject format không an toàn (SVG, GIF nhiều frame, BMP)
- SSRF guard cho image_url: chỉ https, block private IP, fetch by IP đã validate
"""
from __future__ import annotations

import io
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from PIL import Image, UnidentifiedImageError

from config import Settings
from utils.logging import get_logger

log = get_logger(__name__)

Image.MAX_IMAGE_PIXELS = 25_000_000

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
SSRF_BLOCKED_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


class ImageValidationError(ValueError):
    """Ảnh không hợp lệ: format sai, quá lớn, hoặc nghi ngờ bomb."""


class SsrfBlockedError(ValueError):
    """URL bị chặn do nghi ngờ SSRF (private IP, scheme cấm, ...)."""


@dataclass
class ProcessedImage:
    bytes_jpeg: bytes
    width: int
    height: int
    format_original: str
    rotated_degrees: int = 0


def validate_and_decode(image_bytes: bytes, settings: Settings) -> ProcessedImage:
    """Validate header, decode an toàn, normalize về JPEG.

    Raises ImageValidationError nếu:
    - Bytes vượt max_bytes
    - Format không trong allowlist
    - Dimension vượt max_pixels
    - Decompression bomb
    """
    if len(image_bytes) > settings.image_max_bytes:
        raise ImageValidationError(
            f"Ảnh {len(image_bytes)} bytes vượt giới hạn {settings.image_max_bytes}"
        )

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(f"Không thể parse ảnh: {exc}") from exc

    if img.format not in ALLOWED_FORMATS:
        raise ImageValidationError(
            f"Format {img.format} không được hỗ trợ. Chỉ chấp nhận: {sorted(ALLOWED_FORMATS)}"
        )

    width, height = img.size
    if width * height > settings.image_max_pixels:
        raise ImageValidationError(
            f"Ảnh {width}x{height} = {width * height}px vượt giới hạn "
            f"{settings.image_max_pixels}px (nghi ngờ decompression bomb)"
        )

    n_frames = getattr(img, "n_frames", 1)
    if n_frames > 1:
        raise ImageValidationError(f"Ảnh nhiều frame ({n_frames}) chưa hỗ trợ ở Phase 1")

    original_format = img.format or "UNKNOWN"

    try:
        img.load()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(f"Decompression bomb detected: {exc}") from exc

    rotated = 0
    exif = img.getexif()
    orientation = exif.get(0x0112)
    if orientation == 3:
        img = img.rotate(180, expand=True)
        rotated = 180
    elif orientation == 6:
        img = img.rotate(270, expand=True)
        rotated = 270
    elif orientation == 8:
        img = img.rotate(90, expand=True)
        rotated = 90

    if img.mode != "RGB":
        img = img.convert("RGB")

    max_dim = settings.image_max_dimension
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=95, optimize=True)
    return ProcessedImage(
        bytes_jpeg=out.getvalue(),
        width=img.width,
        height=img.height,
        format_original=original_format,
        rotated_degrees=rotated,
    )


def validate_url_for_ssrf(url: str, settings: Settings) -> tuple[str, str]:
    """Validate URL chống SSRF, trả về (resolved_ip, original_host).

    Caller phải dùng resolved_ip để fetch + truyền Host header riêng để chống DNS rebinding.
    """
    parsed = urlparse(url)
    if parsed.scheme not in settings.ssrf_allow_schemes:
        raise SsrfBlockedError(
            f"Scheme {parsed.scheme!r} không được phép. Chỉ: {settings.ssrf_allow_schemes}"
        )
    if not parsed.hostname:
        raise SsrfBlockedError("URL thiếu hostname")

    if parsed.port and parsed.port not in (443,):
        raise SsrfBlockedError(f"Port {parsed.port} không được phép")

    try:
        addr_info = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise SsrfBlockedError(f"Không resolve được DNS: {exc}") from exc

    if not addr_info:
        raise SsrfBlockedError("DNS không trả IP")

    resolved_ip_str = addr_info[0][4][0]
    if settings.ssrf_block_private_ip:
        resolved_ip = ipaddress.ip_address(resolved_ip_str)
        for blocked in SSRF_BLOCKED_NETWORKS:
            if resolved_ip in blocked:
                raise SsrfBlockedError(
                    f"IP {resolved_ip_str} thuộc dải bị chặn ({blocked})"
                )

    return resolved_ip_str, parsed.hostname


async def fetch_image_from_url(url: str, settings: Settings) -> bytes:
    """Fetch ảnh an toàn từ URL với SSRF guard + size cap."""
    resolved_ip, original_host = validate_url_for_ssrf(url, settings)
    safe_url = url.replace(original_host, resolved_ip, 1)
    headers = {"Host": original_host}

    timeout = httpx.Timeout(connect=3.0, read=10.0, write=3.0, pool=2.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        headers=headers,
        verify=True,
    ) as client:
        async with client.stream("GET", safe_url) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ImageValidationError(
                    f"Content-Type {content_type!r} không phải ảnh"
                )
            buffer = io.BytesIO()
            total = 0
            async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                total += len(chunk)
                if total > settings.image_max_bytes:
                    raise ImageValidationError(
                        f"Ảnh từ URL vượt {settings.image_max_bytes} bytes"
                    )
                buffer.write(chunk)
            return buffer.getvalue()


def perceptual_hash_64(image_bytes: bytes) -> str:
    """pHash 64-bit hex string cho OCR cache key (theo phản biện #19).

    Resize 32x32 grayscale → DCT → top-left 8x8 → so với median → 64 bit.
    Hamming distance ≤ 6 coi như duplicate.
    """
    import numpy as np

    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(
        (32, 32), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(img, dtype=np.float64)

    # DCT-II 2D qua 1D theo từng axis
    dct_rows = _dct_1d(pixels, axis=1)
    dct_full = _dct_1d(dct_rows, axis=0)
    dct_low = dct_full[:8, :8]
    median = float(np.median(dct_low[1:].flatten()))
    bits = (dct_low > median).flatten()

    bit_int = 0
    for b in bits:
        bit_int = (bit_int << 1) | int(b)
    return f"{bit_int:016x}"


def _dct_1d(matrix, axis):  # type: ignore[no-untyped-def]
    import numpy as np
    from scipy.fft import dct  # type: ignore[import-untyped]

    return dct(matrix, type=2, norm="ortho", axis=axis)
