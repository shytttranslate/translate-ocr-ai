"""Cache layer Redis-only (đã bỏ L1 cachetools theo phản biện #19).

Key format kèm model_fingerprint để swap model an toàn không stale cache.
Compression zstd cho payload >= threshold.
"""
from __future__ import annotations

from typing import Any

import blake3
import orjson
import redis.asyncio as aioredis
import zstandard as zstd

from config import Settings
from utils.logging import get_logger

log = get_logger(__name__)

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()
_COMPRESSED_MAGIC = b"\x28\xb5\x2f\xfd"  # zstd magic number


def normalize_text_for_cache(text: str) -> str:
    """Unicode NFKC + collapse whitespace, KHÔNG lowercase (giữ formality semantic)."""
    import unicodedata

    text = unicodedata.normalize("NFKC", text).strip()
    return " ".join(text.split())


def hash_text(text: str) -> str:
    return blake3.blake3(text.encode("utf-8")).hexdigest()[:32]


def make_translate_key(
    *,
    model_fp: str,
    prompt_version: str,
    text: str,
    src_lang: str,
    tgt_lang: str,
    formality: str,
    domain: str,
    glossary_id: str = "none",
) -> str:
    normalized = normalize_text_for_cache(text)
    text_hash = hash_text(normalized)
    return (
        f"tr:m{model_fp}:p{prompt_version}:{text_hash}"
        f":{src_lang}:{tgt_lang}:{formality}:{domain}:{glossary_id}"
    )


def make_ocr_key(
    *,
    model_fp: str,
    prompt_version: str,
    image_phash_hex: str,
    include_translation: bool,
    target_lang: str,
) -> str:
    return (
        f"ocr:m{model_fp}:p{prompt_version}:{image_phash_hex}"
        f":t{int(include_translation)}:{target_lang}"
    )


class RedisCache:
    """Wrapper async Redis có compression tự động + write-back async."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_pool_max_connections,
            decode_responses=False,
        )
        self._client: aioredis.Redis = aioredis.Redis(connection_pool=self._pool)

    async def close(self) -> None:
        await self._client.aclose()
        await self._pool.aclose()

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except (aioredis.RedisError, OSError) as exc:
            log.warning("redis_ping_failed", error=str(exc))
            return False

    async def get_json(self, key: str) -> Any | None:
        try:
            raw = await self._client.get(key)
        except aioredis.RedisError as exc:
            log.warning("redis_get_failed", key=key, error=str(exc))
            return None
        if raw is None:
            return None
        if raw[:4] == _COMPRESSED_MAGIC:
            try:
                raw = _ZSTD_DECOMPRESSOR.decompress(raw)
            except zstd.ZstdError:
                return None
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:
            return None

    async def set_json(self, key: str, value: Any, ttl_s: int) -> None:
        payload = orjson.dumps(value)
        if len(payload) >= self._settings.cache_compression_threshold_bytes:
            payload = _ZSTD_COMPRESSOR.compress(payload)
        try:
            await self._client.set(key, payload, ex=ttl_s)
        except aioredis.RedisError as exc:
            # Cache miss tốt hơn 5xx (theo phản biện #19: write-back, fail-soft)
            log.warning("redis_set_failed", key=key, error=str(exc))

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(key)
        except aioredis.RedisError as exc:
            log.warning("redis_delete_failed", key=key, error=str(exc))
