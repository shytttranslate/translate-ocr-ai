"""API key auth với HMAC-SHA256 + pepper, constant-time compare.

Theo phản biện #18 Bảo mật:
- KHÔNG dùng bcrypt (chậm 100ms/req không scale với 150+ req/s)
- HMAC-SHA256(pepper, full_key) đủ vì key entropy đã 192 bit
- Lookup theo prefix 8 ký tự, verify bằng compare_digest constant-time
- Cache positive validation TTL 60s qua Redis (chưa implement persist DB ở Phase 1)
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Literal

from fastapi import Header, HTTPException, status

from config import Settings, get_settings

Tier = Literal["free", "pro", "enterprise"]
KEY_PREFIX_PUBLIC = "vbk_live_"
KEY_PREFIX_TEST = "vbk_test_"


@dataclass(frozen=True)
class ApiKeyRecord:
    key_id: str
    prefix_hint: str
    tier: Tier
    scopes: tuple[str, ...]


def generate_api_key() -> tuple[str, str]:
    """Sinh raw key + prefix_hint. Raw key chỉ trả 1 lần lúc tạo."""
    raw = secrets.token_urlsafe(24)
    full = f"{KEY_PREFIX_PUBLIC}{raw}"
    prefix_hint = full[: len(KEY_PREFIX_PUBLIC) + 8]
    return full, prefix_hint


def hash_api_key(full_key: str, pepper: str) -> str:
    return hmac.new(
        pepper.encode("utf-8"),
        full_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_api_key(provided: str, expected_hash: str, pepper: str) -> bool:
    actual_hash = hash_api_key(provided, pepper)
    return hmac.compare_digest(actual_hash, expected_hash)


# Phase 1 dùng in-memory store; Phase 5 sẽ migrate sang Postgres.
_DEV_KEYS: dict[str, ApiKeyRecord] = {}


def seed_dev_key(settings: Settings, full_key: str, tier: Tier = "free") -> None:
    """Seed 1 key dev cho testing local. KHÔNG dùng ở prod."""
    if settings.is_prod:
        return
    prefix_hint = full_key[: len(KEY_PREFIX_PUBLIC) + 8]
    expected_hash = hash_api_key(full_key, settings.api_key_pepper)
    _DEV_KEYS[expected_hash] = ApiKeyRecord(
        key_id=f"dev-{prefix_hint}",
        prefix_hint=prefix_hint,
        tier=tier,
        scopes=("translate", "detect", "ocr", "ocr-translate"),
    )


async def require_api_key(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ApiKeyRecord:
    """Dependency FastAPI cho endpoint cần auth."""
    settings = get_settings()
    raw_key = _extract_key(authorization, x_api_key)

    if not raw_key or not (
        raw_key.startswith(KEY_PREFIX_PUBLIC) or raw_key.startswith(KEY_PREFIX_TEST)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Thiếu hoặc sai định dạng API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expected_hash = hash_api_key(raw_key, settings.api_key_pepper)
    record = _DEV_KEYS.get(expected_hash)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key không hợp lệ",
        )
    return record


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None
