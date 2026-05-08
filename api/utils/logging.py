"""Structured JSON logging với redaction.

Tuân thủ feedback của #18 Bảo mật:
- Không log full text content trên DEBUG
- Không log image bytes ở mọi level
- API key chỉ log prefix 8 ký tự
- Redact image_url thành scheme+host (bỏ path/query)
"""
from __future__ import annotations

import logging
import sys
from typing import Any
from urllib.parse import urlparse

import structlog
from structlog.types import EventDict


def redact_sensitive(_logger: Any, _name: str, event: EventDict) -> EventDict:
    """Drop hoặc rút gọn các field nhạy cảm trước khi ghi log."""
    if "image_bytes" in event:
        event.pop("image_bytes")
    if "image_b64" in event:
        event.pop("image_b64")

    if (url := event.get("image_url")) and isinstance(url, str):
        try:
            parsed = urlparse(url)
            event["image_url"] = f"{parsed.scheme}://{parsed.netloc}/<redacted>"
        except (ValueError, AttributeError):
            event["image_url"] = "<invalid>"

    if (key := event.get("api_key")) and isinstance(key, str):
        event["api_key"] = key[:8] + "***" if len(key) > 8 else "***"

    if (text := event.get("text")) and isinstance(text, str):
        event["text_len"] = len(text)
        event["text"] = text[:64] + "..." if len(text) > 64 else text

    return event


def configure_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_sensitive,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
