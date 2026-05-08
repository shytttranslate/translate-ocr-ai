"""Cấu hình chung cho API service — load từ env var theo nguyên tắc 12-factor."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    vllm_translator_url: str = "http://vllm-translator:8001"
    vllm_ocr_url: str = "http://vllm-ocr:8002"
    vllm_translator_model: str = "translator"
    vllm_ocr_model: str = "ocr"
    vllm_request_timeout_s: float = 60.0
    vllm_connect_timeout_s: float = 3.0

    redis_url: str = "redis://redis:6379/0"
    redis_ratelimit_url: str = "redis://redis:6379/1"
    redis_pool_max_connections: int = 50

    cache_translation_ttl_s: int = 60 * 60 * 24 * 14  # 14 ngày
    cache_ocr_ttl_s: int = 60 * 60 * 24 * 7  # 7 ngày
    cache_lang_detect_ttl_s: int = 60 * 60 * 24 * 7
    cache_compression_threshold_bytes: int = 2048

    image_max_bytes: int = 10 * 1024 * 1024
    image_max_pixels: int = 25_000_000
    image_max_dimension: int = 2048

    api_key_pepper: str = Field(default="dev-pepper-replace-in-prod", min_length=16)
    api_key_cache_ttl_s: int = 60

    ssrf_allow_schemes: tuple[str, ...] = ("https",)
    ssrf_block_private_ip: bool = True

    rate_limit_free_per_s: int = 10
    rate_limit_pro_per_s: int = 50
    rate_limit_enterprise_per_s: int = 200

    enable_metrics: bool = True
    enable_otel: bool = False
    otel_endpoint: str | None = None

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
