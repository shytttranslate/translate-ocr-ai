"""Cấu hình API service — load từ env theo nguyên tắc 12-factor."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

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

    vllm_translator_url: str = "http://127.0.0.1:9001"
    vllm_translator_model: str = "translator"
    vllm_request_timeout_s: float = 60.0
    vllm_connect_timeout_s: float = 3.0

    enable_metrics: bool = True

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
