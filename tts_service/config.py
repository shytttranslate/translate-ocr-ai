"""Cấu hình TTS service — load từ env theo nguyên tắc 12-factor."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
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

    # Chatterbox model
    tts_model_id: str = "ResembleAI/chatterbox"
    tts_device: Literal["cuda", "cpu"] = "cuda"
    tts_dtype: Literal["fp16", "bf16", "fp32"] = "fp16"

    # Voice registry — đường dẫn relative tới CWD lúc start service (tts_service/).
    tts_voices_dir: str = "voices"
    tts_voices_manifest: str = "voices.json"
    tts_default_voice_id: str = "default"

    # Generation defaults
    tts_default_exaggeration: float = 0.5
    tts_default_cfg_weight: float = 0.5
    tts_default_temperature: float = 0.8

    # Limits
    tts_max_text_chars: int = 2000
    tts_chunk_size_chars: int = 300
    tts_chunk_silence_ms: int = 50
    tts_inference_timeout_s: float = 60.0
    tts_concurrency: int = 1

    # Text normalization (số/currency/percent → words) trước khi vào Chatterbox.
    # Chatterbox không có normalizer built-in → "1000" có thể đọc thành "hundred".
    # Tắt nếu anh muốn raw text (vd debug hoặc text đã normalize sẵn).
    tts_normalize_numbers: bool = True

    # Auto-trim silence/low-energy noise đầu+cuối audio sau khi generate.
    # Chatterbox đôi khi sinh "long_tail" (noise đuôi sau câu) → tạo cảm giác rè rè.
    # top_db càng cao càng strict (cắt nhiều): 30=loose, 35=balanced, 40=strict.
    tts_trim_silence: bool = True
    tts_trim_top_db: float = 35.0

    # Warm-up
    tts_warmup_text: str = "Hello, this is a warm-up."
    tts_warmup_language: str = "en"
    tts_warmup_voice_id: str = "default"

    # HF
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hf_home: str | None = Field(default=None, alias="HF_HOME")

    # Misc
    enable_metrics: bool = True

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
