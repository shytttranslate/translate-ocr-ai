"""Pydantic schemas cho TTS service.

Chatterbox Multilingual 0.5B hỗ trợ 23 ngôn ngữ — KHÔNG có Vietnamese.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# 23 mã ISO Chatterbox Multilingual hỗ trợ. KHÔNG có "vi".
SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
)

LanguageId = Literal[
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
]

LANGUAGE_LABELS: dict[str, str] = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese (Mandarin)",
}


class TTSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000, description="Text cần synthesize")
    language_id: LanguageId = Field(default="en", description="Mã ngôn ngữ ISO (23 mã, KHÔNG có 'vi')")
    voice_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="ID voice preset trên server. Gọi GET /v1/voices để liệt kê.",
    )
    exaggeration: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Mức cường điệu cảm xúc (0=monotone, 1=dramatic)",
    )
    cfg_weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Classifier-Free Guidance weight (giảm cho speaker nói nhanh)",
    )
    temperature: float = Field(
        default=0.8, ge=0.0, le=2.0,
        description="Sampling temperature (0=deterministic, 2=high diversity)",
    )
    seed: int | None = Field(
        default=None, ge=0, le=2**32 - 1,
        description="Random seed — set để reproducible. Không set thì non-deterministic.",
    )
    request_id: str | None = Field(default=None, description="Trace ID từ caller")

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text không được rỗng/whitespace-only")
        return v


class TTSResponse(BaseModel):
    request_id: str
    service: Literal["tts"] = "tts"
    processing_time_ms: int
    audio_base64: str = Field(description="WAV PCM 16-bit, 24kHz mono, base64-encoded")
    sample_rate: Literal[24000] = 24000
    duration_ms: int
    format: Literal["wav"] = "wav"
    voice_id: str
    language_id: str
    chunk_count: int = Field(
        default=1, ge=1,
        description="Số chunk model generate (>=2 nếu text > tts_chunk_size_chars)",
    )
    seed: int | None = Field(default=None, description="Echo seed nếu user set")


class VoiceInfo(BaseModel):
    id: str
    gender: Literal["male", "female", "neutral"] | None = None
    language_hint: str | None = Field(
        default=None,
        description="Ngôn ngữ gốc của voice sample — hint thôi, voice clone xuyên ngôn ngữ OK.",
    )
    description: str | None = None
    has_audio_prompt: bool = Field(
        description="False = dùng default voice của model (không audio_prompt_path)",
    )


class VoicesResponse(BaseModel):
    service: Literal["tts"] = "tts"
    count: int
    default_voice_id: str
    voices: list[VoiceInfo]


class LanguageEntry(BaseModel):
    code: str
    label: str


class LanguagesResponse(BaseModel):
    service: Literal["tts"] = "tts"
    engine: Literal["ChatterboxMultilingualTTS"] = "ChatterboxMultilingualTTS"
    count: int
    languages: list[LanguageEntry]
    note: str = Field(
        default="Vietnamese (vi) chưa được Chatterbox Multilingual 0.5B hỗ trợ.",
    )
