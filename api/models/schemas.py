"""Pydantic v2 schemas cho toàn bộ endpoint.

Áp các fix từ phản biện:
- #15: response unified luôn dùng list (translations: list, kể cả single)
- #18: cap max_length text 50k, batch 100, glossary 500
- ViSa: model_fingerprint trong cache key, request_id để tracing
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Formality = Literal["formal", "informal", "neutral"]
Domain = Literal["general", "technical", "legal", "medical", "casual", "marketing"]
SceneHint = Literal[
    "auto", "sign", "menu", "billboard", "storefront", "document", "product"
]


class GlossaryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)


class TranslateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | list[str] = Field(description="Text đơn hoặc list. Tối đa 100 phần tử.")
    source_lang: str = Field(default="auto", pattern=r"^(auto|[a-z]{2,3}(-[A-Z]{2})?)$")
    target_lang: str = Field(min_length=2, max_length=8)
    formality: Formality = "neutral"
    domain: Domain = "general"
    glossary: list[GlossaryEntry] = Field(default_factory=list, max_length=500)
    context: str | None = Field(default=None, max_length=2000)
    preserve_formatting: bool = True
    return_alternatives: bool = False

    @model_validator(mode="after")
    def _check_text_size(self) -> TranslateRequest:
        if isinstance(self.text, str):
            if len(self.text) > 50_000:
                raise ValueError("text vượt 50000 ký tự")
            if not self.text.strip():
                raise ValueError("text không được rỗng")
        else:
            if len(self.text) == 0:
                raise ValueError("text list rỗng")
            if len(self.text) > 100:
                raise ValueError("batch vượt 100 phần tử")
            for i, t in enumerate(self.text):
                if len(t) > 50_000:
                    raise ValueError(f"text[{i}] vượt 50000 ký tự")
        return self


class TranslationItem(BaseModel):
    source_text: str
    translated_text: str
    source_language: str
    source_language_confidence: float = Field(ge=0.0, le=1.0)
    target_language: str
    alternatives: list[str] = Field(default_factory=list)
    formality_applied: Formality = "neutral"
    warnings: list[str] = Field(default_factory=list)


class TranslateUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    billed_units: int
    billing_model: Literal["translation_token", "ocr_image"] = "translation_token"


class TranslateMetadata(BaseModel):
    total_chars_processed: int
    domain_detected: Domain | None = None
    glossary_terms_applied: int = 0


class TranslateResult(BaseModel):
    translations: list[TranslationItem]
    metadata: TranslateMetadata


class TranslateResponse(BaseModel):
    request_id: str
    service: Literal["translate"] = "translate"
    processing_time_ms: int
    cached: bool
    model_used: str
    result: TranslateResult
    usage: TranslateUsage


class DetectLanguageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=50_000)


class DetectLanguageResponse(BaseModel):
    request_id: str
    detected_language: str
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[tuple[str, float]] = Field(default_factory=list)


class OcrRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str | None = Field(
        default=None,
        description="Base64 string. Nếu cùng image_url → ưu tiên field này.",
    )
    image_url: str | None = Field(
        default=None,
        description="HTTPS URL. Phải pass SSRF guard.",
        pattern=r"^https://.+",
    )
    include_translation: bool = False
    target_lang: str = Field(default="en", min_length=2, max_length=8)
    return_bbox: bool = True
    scene_hint: SceneHint = "auto"

    @model_validator(mode="after")
    def _check_input(self) -> OcrRequest:
        if not self.image and not self.image_url:
            raise ValueError("Phải cung cấp image hoặc image_url")
        return self


class TextBlock(BaseModel):
    id: str
    text: str
    translation: str | None = None
    language: str
    position: str
    bbox_relative: list[float] = Field(min_length=4, max_length=4)
    confidence: Literal["high", "medium", "low"]
    type: str = "other"
    font_style: str = "regular"


class OcrResult(BaseModel):
    detected_languages: list[str]
    primary_language: str
    scene_type: str
    text_blocks: list[TextBlock]
    structured_data: dict[str, list[object]] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)


class OcrResponse(BaseModel):
    request_id: str
    service: Literal["ocr"] = "ocr"
    processing_time_ms: int
    cached: bool
    model_used: str
    result: OcrResult
    usage: TranslateUsage


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str = "0.1.0"
    components: dict[str, object] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: str | None = None
