"""Pydantic v2 schemas cho Phase 2.

Scope đơn giản theo chỉ đạo anh Thịnh:
- /v1/translate: single hoặc batch text
- /v1/json: array of strings
- Language detect tự động khi source_lang=auto
- KHÔNG cache, KHÔNG domain, KHÔNG formality, KHÔNG glossary, KHÔNG auth, KHÔNG rate limit
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Pattern check ISO 639-1/2 hoặc "auto"
LANG_PATTERN = r"^(auto|[a-z]{2,3}(-[A-Z]{2})?)$"


class TranslateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | list[str] = Field(
        description="Text đơn hoặc list (max 100 phần tử, mỗi phần tử max 50000 ký tự)."
    )
    source_lang: str = Field(
        default="auto",
        pattern=LANG_PATTERN,
        description="auto = model tự detect; hoặc ISO code: vi, en, ja, zh, ko, fr, de...",
    )
    target_lang: str = Field(
        min_length=2,
        max_length=8,
        pattern=LANG_PATTERN,
        description="Ngôn ngữ đích (bắt buộc): vi, en, ja, zh, ko, fr, de...",
    )

    @model_validator(mode="after")
    def _check_text_size(self) -> TranslateRequest:
        if isinstance(self.text, str):
            if not self.text.strip():
                raise ValueError("text không được rỗng")
            if len(self.text) > 50_000:
                raise ValueError("text vượt 50000 ký tự")
        else:
            if len(self.text) == 0:
                raise ValueError("text list rỗng")
            if len(self.text) > 100:
                raise ValueError("batch vượt 100 phần tử")
            for i, t in enumerate(self.text):
                if not t.strip():
                    raise ValueError(f"text[{i}] không được rỗng")
                if len(t) > 50_000:
                    raise ValueError(f"text[{i}] vượt 50000 ký tự")
        return self

    @model_validator(mode="after")
    def _check_target_not_auto(self) -> TranslateRequest:
        if self.target_lang == "auto":
            raise ValueError("target_lang không được là 'auto'")
        return self


class TranslationItem(BaseModel):
    source_text: str
    translated_text: str
    detected_source_lang: str = Field(
        description="Ngôn ngữ nguồn (model detect khi source_lang=auto, hoặc echo input)",
    )
    target_lang: str


class TranslateResponse(BaseModel):
    request_id: str
    service: Literal["translate"] = "translate"
    processing_time_ms: int
    model_used: str
    translations: list[TranslationItem]


class JsonTranslateRequest(BaseModel):
    """Translate 1 mảng string. Output là mảng string cùng thứ tự."""
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(
        min_length=1,
        max_length=100,
        description="Mảng string cần dịch (max 100 phần tử, mỗi phần tử max 50000 ký tự).",
    )
    source_lang: str = Field(
        default="auto",
        pattern=LANG_PATTERN,
    )
    target_lang: str = Field(
        min_length=2,
        max_length=8,
        pattern=LANG_PATTERN,
    )

    @model_validator(mode="after")
    def _check_strings(self) -> JsonTranslateRequest:
        for i, t in enumerate(self.texts):
            if not t.strip():
                raise ValueError(f"texts[{i}] không được rỗng")
            if len(t) > 50_000:
                raise ValueError(f"texts[{i}] vượt 50000 ký tự")
        if self.target_lang == "auto":
            raise ValueError("target_lang không được là 'auto'")
        return self


class JsonTranslateResponse(BaseModel):
    request_id: str
    service: Literal["translate-json"] = "translate-json"
    processing_time_ms: int
    model_used: str
    translations: list[str] = Field(
        description="Translation tương ứng từng phần tử input, cùng thứ tự",
    )
    detected_source_lang: str = Field(
        description="Ngôn ngữ nguồn dominant của batch (lấy từ phần tử đầu)",
    )
    target_lang: str


class DictRequest(BaseModel):
    """Tra từ điển: user nhập từ ở native_lang, trả về entry kiểu Cambridge của target_lang."""
    model_config = ConfigDict(extra="forbid")

    word: str = Field(
        min_length=1,
        max_length=200,
        description="Từ cần tra (ở native_lang). Có thể là phrase ngắn.",
    )
    native_lang: str = Field(
        min_length=2,
        max_length=8,
        pattern=LANG_PATTERN,
        description="Ngôn ngữ mẹ đẻ của user (vd vi, en, ja...).",
    )
    target_lang: str = Field(
        default="en",
        min_length=2,
        max_length=8,
        pattern=LANG_PATTERN,
        description="Ngôn ngữ học (mặc định en).",
    )

    @model_validator(mode="after")
    def _check_word(self) -> DictRequest:
        if not self.word.strip():
            raise ValueError("word không được rỗng")
        if self.native_lang == "auto" or self.target_lang == "auto":
            raise ValueError("native_lang và target_lang không được là 'auto'")
        if self.native_lang == self.target_lang:
            raise ValueError("native_lang và target_lang phải khác nhau")
        return self


class DictDefinition(BaseModel):
    part_of_speech: str = Field(
        description="noun | verb | adjective | adverb | preposition | conjunction | interjection | phrase",
    )
    definition_target: str = Field(
        description="Định nghĩa ở target_lang, kiểu Cambridge — concise và rõ ngữ cảnh",
    )
    definition_native: str = Field(
        description="Bản dịch định nghĩa sang native_lang",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="1–3 câu ví dụ tự nhiên ở target_lang",
    )


class DictResponse(BaseModel):
    request_id: str
    service: Literal["dict"] = "dict"
    processing_time_ms: int
    model_used: str

    input_word: str
    native_lang: str
    target_lang: str
    headword: str = Field(description="Từ chính ở target_lang (vd 'freedom')")
    ipa: str = Field(description="Phiên âm IPA, vd /ˈfriː.dəm/")
    definitions: list[DictDefinition] = Field(
        description="1–5 định nghĩa, mỗi định nghĩa kèm part_of_speech và examples",
    )


class OcrRequest(BaseModel):
    """OCR ảnh: input base64, output text blocks kèm bbox pixel coordinates."""
    model_config = ConfigDict(extra="forbid")

    image: str = Field(
        min_length=1,
        description="Ảnh dạng base64 (PNG/JPG/WebP). Max 10MB sau decode.",
    )
    lang: str = Field(
        default="auto",
        description=(
            "Language pack PaddleOCR PP-OCRv5. Hỗ trợ:\n"
            "- auto: tự detect (en trước, fallback CJK nếu confidence thấp)\n"
            "- en: English + Latin extended (gồm tiếng Việt diacritics)\n"
            "- vi: alias của 'en' (PP-OCRv5 'en' model handle Latin extended)\n"
            "- ch: Trung giản thể + Anh\n"
            "- chinese_cht: Trung phồn thể\n"
            "- japan, korean: Nhật, Hàn\n"
            "- ru: East Slavic (Nga, Ukraina, Belarus — PaddleOCR map sang model eslav)"
        ),
    )


class OcrTextBlock(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[list[int]] = Field(
        min_length=4,
        max_length=4,
        description="4 góc theo thứ tự [top-left, top-right, bottom-right, bottom-left], pixel coords",
    )


class OcrResponse(BaseModel):
    request_id: str
    service: Literal["ocr"] = "ocr"
    processing_time_ms: int
    lang: str = Field(description="Language pack user request (có thể là 'auto')")
    detected_lang: str = Field(
        description="Language pack thực tế dùng để recognize (sau khi auto-detect)",
    )
    image_width: int
    image_height: int
    full_text: str = Field(description="Tất cả text blocks nối lại bằng \\n theo thứ tự đọc")
    text_blocks: list[OcrTextBlock]


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str = "0.2.0"
    components: dict[str, object] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: str | None = None
