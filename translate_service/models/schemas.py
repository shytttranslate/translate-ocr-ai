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


class TranslationDetected(BaseModel):
    """Item khi source_lang=auto: kèm detected_source_lang riêng cho mỗi text."""
    translated_text: str
    detected_source_lang: str


class TranslateResponse(BaseModel):
    request_id: str
    processing_time_ms: int
    translations: list[TranslationDetected] | list[str] = Field(
        description=(
            "Theo thứ tự input. source_lang=auto → list[{translated_text,"
            " detected_source_lang}]. Explicit lang → list[str]."
        ),
    )


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
    processing_time_ms: int
    translations: list[TranslationDetected] | list[str] = Field(
        description=(
            "Theo thứ tự input. source_lang=auto → list[{translated_text,"
            " detected_source_lang}]. Explicit lang → list[str]."
        ),
    )


def _camel(s: str) -> str:
    """snake_case → camelCase cho JSON alias."""
    parts = s.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


_CAMEL_CFG = ConfigDict(
    populate_by_name=True,
    alias_generator=_camel,
    extra="forbid",
)


class DictRequest(BaseModel):
    """Tra từ điển đa ngôn ngữ — phục vụ người học ngoại ngữ.

    User là người nói `native_lang` (mẹ đẻ — vd vi), đang học `target_lang`
    (ngoại ngữ — vd en). Tra từ ở `target_lang`, nhận giải nghĩa bằng `native_lang`.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=_camel,
        extra="forbid",
    )

    word: str = Field(
        min_length=1,
        max_length=200,
        description="Từ / cụm từ cần tra (ở target_lang — ngôn ngữ đang học).",
    )
    native_lang: str = Field(
        default="vi",
        min_length=2,
        max_length=8,
        pattern=LANG_PATTERN,
        description="Ngôn ngữ mẹ đẻ của user — output meaning/giải nghĩa ở đây (mặc định vi).",
    )
    target_lang: str = Field(
        min_length=2,
        max_length=8,
        pattern=LANG_PATTERN,
        description="Ngôn ngữ user đang học — ngôn ngữ của từ user nhập: en, ja, zh, ko, fr...",
    )

    @model_validator(mode="after")
    def _check_request(self) -> DictRequest:
        if not self.word.strip():
            raise ValueError("word không được rỗng")
        if self.native_lang == "auto" or self.target_lang == "auto":
            raise ValueError("native_lang và target_lang không được là 'auto'")
        if self.native_lang == self.target_lang:
            raise ValueError("native_lang và target_lang phải khác nhau")
        return self


class DictPhonetic(BaseModel):
    """Phiên âm — IPA + romanization (cho ngôn ngữ non-Latin)."""
    model_config = _CAMEL_CFG

    ipa: str | None = Field(
        default=None,
        description="IPA chuẩn, vd '/bʊk/' hoặc '/ˈfriː.dəm/'. Null nếu không có.",
    )
    romanization: str | None = Field(
        default=None,
        description="Romanization cho non-Latin (pinyin, romaji, revised romanization)."
        " Null cho ngôn ngữ Latin.",
    )


class DictDefinition(BaseModel):
    """Một nét nghĩa — part of speech + meaning ngắn gọn."""
    model_config = _CAMEL_CFG

    part_of_speech: str = Field(
        description="noun|verb|adjective|adverb|preposition|conjunction|interjection"
        "|pronoun|determiner|particle|phrase|idiom",
    )
    meaning: str = Field(
        description="Định nghĩa concise ở native_lang (mẹ đẻ).",
    )


class DictExample(BaseModel):
    """Câu ví dụ hoặc cụm từ + bản dịch."""
    model_config = _CAMEL_CFG

    text: str = Field(description="Câu / cụm ở target_lang (ngoại ngữ).")
    meaning: str = Field(description="Bản dịch sang native_lang (mẹ đẻ).")


class DictWordRef(BaseModel):
    """Tham chiếu tới từ khác — synonym/antonym/related word."""
    model_config = _CAMEL_CFG

    text: str = Field(description="Từ / cụm tham chiếu ở target_lang.")
    meaning: str = Field(description="Giải nghĩa ngắn ở native_lang.")


class DictRelated(BaseModel):
    """Cụm related: đồng nghĩa, trái nghĩa, từ liên quan, mẹo nhớ."""
    model_config = _CAMEL_CFG

    synonyms: list[DictWordRef] = Field(default_factory=list)
    antonyms: list[DictWordRef] = Field(default_factory=list)
    related_words: list[DictWordRef] = Field(default_factory=list)
    memory_tips: list[str] = Field(
        default_factory=list,
        description="1–3 mẹo nhớ ngắn ở native_lang (mẹ đẻ).",
    )


class DictResponse(BaseModel):
    """Entry từ điển đa ngôn ngữ. JSON output dùng camelCase."""
    model_config = _CAMEL_CFG

    request_id: str
    processing_time_ms: int
    model_used: str

    word: str = Field(description="Từ chuẩn hoá (echo lại input đã trim).")
    native_lang: str
    target_lang: str

    phonetic: DictPhonetic
    short_meaning: str = Field(
        description="Nghĩa ngắn 1 dòng ở native_lang (mẹ đẻ) — phù hợp hiển thị nhanh.",
    )

    definitions: list[DictDefinition] = Field(description="1–2 nét nghĩa chính.")
    examples: list[DictExample] = Field(description="1–2 câu ví dụ tiêu biểu.")
    phrases: list[DictExample] = Field(
        default_factory=list,
        description="0–4 cụm từ / collocation thường dùng.",
    )
    related: DictRelated


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str = "0.2.0"
    components: dict[str, object] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: str | None = None
