"""Pydantic v2 schemas cho Phase 2.

Scope đơn giản theo chỉ đạo anh Thịnh:
- /v1/translate: single hoặc batch text
- /v1/json: array of strings
- Language detect tự động khi source_lang=auto
- KHÔNG cache, KHÔNG domain, KHÔNG formality, KHÔNG glossary, KHÔNG auth, KHÔNG rate limit
"""
from __future__ import annotations

from typing import Any, Literal

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


# Pattern check ISO 639-1/2 hoặc "auto" (cho HTML request)
_HTML_MAX_BYTES = 5_000_000  # 5MB


class TranslateHtmlRequest(BaseModel):
    """Translate HTML preserving structure. Skip script/style/code, walk DOM,
    XLIFF-style placeholder cho inline tags."""
    model_config = ConfigDict(extra="forbid")

    html: str = Field(
        min_length=1,
        max_length=_HTML_MAX_BYTES,
        description="HTML cần dịch (max 5MB hoặc 5M ký tự).",
    )
    source_lang: str = Field(
        default="auto",
        pattern=LANG_PATTERN,
        description="auto = model tự detect.",
    )
    target_lang: str = Field(
        min_length=2,
        max_length=8,
        pattern=LANG_PATTERN,
        description="Ngôn ngữ đích (bắt buộc).",
    )
    translate_attributes: bool = Field(
        default=True,
        description="Dịch alt/title/aria-label/placeholder + meta description. False = chỉ text nodes.",
    )
    ignore_terms: list[str] = Field(
        default_factory=list,
        max_length=500,
        description=(
            "Danh sách từ/cụm từ KHÔNG dịch (giữ nguyên). "
            "Match theo word boundary, mặc định case-sensitive. "
            "Vd: ['Apple', 'MacBook Pro', 'iPhone 15 Pro Max']."
        ),
    )
    ignore_case: bool = Field(
        default=False,
        description="Case-insensitive match cho ignore_terms (default false: 'Apple' ≠ 'apple').",
    )

    @model_validator(mode="after")
    def _check(self) -> TranslateHtmlRequest:
        if self.target_lang == "auto":
            raise ValueError("target_lang không được là 'auto'")
        if not self.html.strip():
            raise ValueError("html không được rỗng")
        for t in self.ignore_terms:
            if not t or not t.strip():
                raise ValueError("ignore_terms không được chứa string rỗng")
            if len(t) > 200:
                raise ValueError(f"ignore_terms[*] vượt 200 ký tự: {t[:30]!r}...")
        return self


class HtmlHealthInfo(BaseModel):
    """Metric độ "sạch" của HTML input — cho client biết HTML có cần fix không."""
    health: Literal["clean", "minor", "moderate"] = Field(
        description="severe sẽ raise 422, không xuất hiện ở response 200.",
    )
    error_rate: float = Field(description="Tỷ lệ error trên tổng tag (đã weighted fatal × 5).")
    structure_diff: float = Field(description="|raw_open_tags − parsed_tags| / raw_open_tags.")
    errors_total: int
    fatals_total: int
    parse_tier: Literal["lxml", "html5lib", "fragment_wrap"] = Field(
        description="Parser đã dùng. fragment_wrap = input không có <html>, đã wrap tự động.",
    )


class TranslateJsonObjectRequest(BaseModel):
    """Translate JSON object/array recursively. String values → translate,
    structure preserved. 3 exclusion options: words, paths, common keys."""
    model_config = ConfigDict(extra="forbid")

    json_data: Any = Field(
        description=(
            "JSON cần dịch — object, array, hoặc string. Mọi string value sẽ được dịch trừ"
            " khi match exclusion rules hoặc skip filter (numbers, URLs, emails, IDs...)."
        ),
    )
    source_lang: str = Field(default="auto", pattern=LANG_PATTERN)
    target_lang: str = Field(
        min_length=2,
        max_length=8,
        pattern=LANG_PATTERN,
    )

    words_not_to_translate: str | list[str] = Field(
        default_factory=list,
        description=(
            "Từ/cụm từ KHÔNG dịch (giữ nguyên trong text). Format: string với separator"
            " `;` (vd `\"Earbuds; New York\"`) HOẶC list (vd `[\"Earbuds\", \"New York\"]`)."
            " Word boundary, default case-sensitive."
        ),
        examples=["Earbuds; New York", ["Earbuds", "New York"]],
    )
    paths_to_exclude: str | list[str] = Field(
        default_factory=list,
        description=(
            "JSON path không dịch (dot-notation, `*` = wildcard array index). "
            "Format: string với separator `;` HOẶC list. Pattern khớp prefix subtree. "
            "Vd: `product.media.img_desc` skip path đó + subtree; `items.*.url` skip mọi index."
        ),
        examples=["product.media.img_desc; items.*.image_url", ["product.media.img_desc"]],
    )
    common_keys_to_exclude: str | list[str] = Field(
        default_factory=list,
        description=(
            "Tên key KHÔNG dịch ở BẤT KỲ depth nào trong JSON. "
            "Format: string với separator `;` HOẶC list. "
            "Vd: `name; price` → mọi `.name` và `.price` ở any nesting đều skip."
        ),
        examples=["name; price", ["name", "price"]],
    )
    ignore_case: bool = Field(
        default=False,
        description="Case-insensitive match cho words_not_to_translate.",
    )
    skip_non_text: bool = Field(
        default=True,
        description=(
            "Skip strings không phải human text: số, $99.99, 50%, URL, email, UUID, hash,"
            " ISO date, code/ID. Set false để dịch hết."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> TranslateJsonObjectRequest:
        if self.target_lang == "auto":
            raise ValueError("target_lang không được là 'auto'")
        if self.json_data is None:
            raise ValueError("json_data không được null")
        # Validate type
        if not isinstance(self.json_data, (dict, list, str)):
            raise ValueError(
                f"json_data phải là object, array hoặc string — nhận {type(self.json_data).__name__}"
            )
        return self


class JsonTranslationStats(BaseModel):
    strings_translated: int = Field(description="Số string value đã dịch.")
    strings_skipped: int = Field(
        description="Số string value bị skip (filter non-text + exclusion rules)."
    )
    chars_translated: int = Field(description="Tổng ký tự đã đẩy vào model.")


class TranslateJsonObjectResponse(BaseModel):
    request_id: str
    processing_time_ms: int
    json_data: Any = Field(description="JSON đã dịch, structure preserved.")
    detected_source_lang: str | None = Field(
        default=None,
        description="Lang dominant khi source_lang=auto. None nếu explicit.",
    )
    stats: JsonTranslationStats
    warnings: list[str] = Field(default_factory=list)


class TranslateHtmlResponse(BaseModel):
    request_id: str
    processing_time_ms: int
    html: str = Field(description="HTML đã dịch, structure preserved.")
    detected_source_lang: str | None = Field(
        default=None,
        description="Lang dominant của batch khi source_lang=auto. None nếu explicit lang.",
    )
    health: HtmlHealthInfo
    segments_translated: int = Field(description="Số segment đã dịch (text nodes + attributes).")
    chars_translated: int = Field(description="Tổng ký tự đã đẩy vào model.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Cảnh báo: parser auto-fix, placeholder mất, post-translate diff, ...",
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
