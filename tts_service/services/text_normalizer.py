"""Text normalization trước khi đưa vào Chatterbox.

Chatterbox 0.5B (và đa phần TTS LLM autoregressive) KHÔNG có normalizer
built-in. Nếu đưa raw "1000 dollars" vào, model có thể đọc thành "hundred"
hoặc nuốt chữ số. Pipeline này pre-process text để chuyển:

- Currency: $1000  → 1000 dollars     (theo locale của language_id)
- Percent:  50%    → 50 percent       (theo locale)
- Decimal:  3.14   → three point one four
- Integer:  1000   → one thousand     (qua num2words)
- Thousand sep:  1,000 → 1000  (English style; chỉ strip nếu match)

Hỗ trợ 23 ngôn ngữ Chatterbox. Một số ngôn ngữ num2words không cover (vd
sw, el, ms) → fallback English (thêm log warning).
"""
from __future__ import annotations

import re
from typing import Final

from num2words import num2words

from utils.logging import get_logger

log = get_logger(__name__)


# Map Chatterbox language_id → num2words lang code.
# num2words 0.5.13 supports: ar, az, be, ca, cs, cy, da, de, en, es, fa, fi,
#   fr, he, hu, id, is, it, ja, ko, lt, lv, nl, no, pl, pt, ro, ru, sk, sl,
#   sr, sv, te, th, tr, uk, vi (KHÔNG có zh, hi, el, ms, sw, ko-only).
# Các lang không cover → fallback English.
_LANG_MAP: Final[dict[str, str]] = {
    "ar": "ar", "da": "da", "de": "de", "el": "en",
    "en": "en", "es": "es", "fi": "fi", "fr": "fr",
    "he": "he", "hi": "en", "it": "it", "ja": "ja",
    "ko": "ko", "ms": "id", "nl": "nl", "no": "no",
    "pl": "pl", "pt": "pt", "ru": "ru", "sv": "sv",
    "sw": "en", "tr": "tr", "zh": "en",
}

# Currency symbol → tên loại tiền theo locale.
_CURRENCY: Final[dict[str, dict[str, str]]] = {
    "en": {"$": "dollars", "€": "euros", "£": "pounds", "¥": "yen", "₹": "rupees", "₩": "won"},
    "fr": {"$": "dollars", "€": "euros", "£": "livres", "¥": "yens"},
    "de": {"$": "Dollar", "€": "Euro", "£": "Pfund", "¥": "Yen"},
    "es": {"$": "dólares", "€": "euros", "£": "libras", "¥": "yenes"},
    "it": {"$": "dollari", "€": "euro", "£": "sterline", "¥": "yen"},
    "pt": {"$": "dólares", "€": "euros", "£": "libras", "¥": "ienes"},
    "nl": {"$": "dollar", "€": "euro", "£": "pond", "¥": "yen"},
    "ru": {"$": "долларов", "€": "евро", "£": "фунтов", "¥": "иен"},
    "pl": {"$": "dolarów", "€": "euro", "£": "funtów", "¥": "jenów"},
    "tr": {"$": "dolar", "€": "euro", "£": "sterlin", "¥": "yen"},
    "zh": {"$": "美元", "€": "欧元", "£": "英镑", "¥": "日元", "₩": "韩元"},
    "ja": {"$": "ドル", "€": "ユーロ", "£": "ポンド", "¥": "円", "₩": "ウォン"},
    "ko": {"$": "달러", "€": "유로", "£": "파운드", "¥": "엔", "₩": "원"},
    "ar": {"$": "دولار", "€": "يورو", "£": "جنيه", "¥": "ين"},
    "hi": {"$": "डॉलर", "€": "यूरो", "£": "पाउंड", "¥": "येन", "₹": "रुपये"},
}

# Percentage word per locale.
_PERCENT: Final[dict[str, str]] = {
    "en": "percent", "fr": "pour cent", "de": "Prozent", "es": "por ciento",
    "it": "per cento", "pt": "por cento", "nl": "procent", "ru": "процентов",
    "pl": "procent", "tr": "yüzde", "zh": "百分之", "ja": "パーセント",
    "ko": "퍼센트", "ar": "بالمئة", "hi": "प्रतिशत",
}

# Số nguyên có dấu phẩy thousand separator: 1,000 / 1,234,567.
# CHỈ strip nếu match đúng pattern (group đầu 1-3 chữ số, các group sau đúng 3 chữ số).
_THOUSAND_SEP_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+)(?!\d)")

# Currency PREFIX: <symbol><optional space><number>. EN/JA/ZH style.
_CURRENCY_PREFIX_RE = re.compile(r"([$€£¥₹₩₫฿])\s*(\d+(?:\.\d+)?)")

# Currency SUFFIX: <number><optional space><symbol>. FR/DE/IT/ES style.
_CURRENCY_SUFFIX_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([$€£¥₹₩₫฿])")

# Percentage: <number><optional space>%
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Số: integer hoặc decimal. KHÔNG match số đã có dấu phẩy thousand sep (đã xử lý ở step 1).
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Giới hạn số quá lớn (num2words có thể OOM hoặc trả string khổng lồ).
_MAX_INT = 10**15


def _safe_num2words(num: int | float, lang: str) -> str:
    """Wrap num2words với fallback English nếu lang không hỗ trợ."""
    n2w_lang = _LANG_MAP.get(lang, "en")
    try:
        return num2words(num, lang=n2w_lang)
    except (NotImplementedError, KeyError, Exception) as exc:  # noqa: BLE001
        log.warning("num2words_failed", num=num, lang=n2w_lang, error=str(exc))
        if n2w_lang != "en":
            try:
                return num2words(num, lang="en")
            except Exception:  # noqa: BLE001
                pass
        return str(num)


def _strip_thousand_separators(text: str) -> str:
    """1,234,567 → 1234567 (chỉ match pattern đúng)."""
    return _THOUSAND_SEP_RE.sub(lambda m: m.group(1).replace(",", ""), text)


def _replace_currency(text: str, language_id: str) -> str:
    """$1000 → 1000 dollars (prefix style EN/JA/ZH).
    50€ → 50 euros (suffix style FR/DE/IT/ES).
    Cover cả 2 thứ tự để xử lý input đa locale.
    """
    locale_map = _CURRENCY.get(language_id, _CURRENCY["en"])

    def _sub_prefix(m: re.Match[str]) -> str:
        sym, num = m.group(1), m.group(2)
        word = locale_map.get(sym)
        if word is None:
            return m.group(0)
        return f"{num} {word}"

    def _sub_suffix(m: re.Match[str]) -> str:
        num, sym = m.group(1), m.group(2)
        word = locale_map.get(sym)
        if word is None:
            return m.group(0)
        return f"{num} {word}"

    text = _CURRENCY_PREFIX_RE.sub(_sub_prefix, text)
    text = _CURRENCY_SUFFIX_RE.sub(_sub_suffix, text)
    return text


def _replace_percent(text: str, language_id: str) -> str:
    """50% → 50 percent (per-locale)."""
    word = _PERCENT.get(language_id, "percent")
    return _PERCENT_RE.sub(lambda m: f"{m.group(1)} {word}", text)


def _replace_numbers(text: str, language_id: str) -> str:
    """1000 → one thousand. Convert mọi số còn lại sang words."""

    def _sub(m: re.Match[str]) -> str:
        s = m.group()
        try:
            if "." in s:
                num: int | float = float(s)
            else:
                num = int(s)
                if abs(num) > _MAX_INT:
                    return s  # quá lớn, bỏ qua
            return _safe_num2words(num, language_id)
        except (ValueError, OverflowError):
            return s

    return _NUMBER_RE.sub(_sub, text)


def normalize_text(text: str, language_id: str) -> str:
    """Pipeline normalize đầy đủ. KHÔNG raise — fallback giữ nguyên text gốc.

    Thứ tự xử lý quan trọng:
      1. Strip thousand separator (1,000 → 1000)
      2. Currency ($1000 → 1000 dollars)
      3. Percent (50% → 50 percent)
      4. Số còn lại → words
    """
    if not text:
        return text
    try:
        text = _strip_thousand_separators(text)
        text = _replace_currency(text, language_id)
        text = _replace_percent(text, language_id)
        text = _replace_numbers(text, language_id)
        return text
    except Exception as exc:  # noqa: BLE001
        log.warning("normalize_failed", error=str(exc), language=language_id)
        return text
