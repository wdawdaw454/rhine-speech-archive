"""Chinese text normalization for CER / eval."""
from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(
    r"[　-〿＀-￯"
    r"…—‘’“”、。，！？"
    r"\!\?\.,;:\"'\(\)\[\]\{\}]"
)

_FULLWIDTH_DIGIT = str.maketrans("０１２３４５６７８９", "0123456789")
_FULLWIDTH_UPPER = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
)
_FULLWIDTH_LOWER = str.maketrans(
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "abcdefghijklmnopqrstuvwxyz",
)


def _convert_fullwidth(text: str) -> str:
    return (
        text.translate(_FULLWIDTH_DIGIT)
        .translate(_FULLWIDTH_UPPER)
        .translate(_FULLWIDTH_LOWER)
    )


def _try_simplified(text: str) -> str:
    try:
        import zhconv
        return zhconv.convert(text, "zh-cn")
    except ImportError:
        return text


def normalize_chinese_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_whitespace: bool = True,
    to_simplified_chinese: bool = True,
    normalize_width: bool = True,
    normalize_numbers: bool = True,
) -> str:
    if not text:
        return text

    if normalize_width:
        text = _convert_fullwidth(text)

    if to_simplified_chinese:
        text = _try_simplified(text)

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = _PUNCT_RE.sub("", text)

    if remove_whitespace:
        text = "".join(text.split())

    # Apply NFKC only when punctuation is being removed, to avoid converting
    # full-width punctuation to its half-width form (which downstream callers
    # may want to preserve when remove_punctuation=False).
    if normalize_numbers and remove_punctuation:
        text = unicodedata.normalize("NFKC", text)

    return text