"""Aggregate-safe text and social-media feature extraction for data quality checks."""

from __future__ import annotations

import math
import unicodedata

import emoji
import pandas as pd
import regex

from cyberbullying_detection.preprocessing.text import (
    HASHTAG_PATTERN,
    URL_PATTERN,
    USER_MENTION_PATTERN,
)

REPEATED_CHARACTER_PATTERN = regex.compile(r"(?s)(.)\1{2,}")

FEATURE_COLUMNS = [
    "character_length",
    "word_count",
    "url_count",
    "user_mention_count",
    "hashtag_count",
    "has_emoji",
    "uppercase_ratio",
    "punctuation_count",
    "repeated_character_count",
]


def _text_or_empty(value: object) -> str:
    """Return text for descriptive features without changing source data."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _uppercase_ratio(text: str) -> float:
    """Return uppercase alphabetic characters divided by all alphabetic characters."""
    alphabetic_count = sum(character.isalpha() for character in text)
    if alphabetic_count == 0:
        return 0.0
    return sum(character.isupper() for character in text) / alphabetic_count


def _punctuation_count(text: str) -> int:
    """Count Unicode punctuation characters, including social-media punctuation."""
    return sum(unicodedata.category(character).startswith("P") for character in text)


def extract_text_features(texts: pd.Series) -> pd.DataFrame:
    """Return deterministic per-row descriptive features without retaining new text."""
    records: list[dict[str, float | int]] = []
    for value in texts:
        text = _text_or_empty(value)
        records.append(
            {
                "character_length": len(text),
                "word_count": len(text.split()),
                "url_count": len(URL_PATTERN.findall(text)),
                "user_mention_count": len(USER_MENTION_PATTERN.findall(text)),
                "hashtag_count": len(HASHTAG_PATTERN.findall(text)),
                "has_emoji": int(emoji.emoji_count(text) > 0),
                "uppercase_ratio": _uppercase_ratio(text),
                "punctuation_count": _punctuation_count(text),
                "repeated_character_count": len(
                    REPEATED_CHARACTER_PATTERN.findall(text)
                ),
            }
        )
    return pd.DataFrame(records, index=texts.index, columns=FEATURE_COLUMNS)
