"""Deterministic text normalisation and pre-modelling preprocessing pipelines."""

from __future__ import annotations

from collections.abc import Callable
import math
import unicodedata

import emoji
import regex
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_PLACEHOLDER = "<URL>"
USER_PLACEHOLDER = "<USER>"

URL_PATTERN = regex.compile(r'(?i)\b(?:https?://|www\.)[^\s<>"]+')
USER_MENTION_PATTERN = regex.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_PATTERN = regex.compile(r"(?<!\w)#([\p{L}\p{N}_]+)")
WHITESPACE_PATTERN = regex.compile(r"\s+")


def _coerce_text(value: object) -> str:
    """Return a safe string while preserving validation's responsibility for bad rows."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def normalize_unicode(value: object) -> str:
    """Apply Unicode NFKC normalisation without changing case or tokens."""
    return unicodedata.normalize("NFKC", _coerce_text(value))


def normalize_whitespace(value: str) -> str:
    """Collapse all whitespace into single spaces and trim the result."""
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def preprocess_p0(value: object) -> str:
    """Apply minimal, loss-conscious social-media normalisation.

    P0 retains casing, hashtags, punctuation, emoji, repeated characters, and stop
    words. URLs and user mentions become stable placeholders.
    """
    text = normalize_unicode(value)
    text = URL_PATTERN.sub(f" {URL_PLACEHOLDER} ", text)
    text = USER_MENTION_PATTERN.sub(f" {USER_PLACEHOLDER} ", text)
    return normalize_whitespace(text)


def preprocess_p1(value: object) -> str:
    """Apply social-media-aware normalisation while retaining useful signals.

    Hashtags retain their lexical content through a ``hashtag_`` prefix. Emoji are
    converted to text aliases; negation remains because no stop-word removal occurs.
    """
    text = normalize_unicode(value).lower()
    text = URL_PATTERN.sub(" <url> ", text)
    text = USER_MENTION_PATTERN.sub(" <user> ", text)
    text = HASHTAG_PATTERN.sub(r" hashtag_\1 ", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    return normalize_whitespace(text)


def preprocess_p2(
    value: object,
    *,
    enable_lemmatization: bool = False,
    lemmatizer: Callable[[str], str] | None = None,
) -> str:
    """Apply aggressive lexical cleaning for a later controlled comparison.

    Lemmatization remains disabled by default because its language model and error
    profile are an explicit future methodological choice. Supplying a lemmatizer is
    required only when that option is enabled.
    """
    if enable_lemmatization and lemmatizer is None:
        raise ValueError("A lemmatizer is required when enable_lemmatization=True.")

    text = normalize_unicode(value).lower()
    text = URL_PATTERN.sub(" ", text)
    text = USER_MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r" \1 ", text)
    text = regex.sub(r"[^\p{L}\p{N}\s]", " ", text)
    tokens = [token for token in normalize_whitespace(text).split() if token not in ENGLISH_STOP_WORDS]
    if enable_lemmatization:
        tokens = [lemmatizer(token) for token in tokens]
    return " ".join(tokens)


def duplicate_matching_normalize(value: object) -> str:
    """Normalise text for duplicate matching while preserving non-URL content.

    This intentionally differs from P0/P1/P2: it retains punctuation and emoji but
    applies NFKC, lowercasing, URL/mention placeholders, and whitespace collapsing.
    """
    text = normalize_unicode(value).lower()
    text = URL_PATTERN.sub(" <url> ", text)
    text = USER_MENTION_PATTERN.sub(" <user> ", text)
    return normalize_whitespace(text)


def normalize_tweet_text(value: object) -> str:
    """Backward-compatible alias for the P0 preparation pipeline."""
    return preprocess_p0(value)


def canonicalize_text(value: object) -> str:
    """Return the canonical duplicate-matching representation of a tweet."""
    return duplicate_matching_normalize(value)
