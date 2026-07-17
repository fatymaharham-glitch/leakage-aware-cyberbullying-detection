"""Stable hashes for deterministic duplicate grouping."""

from __future__ import annotations

import hashlib


def text_hash(value: str) -> str:
    """Return a SHA-256 hash for a canonical text value."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
