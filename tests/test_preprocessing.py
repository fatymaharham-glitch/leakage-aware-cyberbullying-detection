"""Tests for deterministic social-media preprocessing functions."""

from __future__ import annotations

from cyberbullying_detection.preprocessing.text import preprocess_p0, preprocess_p1, preprocess_p2


def test_p0_preserves_case_and_social_signals_while_replacing_addresses() -> None:
    result = preprocess_p0("Hello @Person! Visit https://example.test #Topic")

    assert "Hello" in result
    assert "<USER>" in result
    assert "<URL>" in result
    assert "#Topic" in result
    assert "!" in result


def test_p1_lowercases_and_preserves_hashtag_content() -> None:
    result = preprocess_p1("Hello @Person #Topic")

    assert result == "hello <user> hashtag_topic"


def test_p2_removes_addresses_punctuation_and_stop_words() -> None:
    result = preprocess_p2("This is a HELLO, @Person! https://example.test #Topic")

    assert "hello" in result
    assert "topic" in result
    assert "@" not in result
    assert "http" not in result
    assert "," not in result
