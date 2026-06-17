"""Shared utilities for splitting identifiers into words and reassembling.

Handles both ``snake_case`` and ``CamelCase`` conventions.
"""

from __future__ import annotations

import re
from typing import List

# Matches CamelCase words: ``SomeWord`` → ``Some``, ``Word``;
# ``HTTPError`` → ``HTTP``, ``Error``.
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|$)")


def split_identifier(name: str) -> List[str]:
    """Split an identifier into its constituent alpha-words.

    Handles ``snake_case``, ``CamelCase``, and combinations like
    ``camelCase_snake``.  Leading underscores (``_private``) are preserved
    as a separate leading word.
    """
    leading_underscores = ""
    while name.startswith("_"):
        leading_underscores += "_"
        name = name[1:]

    if not name:
        return [leading_underscores] if leading_underscores else []

    words: List[str] = []
    if leading_underscores:
        words.append(leading_underscores)

    for part in name.split("_"):
        if not part:
            words.append("_")  # keep position for double-underscore.
            continue
        camel_words = _CAMEL_RE.findall(part)
        words.extend(camel_words or [part])

    return words


def reassemble_identifier(original: str, corrected_words: List[str]) -> str:
    """Reassemble an identifier from corrected words, preserving the
    original's underscore layout and CamelCase segmentation.

    ``corrected_words`` must correspond 1:1 to the words returned by
    :func:`split_identifier`.
    """
    original_words = split_identifier(original)
    if len(original_words) != len(corrected_words):
        return original

    # Restore each word's case to match the original word's style.
    restored = [
        _restore_case(ow, cw)
        for ow, cw in zip(original_words, corrected_words)
    ]

    # Rebuild by splitting on underscores and replacing CamelCase words
    # in each part.
    words_iter = iter(restored)
    parts = original.split("_")
    new_parts: List[str] = []
    for part in parts:
        if not part:
            new_parts.append("")
            continue
        camel_words = _CAMEL_RE.findall(part)
        new_camel = [next(words_iter) for _ in camel_words]
        new_parts.append("".join(new_camel))
    return "_".join(new_parts)


def _restore_case(original_word: str, corrected_lower: str) -> str:
    """Apply the original word's casing pattern to the corrected word."""
    cl = corrected_lower.lower()
    if original_word.isupper():
        return cl.upper()
    if original_word[0].isupper() and original_word[1:].islower():
        return cl.capitalize()
    if original_word.islower():
        return cl.lower()
    # Mixed case — can't infer intent, preserve original.
    return original_word
