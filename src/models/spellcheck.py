"""Spellchecker-based baseline model.

Splits each identifier into constituent words using :mod:`text_utils`,
runs a spellchecker on each word, then reassembles.
"""

from __future__ import annotations

from typing import Dict, List

from .base import NameFixer
from ..text_utils import split_identifier, reassemble_identifier

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None  # type: ignore[assignment,misc]


class SpellCheckerFixer(NameFixer):
    """Baseline: spellcheck each word inside an identifier."""

    def __init__(self, language: str = "en") -> None:
        if SpellChecker is None:
            raise ImportError(
                "pyspellchecker is required.  Install it with:  uv add pyspellchecker"
            )
        self._spell = SpellChecker(language=language)

    def fix_names(self, code: str, names: List[str]) -> Dict[str, str]:
        """Spellcheck each identifier and return corrections.

        ``code`` is ignored in this baseline — only the name strings are used.
        Names shorter than 3 characters are skipped (spellcheckers are
        unreliable on very short tokens).
        """
        fixes: Dict[str, str] = {}
        for name in names:
            if len(name) < 3:
                continue
            corrected = self._spellcheck_identifier(name)
            if corrected and corrected != name:
                fixes[name] = corrected
        return fixes

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _spellcheck_identifier(self, name: str) -> str:
        """Correct an entire identifier, preserving case and separators."""
        words = split_identifier(name)
        if not words:
            return name
        corrected_words = [self._correct_word(w) for w in words]
        if corrected_words == words:
            return name
        return reassemble_identifier(name, corrected_words)

    def _correct_word(self, word: str) -> str:
        """Spellcheck a single alpha-word, preserving case style."""
        if not word.isalpha():
            return word
        corrected = self._spell.correction(word.lower())
        if corrected is None:
            return word
        return corrected
