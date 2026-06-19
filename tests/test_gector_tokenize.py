"""Unit tests for :mod:`src.gector.tokenize_code`.

Tests cover:
- ``code_tokens_from_source`` for valid and invalid Python.
- ``align_to_subwords`` and ``first_subword_mask``.
- ``name_tag_from_edit`` fallback logic.
"""

from __future__ import annotations

import unittest

from transformers import AutoTokenizer

from src.gector.tokenize_code import (
    code_tokens_from_source,
    align_to_subwords,
    first_subword_mask,
    name_tag_from_edit,
    CodeToken,
)
import tokenize as _tokenize


class CodeTokensFromSourceTests(unittest.TestCase):
    """Tests for :func:`code_tokens_from_source`."""

    def test_basic_identifier(self) -> None:
        tokens = code_tokens_from_source("def calculate(x): pass")
        # Should include NAME tokens, operators, keywords.
        names = [t.text for t in tokens if t.is_name]
        self.assertIn("calculate", names)
        self.assertIn("x", names)

    def test_keywords_are_names(self) -> None:
        """Python tokenize classifies keywords as NAME tokens."""
        tokens = code_tokens_from_source("def foo(): return bar")
        keywords = {"def", "return"}
        identifiers = {"foo", "bar"}
        for t in tokens:
            if t.text in keywords:
                self.assertTrue(t.is_name, f"Keyword {t.text!r} is token type NAME")
            if t.text in identifiers:
                self.assertTrue(t.is_name, f"{t.text!r} should be a NAME")

    def test_empty_source(self) -> None:
        tokens = code_tokens_from_source("")
        self.assertEqual(len(tokens), 0)

    def test_unparseable_source_fallback(self) -> None:
        """Unterminated string or other token errors fall back to whitespace splits."""
        tokens = code_tokens_from_source("foo bar baz (")
        self.assertGreater(len(tokens), 0)
        # All tokens in fallback mode are treated as NAME.
        for t in tokens:
            self.assertTrue(t.is_name)

    def test_comment_preserved(self) -> None:
        tokens = code_tokens_from_source("# this is a comment\nx = 1")
        comment_texts = [t.text for t in tokens if t.tok_type == _tokenize.COMMENT]
        self.assertGreaterEqual(len(comment_texts), 1)
        self.assertIn("this is a comment", comment_texts[0])

    def test_positions_present(self) -> None:
        tokens = code_tokens_from_source("x = 1")
        self.assertGreater(len(tokens), 0)
        for t in tokens:
            self.assertIsInstance(t.row, int)
            self.assertIsInstance(t.col, int)
            self.assertGreater(t.row, 0)


class AlignToSubwordsTests(unittest.TestCase):
    """Tests for :func:`align_to_subwords`."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base", use_fast=True
        )

    def test_produces_input_ids_and_word_ids(self) -> None:
        code_tokens = code_tokens_from_source("def foo(x):\n    return x + 1")
        input_ids, word_ids = align_to_subwords(code_tokens, self.tokenizer)
        self.assertIsInstance(input_ids, list)
        self.assertIsInstance(word_ids, list)
        self.assertEqual(len(input_ids), len(word_ids))
        self.assertGreater(len(input_ids), 0)

    def test_special_tokens_none(self) -> None:
        """CLS and SEP positions should have word_ids=None."""
        code_tokens = code_tokens_from_source("x = 1")
        input_ids, word_ids = align_to_subwords(code_tokens, self.tokenizer)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        for pos, tid in enumerate(input_ids):
            if tid in (cls_id, sep_id):
                self.assertIsNone(word_ids[pos])

    def test_truncation(self) -> None:
        code_tokens = code_tokens_from_source("x " * 500)
        input_ids, word_ids = align_to_subwords(code_tokens, self.tokenizer, max_length=32)
        self.assertLessEqual(len(input_ids), 32)


class FirstSubwordMaskTests(unittest.TestCase):
    """Tests for :func:`first_subword_mask`."""

    def test_typical(self) -> None:
        word_ids = [None, 0, 0, 1, 2, 2, None]
        mask = first_subword_mask(word_ids)
        expected = [False, True, False, True, True, False, False]
        self.assertEqual(mask, expected)

    def test_all_special(self) -> None:
        mask = first_subword_mask([None, None, None])
        self.assertEqual(mask, [False, False, False])

    def test_single_word(self) -> None:
        mask = first_subword_mask([None, 0, 0, 0, None])
        self.assertEqual(mask, [False, True, False, False, False])


class NameTagFromEditTests(unittest.TestCase):
    """Tests for :func:`name_tag_from_edit`."""

    def test_char_edit_tag(self) -> None:
        self.assertEqual(name_tag_from_edit("nubmer", "number"), "$SWAP_2")

    def test_fallback_replace_tag(self) -> None:
        """Multi-char edit → $REPLACE_* tag."""
        self.assertEqual(name_tag_from_edit("nbr", "number"), "$REPLACE_number")


if __name__ == "__main__":
    unittest.main()
