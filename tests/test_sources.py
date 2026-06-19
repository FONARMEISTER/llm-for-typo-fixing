"""Unit tests for ``src/sources.py``.

Tests ``load_demo``, ``iter_source``, and validates that all bundled
demo snippets are syntactically valid Python.
"""

from __future__ import annotations

import ast
import unittest

from src import sources


class LoadDemoTests(unittest.TestCase):
    """Tests for ``load_demo``."""

    def test_returns_all_eight_without_limit(self) -> None:
        snippets = list(sources.load_demo())
        self.assertEqual(len(snippets), 8)

    def test_max_samples_limit(self) -> None:
        snippets = list(sources.load_demo(max_samples=3))
        self.assertEqual(len(snippets), 3)

    def test_max_samples_exceeds_total(self) -> None:
        """max_samples larger than total → return all available."""
        snippets = list(sources.load_demo(max_samples=100))
        self.assertEqual(len(snippets), 8)

    def test_max_samples_zero(self) -> None:
        snippets = list(sources.load_demo(max_samples=0))
        self.assertEqual(len(snippets), 0)

    def test_all_snippets_are_valid_python(self) -> None:
        for i, snippet in enumerate(sources.load_demo()):
            with self.subTest(i=i):
                ast.parse(snippet)

    def test_all_snippets_are_non_empty(self) -> None:
        for i, snippet in enumerate(sources.load_demo()):
            with self.subTest(i=i):
                self.assertGreater(len(snippet.strip()), 0)


class IterSourceTests(unittest.TestCase):
    """Tests for ``iter_source`` lookup function."""

    def test_iter_demo_works(self) -> None:
        snippets = list(sources.iter_source("demo"))
        self.assertEqual(len(snippets), 8)

    def test_unknown_source_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            list(sources.iter_source("nonexistent_source"))
        self.assertIn("nonexistent_source", str(ctx.exception))
        self.assertIn("demo", str(ctx.exception).lower())

    def test_all_registered_sources_are_callable(self) -> None:
        """Every key in ``SOURCES`` maps to a callable."""
        for name in sources.SOURCES:
            with self.subTest(name=name):
                self.assertTrue(callable(sources.SOURCES[name]))


class KnownSplitSizesTests(unittest.TestCase):
    """Tests for ``_KNOWN_SPLIT_SIZES`` sanity checks."""

    def test_demo_has_train_only(self) -> None:
        self.assertIn("train", sources._KNOWN_SPLIT_SIZES["demo"])
        self.assertEqual(sources._KNOWN_SPLIT_SIZES["demo"]["train"], 8)

    def test_mbpp_has_three_splits(self) -> None:
        sizes = sources._KNOWN_SPLIT_SIZES["mbpp"]
        self.assertIn("train", sizes)
        self.assertIn("validation", sizes)
        self.assertIn("test", sizes)

    def test_all_sizes_are_positive(self) -> None:
        for src_name, splits in sources._KNOWN_SPLIT_SIZES.items():
            for split_name, size in splits.items():
                with self.subTest(kind=src_name, split=split_name):
                    self.assertGreater(size, 0)


if __name__ == "__main__":
    unittest.main()
