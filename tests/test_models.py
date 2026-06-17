"""Tests for models (spellcheck baseline)."""

from __future__ import annotations

import unittest

from src.models.spellcheck import SpellCheckerFixer


class SpellCheckerFixerTests(unittest.TestCase):
    """Test the spellchecker baseline on common typo patterns."""

    def setUp(self) -> None:
        self.fixer = SpellCheckerFixer()

    def test_simple_typo_swap(self):
        fixes = self.fixer.fix_names("", ["nubmer", "rsult", "value"])
        self.assertIn("nubmer", fixes)
        self.assertEqual(fixes["nubmer"], "number")
        self.assertIn("rsult", fixes)
        self.assertEqual(fixes["rsult"], "result")
        # 'value' is correctly spelled — should not be changed.
        self.assertNotIn("value", fixes)

    def test_snake_case_typo(self):
        fixes = self.fixer.fix_names("", ["my_varaible", "test_case"])
        self.assertIn("my_varaible", fixes)
        self.assertEqual(fixes["my_varaible"], "my_variable")

    def test_camel_case_typo(self):
        fixes = self.fixer.fix_names("", ["MyClasz", "calculate_total"])
        self.assertIn("MyClasz", fixes)
        self.assertEqual(fixes["MyClasz"], "MyClass")

    def test_no_fixes_on_correct(self):
        fixes = self.fixer.fix_names("", ["variable", "function_name", "MyClass"])
        self.assertEqual(fixes, {})

    def test_single_char_not_fixed(self):
        """Names shorter than 3 chars wouldn't reach the fixer anyway,
        but if they did, spellchecker should handle them gracefully."""
        # The harness filters these out before calling fix_names, but
        # we still test that a single-char name doesn't crash.
        fixes = self.fixer.fix_names("", ["a", "ab"])
        self.assertEqual(fixes, {})

    def test_code_argument_ignored(self):
        """The spellchecker ignores the code argument — only uses names."""
        code = "x = 1\n"
        fixes = self.fixer.fix_names(code, ["nubmer"])
        self.assertIn("nubmer", fixes)
        self.assertEqual(fixes["nubmer"], "number")


if __name__ == "__main__":
    unittest.main()
