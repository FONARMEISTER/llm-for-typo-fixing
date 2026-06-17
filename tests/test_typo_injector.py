"""Tests for the typo injector.

Run with::

    uv run python -m unittest discover -s tests
"""

from __future__ import annotations

import ast
import random
import unittest

from src.typo_injector import (
    _corrupt_comments_in_source,
    _extract_renameable_identifiers,
    inject_typos,
    make_typo,
)


SAMPLE = """def factorial(number):
    result = 1
    for value in range(2, number + 1):
        result = result * value
    return result
"""

SAMPLE_WITH_COMMENT = """def factorial(number):
    # Calculate the factorial of a number
    result = 1
    for value in range(2, number + 1):
        result = result * value  # multiply and accumulate
    return result
"""

SAMPLE_NESTED = """def outer():
    result = 1
    def inner():
        result = 2
        return result
    return result
"""

SAMPLE_ATTR = """class Calculator:
    def compute(self, x, y):
        total = x + y
        return total

obj = Calculator()
obj.compute(1, 2)
"""


class ExtractIdentifiersTests(unittest.TestCase):
    """Test Jedi-based identifier extraction."""

    def test_picks_up_user_names(self):
        ids = _extract_renameable_identifiers(SAMPLE)
        self.assertIn("factorial", ids)
        self.assertIn("number", ids)
        self.assertIn("result", ids)
        self.assertIn("value", ids)
        # Each name maps to a list of definition positions.
        self.assertGreaterEqual(len(ids["result"]), 1)
        self.assertGreaterEqual(len(ids["number"]), 1)

    def test_skips_builtins_and_keywords(self):
        ids = _extract_renameable_identifiers(SAMPLE)
        self.assertNotIn("range", ids)
        self.assertNotIn("for", ids)
        self.assertNotIn("return", ids)
        self.assertNotIn("def", ids)

    def test_skips_protected_names(self):
        ids = _extract_renameable_identifiers("class Foo:\n    def __init__(self):\n        pass\n")
        self.assertNotIn("self", ids)
        self.assertNotIn("__init__", ids)
        self.assertIn("Foo", ids)

    def test_skips_short_names(self):
        ids = _extract_renameable_identifiers("x = 1\ny = 2\n")
        self.assertNotIn("x", ids)
        self.assertNotIn("y", ids)

    def test_includes_attribute_owners(self):
        ids = _extract_renameable_identifiers(SAMPLE_ATTR)
        self.assertIn("Calculator", ids)
        self.assertIn("compute", ids)
        self.assertIn("obj", ids)


class InjectTyposTests(unittest.TestCase):
    """Integration tests for :func:`inject_typos`."""

    def test_changes_source_and_records_edits(self):
        rng = random.Random(123)
        result = inject_typos(
            SAMPLE, rng=rng, max_edits=2, p_edit=1.0, corrupt_comments=False,
        )
        self.assertTrue(result.has_errors)
        self.assertNotEqual(result.corrupted, result.original)
        self.assertGreaterEqual(len(result.edits), 1)
        for edit in result.edits:
            self.assertIn(edit.original_name, SAMPLE)
            self.assertNotEqual(edit.original_name, edit.corrupted_name)

    def test_empty_when_no_candidates(self):
        result = inject_typos("x = 1\n", rng=random.Random(0))
        self.assertFalse(result.has_errors)
        self.assertEqual(result.corrupted, result.original)

    def test_corrupted_code_is_syntactically_valid(self):
        """Every corrupted snippet must parse as valid Python."""
        rng = random.Random(42)
        for _ in range(20):
            result = inject_typos(
                SAMPLE, rng=rng, max_edits=3, p_edit=0.9, corrupt_comments=False,
            )
            if not result.has_errors:
                continue
            try:
                ast.parse(result.corrupted)
            except SyntaxError as exc:
                self.fail(
                    f"Corrupted code has SyntaxError: {exc}\n"
                    f"Original: {result.original}\n"
                    f"Corrupted: {result.corrupted}\n"
                    f"Edits: {result.edits}"
                )

    def test_nested_scopes_independent(self):
        """Renaming outer 'result' must not affect inner 'result'."""
        # Try several seeds to find a case where a multi-occurrence name is renamed.
        for seed in range(200):
            rng = random.Random(seed)
            result = inject_typos(
                SAMPLE_NESTED,
                rng=rng,
                max_edits=1,
                p_edit=1.0,
                corrupt_comments=False,
            )
            if not result.has_errors:
                continue
            corrupted_name = result.edits[0].corrupted_name
            original_name = result.edits[0].original_name
            original_count = SAMPLE_NESTED.count(original_name)
            if original_count < 2:
                # Skip names that appear only once (can't test independence).
                continue
            corrupted_count = result.corrupted.count(corrupted_name)
            self.assertLess(
                corrupted_count, original_count,
                f"All {original_count} occurrences of '{original_name}' were renamed, "
                f"but they are in different scopes and should be independent.",
            )
            return  # test satisfied
        self.skipTest("No multi-occurrence name was chosen for renaming")

    def test_attribute_methods_are_touched(self):
        """Names after ``.`` should be found and corruptable."""
        rng = random.Random(7)
        result = inject_typos(
            SAMPLE_ATTR,
            rng=rng,
            max_edits=2,
            p_edit=1.0,
            corrupt_comments=False,
        )
        if not result.has_errors:
            self.skipTest("No edit produced — random chance")
        # The corrupted name should appear somewhere in the source.
        for edit in result.edits:
            self.assertIn(edit.corrupted_name, result.corrupted)


class CommentCorruptionTests(unittest.TestCase):
    """Tests for comment corruption."""

    def test_comments_are_corrupted(self):
        rng = random.Random(42)
        result = inject_typos(
            SAMPLE_WITH_COMMENT,
            rng=rng,
            max_edits=0,
            p_edit=0.0,
            corrupt_comments=True,
            p_comment_word=1.0,
        )
        self.assertTrue(result.has_errors)
        self.assertTrue(
            any(word not in result.original for word in result.corrupted.split()),
            "Comment should contain corrupted words",
        )

    def test_comments_disabled(self):
        rng = random.Random(42)
        result = inject_typos(
            SAMPLE_WITH_COMMENT,
            rng=rng,
            max_edits=0,
            p_edit=0.0,
            corrupt_comments=False,
        )
        self.assertFalse(result.has_errors)
        self.assertEqual(result.corrupted, result.original)

    def test_comment_corrupted_code_valid(self):
        rng = random.Random(123)
        for _ in range(10):
            result = inject_typos(
                SAMPLE_WITH_COMMENT,
                rng=rng,
                max_edits=2,
                p_edit=1.0,
                corrupt_comments=True,
                p_comment_word=1.0,
            )
            if not result.has_errors:
                continue
            try:
                ast.parse(result.corrupted)
            except SyntaxError as exc:
                self.fail(f"Corrupted code with comments has SyntaxError: {exc}")

    def test_code_not_in_comments(self):
        """Code patterns inside comments should not be corrupted."""
        src = "x = 1  # TODO: fix this later\n"
        rng = random.Random(42)
        result = inject_typos(
            src,
            rng=rng,
            max_edits=0,
            p_edit=0.0,
            corrupt_comments=True,
            p_comment_word=1.0,
        )
        # 'TODO' must stay intact.
        self.assertIn("TODO", result.corrupted)


class MakeTypoTests(unittest.TestCase):
    def test_produces_valid_distinct_identifier(self):
        rng = random.Random(0)
        for name in ["factorial", "my_variable", "MyClassName", "counter"]:
            typo = make_typo(name, rng)
            self.assertIsNotNone(typo, f"no typo for {name}")
            self.assertNotEqual(typo, name)
            self.assertTrue(typo.isidentifier())

    def test_preserves_underscore_structure(self):
        """A typo must not change the underscore layout."""
        rng = random.Random(0)
        for name in ["test_list", "my_var_name", "snake_case_id"]:
            for _ in range(100):
                typo = make_typo(name, rng)
                self.assertIsNotNone(typo)
                self.assertEqual(
                    typo.count("_"), name.count("_"),
                    f"underscore count changed: {name!r} -> {typo!r}",
                )
                self.assertNotIn(
                    "__", typo,
                    f"introduced double underscore: {name!r} -> {typo!r}",
                )
                original_parts = name.split("_")
                typo_parts = typo.split("_")
                self.assertEqual(
                    len(typo_parts), len(original_parts),
                    f"underscore segmentation changed: {name!r} -> {typo!r}",
                )
                for part in typo_parts:
                    self.assertTrue(
                        part, f"empty segment between underscores: {typo!r}",
                    )


if __name__ == "__main__":
    unittest.main()
