"""Smoke tests for the typo injector.

Run with::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import random
import unittest

from src.typo_injector import (
    _extract_renameable_identifiers,
    _replace_identifier,
    inject_typos,
    make_typo,
)


SAMPLE = """def factorial(number):
    result = 1
    for value in range(2, number + 1):
        result = result * value
    return result
"""


class ExtractIdentifiersTests(unittest.TestCase):
    def test_picks_up_user_names(self):
        ids = _extract_renameable_identifiers(SAMPLE)
        self.assertIn("factorial", ids)
        self.assertIn("number", ids)
        self.assertIn("result", ids)
        self.assertIn("value", ids)
        # `number` is used 2x: parameter + range bound
        self.assertEqual(ids["number"], 2)

    def test_skips_builtins_and_keywords(self):
        ids = _extract_renameable_identifiers(SAMPLE)
        self.assertNotIn("range", ids)
        self.assertNotIn("for", ids)
        self.assertNotIn("return", ids)
        self.assertNotIn("def", ids)


class ReplaceIdentifierTests(unittest.TestCase):
    def test_replaces_all_occurrences(self):
        out = _replace_identifier(SAMPLE, "result", "rezult")
        self.assertNotIn("result", out)
        self.assertEqual(out.count("rezult"), SAMPLE.count("result"))

    def test_does_not_touch_strings(self):
        source = 'def greet(name):\n    print("name is great")\n    return name\n'
        out = _replace_identifier(source, "name", "nme")
        self.assertIn('"name is great"', out)
        self.assertIn("nme", out)


class MakeTypoTests(unittest.TestCase):
    def test_produces_valid_distinct_identifier(self):
        rng = random.Random(0)
        for name in ["factorial", "my_variable", "MyClassName", "counter"]:
            typo = make_typo(name, rng)
            self.assertIsNotNone(typo, f"no typo for {name}")
            self.assertNotEqual(typo, name)
            self.assertTrue(typo.isidentifier())

    def test_preserves_underscore_structure(self):
        """A typo must not change the underscore layout — that reads as a naming
        style change (e.g. ``test_list -> testlist``), not a typo.

        Concretely we require: same number of underscores, no new runs of
        consecutive underscores, and the split-by-``_`` parts in the typo line up
        one-to-one with the original (same count, every part non-empty)."""
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


class InjectTyposTests(unittest.TestCase):
    def test_changes_source_and_records_edits(self):
        rng = random.Random(123)
        result = inject_typos(SAMPLE, rng=rng, max_edits=2, p_edit=1.0)
        self.assertTrue(result.has_errors)
        self.assertNotEqual(result.corrupted, result.original)
        self.assertGreaterEqual(len(result.edits), 1)
        for edit in result.edits:
            self.assertIn(edit.original_name, SAMPLE)
            self.assertIn(edit.corrupted_name, result.corrupted)
            self.assertNotIn(edit.corrupted_name, SAMPLE)

    def test_empty_when_no_candidates(self):
        result = inject_typos("x = 1\n", rng=random.Random(0))
        self.assertFalse(result.has_errors)
        self.assertEqual(result.corrupted, result.original)


if __name__ == "__main__":
    unittest.main()
