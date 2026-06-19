"""Tests for the evaluation harness."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

from Levenshtein import distance as _lev_distance

from src.harness import (
    SampleResult,
    _iter_jsonl,
    _normalized_edit_distance,
    _per_sample_identifier_counts,
    compute_metrics,
    evaluate,
)
from src.models.base import NameFixer
from src.typo_injector import inject_typos


# --------------------------------------------------------------------------- #
# Toy models for testing
# --------------------------------------------------------------------------- #


class PerfectFixer(NameFixer):
    """Returns the exact ground-truth mapping, provided at construction time."""

    def __init__(self, mapping: Dict[str, str]) -> None:
        self._mapping = mapping

    def fix_names(self, code: str, names: List[str]) -> Dict[str, str]:
        return {
            name: self._mapping[name]
            for name in names
            if name in self._mapping
        }


class NoopFixer(NameFixer):
    """Returns no fixes — leaves everything as-is."""

    def fix_names(self, code: str, names: List[str]) -> Dict[str, str]:
        return {}


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #


def _make_test_dataset(samples: List[Dict], path: str) -> str:
    """Write a small JSONL file and return its path."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class EditDistanceTests(unittest.TestCase):
    def test_library_distance(self):
        self.assertEqual(_lev_distance("abc", "abc"), 0)
        self.assertEqual(_lev_distance("cat", "cats"), 1)
        self.assertEqual(_lev_distance("cat", "cut"), 1)
        self.assertEqual(_lev_distance("", ""), 0)

    def test_normalized_bounds(self):
        d = _normalized_edit_distance("abc", "abcdef")
        self.assertTrue(0.0 <= d <= 1.0)

    def test_normalized_zero(self):
        self.assertEqual(_normalized_edit_distance("", ""), 0.0)


class IterJsonlTests(unittest.TestCase):
    """Unit tests for ``_iter_jsonl``."""

    def test_skips_empty_lines(self) -> None:
        """Blank lines in JSONL are skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"a": 1}\n\n{"b": 2}\n')
            items = list(_iter_jsonl(path))
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0], {"a": 1})
            self.assertEqual(items[1], {"b": 2})

    def test_file_with_only_empty_lines(self) -> None:
        """File with only blank lines yields zero items."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "empty.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n\n\n")
            items = list(_iter_jsonl(path))
            self.assertEqual(len(items), 0)


class IdentifierCountsTests(unittest.TestCase):
    def test_perfect_fix(self):
        result = SampleResult(
            sample_index=0,
            corrupted_code="x = 1",
            ground_truth_code="y = 1",
            predicted_code="y = 1",
            model_fixes={"x": "y"},
            gt_original_names=["y"],
            gt_corrupted_names=["x"],
        )
        tp, fp, fn = _per_sample_identifier_counts(result)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_missed_fix(self):
        result = SampleResult(
            sample_index=0,
            corrupted_code="x = 1",
            ground_truth_code="y = 1",
            predicted_code="x = 1",
            model_fixes={},
            gt_original_names=["y"],
            gt_corrupted_names=["x"],
        )
        tp, fp, fn = _per_sample_identifier_counts(result)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 1)

    def test_wrong_fix(self):
        result = SampleResult(
            sample_index=0,
            corrupted_code="x = 1",
            ground_truth_code="y = 1",
            predicted_code="z = 1",
            model_fixes={"x": "z"},
            gt_original_names=["y"],
            gt_corrupted_names=["x"],
        )
        tp, fp, fn = _per_sample_identifier_counts(result)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

    def test_false_positive(self):
        """Model renamed an identifier that wasn't corrupted."""
        result = SampleResult(
            sample_index=0,
            corrupted_code="a = 1\nb = 2",
            ground_truth_code="aa = 1\nb = 2",
            predicted_code="aa = 1\nbb = 2",
            model_fixes={"b": "bb"},
            gt_original_names=["aa"],
            gt_corrupted_names=["a"],
        )
        tp, fp, fn = _per_sample_identifier_counts(result)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)


class MetricsTests(unittest.TestCase):
    def test_empty_no_crash(self):
        m = compute_metrics([])
        self.assertEqual(m.total_samples, 0)
        self.assertEqual(m.exact_match_rate, 0.0)

    def test_all_perfect(self):
        results = [
            SampleResult(
                sample_index=0,
                corrupted_code="x = 1",
                ground_truth_code="y = 1",
                predicted_code="y = 1",
                model_fixes={"x": "y"},
                gt_original_names=["y"],
                gt_corrupted_names=["x"],
            ),
        ]
        m = compute_metrics(results)
        self.assertEqual(m.exact_match_rate, 1.0)
        self.assertEqual(m.identifier_precision, 1.0)
        self.assertEqual(m.identifier_recall, 1.0)
        self.assertEqual(m.identifier_f1, 1.0)

    def test_mixed(self):
        results = [
            SampleResult(
                sample_index=0,
                corrupted_code="a = 1",
                ground_truth_code="aa = 1",
                predicted_code="aa = 1",
                model_fixes={"a": "aa"},
                gt_original_names=["aa"],
                gt_corrupted_names=["a"],
            ),
            SampleResult(
                sample_index=1,
                corrupted_code="b = 2",
                ground_truth_code="bb = 2",
                predicted_code="b = 2",
                model_fixes={},
                gt_original_names=["bb"],
                gt_corrupted_names=["b"],
            ),
        ]
        m = compute_metrics(results)
        self.assertEqual(m.exact_match_rate, 0.5)
        self.assertAlmostEqual(m.identifier_precision, 1.0)
        self.assertAlmostEqual(m.identifier_recall, 0.5)
        self.assertAlmostEqual(m.identifier_f1, 2 / 3, places=3)


# --------------------------------------------------------------------------- #
# End-to-end pipeline tests
# --------------------------------------------------------------------------- #


SAMPLE_FACTORIAL = """\
def factorial(n):
    r = 1
    for i in range(2, n + 1):
        r = r * i
    return r
"""

SAMPLE_IS_PRIME = """\
def is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True
"""

SAMPLE_ADD = """\
def add(a, b):
    return a + b
"""

SAMPLE_LONG = """\
def factorial(number):
    result = 1
    for value in range(2, number + 1):
        result = result * value
    return result
"""


class FullPipelineTests(unittest.TestCase):
    """End-to-end tests with real datasets and the injector."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def _dataset_path(self) -> str:
        return str(Path(self._tmpdir) / "test.jsonl")

    def _generate_samples(self, src: str, rng: random.Random, count: int) -> List[dict]:
        samples: List[dict] = []
        for _ in range(count):
            result = inject_typos(
                src, rng=rng, max_edits=2, p_edit=1.0, corrupt_comments=False,
            )
            if result.has_errors:
                samples.append({
                    "code": result.corrupted,
                    "fixed": result.original,
                    "has_errors": True,
                    "edits": [e.to_dict() for e in result.edits],
                    "language": "python",
                })
        return samples

    def test_perfect_fixer_achieves_em(self):
        """A model that knows the correct answers should get exact match."""
        rng = random.Random(42)
        samples: List[dict] = []
        for src in (SAMPLE_FACTORIAL, SAMPLE_IS_PRIME):
            samples.extend(self._generate_samples(src, rng, count=2))

        if len(samples) == 0:
            self.skipTest("No corrupted samples generated — random chance")

        ds_path = self._dataset_path()
        _make_test_dataset(samples, ds_path)

        fixes: Dict[str, str] = {}
        for s in samples:
            for edit in s["edits"]:
                fixes[edit["corrupted_name"]] = edit["original_name"]

        model = PerfectFixer(fixes)
        results, metrics = evaluate(model, ds_path)

        self.assertGreater(len(results), 0)
        self.assertEqual(
            metrics.exact_match_rate, 1.0,
            f"PerfectFixer should get EM=1.0, got {metrics.exact_match_rate}",
        )
        self.assertAlmostEqual(metrics.identifier_f1, 1.0)

    def test_noop_fixer_gets_zero(self):
        """A model that changes nothing should get 0 recall / 0 EM."""
        rng = random.Random(42)
        samples = self._generate_samples(SAMPLE_FACTORIAL, rng, count=3)

        if len(samples) == 0:
            self.skipTest("No corrupted samples generated")

        ds_path = self._dataset_path()
        _make_test_dataset(samples, ds_path)

        model = NoopFixer()
        results, metrics = evaluate(model, ds_path)

        self.assertGreater(len(results), 0)
        self.assertEqual(metrics.exact_match_rate, 0.0)
        self.assertEqual(metrics.identifier_recall, 0.0)

    def test_max_samples_limit(self):
        """:param:`max_samples` must limit the number of evaluated samples."""
        rng = random.Random(42)
        samples = self._generate_samples(SAMPLE_ADD, rng, count=5)

        if len(samples) < 2:
            self.skipTest("Not enough corrupted samples")

        ds_path = self._dataset_path()
        _make_test_dataset(samples, ds_path)

        model = NoopFixer()
        results, metrics = evaluate(model, ds_path, max_samples=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(metrics.total_samples, 2)

    def test_spellchecker_end_to_end(self):
        """Spellchecker should run without crashing on real typos."""
        rng = random.Random(0)
        samples = self._generate_samples(SAMPLE_LONG, rng, count=3)

        if len(samples) == 0:
            self.skipTest("No corrupted samples generated")

        ds_path = self._dataset_path()
        _make_test_dataset(samples, ds_path)

        from src.models.spellcheck import SpellCheckerFixer

        model = SpellCheckerFixer()
        results, metrics = evaluate(model, ds_path)

        self.assertGreaterEqual(
            metrics.identifier_recall, 0.0,
            "Spellchecker recall >= 0.0 (at least runs without crash)",
        )


if __name__ == "__main__":
    unittest.main()
