"""Evaluation harness for identifier name-fixing models.

Reads a JSONL dataset, runs a :class:`NameFixer` on each corrupted sample,
applies suggested fixes via Jedi's scope-aware refactoring, and reports
metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from Levenshtein import distance as _lev_distance

from .identifier_utils import apply_jedi_rename, extract_renameable_identifiers
from .models import NameFixer


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass
class SampleResult:
    """Result of processing one dataset sample."""

    sample_index: int
    corrupted_code: str
    ground_truth_code: str
    predicted_code: str
    model_fixes: Dict[str, str]  # {corrupted_name -> fixed_name}

    # Ground-truth edit info from the dataset.
    gt_original_names: List[str]
    gt_corrupted_names: List[str]

    @property
    def exact_match(self) -> bool:
        return self.predicted_code == self.ground_truth_code


@dataclass
class EvalMetrics:
    """Aggregate evaluation metrics."""

    total_samples: int
    exact_match_rate: float
    identifier_precision: float
    identifier_recall: float
    identifier_f1: float
    avg_normalized_edit_distance: float


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def _iter_jsonl(path: str) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalized_edit_distance(pred: str, gt: str) -> float:
    """Levenshtein distance divided by max length (in [0, 1])."""
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 0.0
    return _lev_distance(pred, gt) / max_len


def _process_sample(sample: dict, index: int, model: NameFixer) -> SampleResult:
    """Run one sample through the model pipeline.

    1. Extract identifiers from corrupted code.
    2. Ask model for fixes.
    3. Apply each fix via Jedi rename (re-extracting positions after each).
    4. Return the result.
    """
    corrupted = sample["code"]
    ground_truth = sample["fixed"]

    # Extract identifiers from corrupted code.
    identifiers = extract_renameable_identifiers(corrupted)
    all_names = list(identifiers.keys())

    # Get model suggestions.
    fixes = model.fix_names(corrupted, all_names)

    # Apply fixes sequentially, re-extracting positions each time because
    # earlier renames may shift character offsets in subsequent lines.
    predicted = corrupted
    for corrupted_name, fixed_name in fixes.items():
        if corrupted_name == fixed_name:
            continue
        current_ids = extract_renameable_identifiers(predicted)
        if corrupted_name not in current_ids:
            continue
        # Apply rename at each definition position of this name.
        for line, col in current_ids[corrupted_name]:
            changed = apply_jedi_rename(predicted, line, col, fixed_name)
            if changed:
                # Take the primary file's new code (first value in the dict).
                predicted = next(iter(changed.values()))

    # Ground-truth edit info from the dataset record.
    gt_edits = sample.get("edits", [])
    gt_original_names = [e["original_name"] for e in gt_edits]
    gt_corrupted_names = [e["corrupted_name"] for e in gt_edits]

    return SampleResult(
        sample_index=index,
        corrupted_code=corrupted,
        ground_truth_code=ground_truth,
        predicted_code=predicted,
        model_fixes=fixes,
        gt_original_names=gt_original_names,
        gt_corrupted_names=gt_corrupted_names,
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_metrics(results: List[SampleResult]) -> EvalMetrics:
    """Aggregate per-sample results into :class:`EvalMetrics`."""
    total = len(results)
    if total == 0:
        return EvalMetrics(
            total_samples=0,
            exact_match_rate=0.0,
            identifier_precision=0.0,
            identifier_recall=0.0,
            identifier_f1=0.0,
            avg_normalized_edit_distance=0.0,
        )

    exact_matches = sum(1 for r in results if r.exact_match)

    # Per-identifier true/false positives/negatives.
    tp_sum, fp_sum, fn_sum = 0, 0, 0
    for r in results:
        tp, fp, fn = _per_sample_identifier_counts(r)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_edit = (
        sum(_normalized_edit_distance(r.predicted_code, r.ground_truth_code) for r in results)
        / total
    )

    return EvalMetrics(
        total_samples=total,
        exact_match_rate=exact_matches / total,
        identifier_precision=precision,
        identifier_recall=recall,
        identifier_f1=f1,
        avg_normalized_edit_distance=avg_edit,
    )


def _per_sample_identifier_counts(result: SampleResult) -> Tuple[int, int, int]:
    """Count TP / FP / FN for one sample at the identifier level.

    - TP: model correctly fixed a corrupted identifier (suggested fix
      matches ground-truth original name).
    - FP: model suggested a fix to an identifier that was NOT corrupted
      in ground truth, or fixed to the wrong target.
    - FN: identifier was corrupted in ground truth but model did not fix it.
    """
    gt_corrupted_set = set(result.gt_corrupted_names)
    # Build a mapping: corrupted_name -> original_name from ground truth.
    gt_mapping: Dict[str, str] = dict(zip(result.gt_corrupted_names, result.gt_original_names))

    model_fixes = result.model_fixes  # {corrupted -> fixed}

    tp = 0
    fp = 0
    fn = 0

    # For identifiers the model changed:
    for corrupted_name, fixed_name in model_fixes.items():
        if corrupted_name in gt_corrupted_set:
            if fixed_name == gt_mapping.get(corrupted_name):
                tp += 1
            else:
                fp += 1  # fixed to wrong name.
        else:
            fp += 1  # model changed an identifier that wasn't corrupted.

    # For identifiers the model left alone but that SHOULD have been fixed:
    for corr_name in gt_corrupted_set:
        if corr_name not in model_fixes:
            fn += 1

    return tp, fp, fn


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def evaluate(
    model: NameFixer,
    dataset_path: str,
    *,
    max_samples: Optional[int] = None,
) -> Tuple[List[SampleResult], EvalMetrics]:
    """Run the full evaluation pipeline.

    Args:
        model: A :class:`NameFixer` instance.
        dataset_path: Path to a JSONL dataset.
        max_samples: Limit to first N error samples (useful for fast tests).

    Returns:
        ``(per_sample_results, aggregate_metrics)``.
    """
    all_samples = list(_iter_jsonl(dataset_path))
    results: List[SampleResult] = []

    for i, sample in enumerate(all_samples):
        if not sample.get("has_errors", False):
            continue
        results.append(_process_sample(sample, i, model))
        if max_samples is not None and len(results) >= max_samples:
            break

    metrics = compute_metrics(results)
    return results, metrics
