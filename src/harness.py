"""Evaluation harness for identifier name-fixing models.

Reads a JSONL dataset, runs a :class:`NameFixer` on each corrupted sample,
applies suggested fixes via libcst's scope-aware refactoring, and reports
metrics.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from Levenshtein import distance as _lev_distance
from tqdm import tqdm

from .identifier_utils import (
    UnparseableCodeError,
    apply_rename,
    extract_renameable_identifiers,
)
from .models import NameFixer, make_model


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

    # False positives on clean samples.
    clean_samples_total: int
    clean_false_positive_count: int  # number of names falsely flagged as needing a fix.
    clean_false_positive_rate: float  # FP names / total clean names.

    # Timing.
    total_time_seconds: float
    avg_time_per_sample_seconds: float
    avg_time_per_kb_seconds: float


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


def _process_sample(
    sample: dict,
    index: int,
    model: NameFixer,
) -> Optional[SampleResult]:
    """Run one sample through the model pipeline.

    1. Extract identifiers from corrupted code.
    2. Ask model for fixes.
    3. Apply all fixes in a single libcst pass (no re-extraction needed).
    4. Return the result.

    Returns ``None`` if the code cannot be parsed (Python 2 syntax, merge
    conflict markers) — such samples should be silently skipped.
    """
    corrupted = sample["code"]
    ground_truth = sample["fixed"]

    # Extract identifiers from corrupted code.
    try:
        identifiers = extract_renameable_identifiers(corrupted)
    except UnparseableCodeError:
        return None
    all_names = list(identifiers.keys())

    # Get model suggestions.
    fixes = model.fix_names(corrupted, all_names)

    # Remove no-op fixes.
    fixes = {k: v for k, v in fixes.items() if k != v}

    # Apply all fixes at once — libcst handles scope awareness internally.
    try:
        predicted = apply_rename(corrupted, fixes) if fixes else corrupted
    except UnparseableCodeError:
        return None

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


def _check_clean_sample(sample: dict, model: NameFixer) -> Tuple[int, int]:
    """Check whether the model incorrectly proposes fixes on a clean sample.

    Returns ``(false_positive_count, total_names)`` — the number of
    identifiers the model falsely flagged for editing, and the total
    number of identifiers checked.
    Returns ``(0, 0)`` if the code cannot be parsed or has no identifiers.
    """
    try:
        identifiers = extract_renameable_identifiers(sample["code"])
    except UnparseableCodeError:
        return 0, 0
    names = list(identifiers.keys())
    if not names:
        return 0, 0
    fixes = model.fix_names(sample["code"], names)
    fp = sum(1 for k, v in fixes.items() if k != v)
    return fp, len(names)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_metrics(
    results: List[SampleResult],
    clean_fp_count: int = 0,
    clean_names_total: int = 0,
    clean_samples_total: int = 0,
) -> EvalMetrics:
    """Aggregate per-sample results into :class:`EvalMetrics`.

    Timing fields are set to zero — callers should fill them in.
    """
    total = len(results)
    if total == 0:
        return EvalMetrics(
            total_samples=0,
            exact_match_rate=0.0,
            identifier_precision=0.0,
            identifier_recall=0.0,
            identifier_f1=0.0,
            avg_normalized_edit_distance=0.0,
            clean_samples_total=clean_samples_total,
            clean_false_positive_count=clean_fp_count,
            clean_false_positive_rate=(
                clean_fp_count / clean_names_total if clean_names_total else 0.0
            ),
            total_time_seconds=0.0,
            avg_time_per_sample_seconds=0.0,
            avg_time_per_kb_seconds=0.0,
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
        clean_samples_total=clean_samples_total,
        clean_false_positive_count=clean_fp_count,
        clean_false_positive_rate=(
            clean_fp_count / clean_names_total if clean_names_total else 0.0
        ),
        total_time_seconds=0.0,
        avg_time_per_sample_seconds=0.0,
        avg_time_per_kb_seconds=0.0,
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


def _eval_worker(payload: Tuple[dict, int, str, dict]) -> Optional[SampleResult]:
    """Process one sample in a child process — free function for pickling.

    libcst is thread-safe and process-safe by design — each process has an
    independent memory space, and libcst has no shared file cache.

    Returns ``None`` if the sample cannot be parsed.
    """
    sample, index, model_name, model_kwargs = payload

    model = make_model(model_name, **model_kwargs)
    return _process_sample(sample, index, model)


# --------------------------------------------------------------------------- #
# Top-level entry points
# --------------------------------------------------------------------------- #


def evaluate(
    model: NameFixer,
    dataset_path: str,
    *,
    max_samples: Optional[int] = None,
    workers: int = 1,
) -> Tuple[List[SampleResult], EvalMetrics]:
    """Run the full evaluation pipeline, optionally in parallel.

    Args:
        model: A :class:`NameFixer` instance (used only for ``workers=1``).
        dataset_path: Path to a JSONL dataset.
        max_samples: Limit to first N error samples (useful for fast tests).
        workers: Number of worker processes.  ``1`` uses the calling process
            (no pickling overhead).  ``0`` auto-detects CPU count.

    Returns:
        ``(per_sample_results, aggregate_metrics)``.
    """
    # Load and filter samples.
    all_samples = list(_iter_jsonl(dataset_path))
    error_samples = [
        (s, i) for i, s in enumerate(all_samples) if s.get("has_errors", False)
    ]
    clean_samples = [
        s for s in all_samples if not s.get("has_errors", False)
    ]
    if max_samples is not None:
        error_samples = error_samples[:max_samples]

    # Auto-detect worker count.
    n_workers = workers if workers > 0 else os.cpu_count() or 1

    # If the model supports parallel API requests, use a thread pool to
    # overlap I/O-bound HTTP calls.  Local models get max_parallel_requests=1
    # (serial) — cloud APIs benefit from 10–20 concurrent requests.
    max_parallel = getattr(model, "max_parallel_requests", 1)

    t_start = time.perf_counter()

    # --- Error samples ---
    if max_parallel > 1:
        results = _evaluate_threaded(error_samples, model, max_parallel)
    elif n_workers <= 1:
        results: List[SampleResult] = []
        for sample, idx in tqdm(error_samples, desc="Evaluating", unit="sample"):
            sr = _process_sample(sample, idx, model)
            if sr is not None:
                results.append(sr)
    else:
        model_name = model.name
        model_kwargs = getattr(model, "_init_kwargs", {})
        results = _evaluate_parallel(error_samples, model_name, model_kwargs, n_workers)

    # --- Clean samples (false-positive detection) ---
    clean_fp_count = 0
    clean_names_total = 0
    if clean_samples and max_parallel > 1:
        # Thread-pool path — parallelise clean checks.
        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futures = [
                ex.submit(_check_clean_sample, s, model) for s in clean_samples
            ]
            for future in as_completed(futures):
                fp, nt = future.result()
                clean_fp_count += fp
                clean_names_total += nt
    elif clean_samples:
        for sample in tqdm(clean_samples, desc="Clean check", unit="sample"):
            fp, nt = _check_clean_sample(sample, model)
            clean_fp_count += fp
            clean_names_total += nt

    elapsed = time.perf_counter() - t_start

    metrics = compute_metrics(
        results,
        clean_fp_count=clean_fp_count,
        clean_names_total=clean_names_total,
        clean_samples_total=len(clean_samples),
    )

    # Fill in timing.
    n = len(results)
    total_kb = sum(len(r.corrupted_code.encode("utf-8")) for r in results) / 1024.0
    metrics.total_time_seconds = elapsed
    metrics.avg_time_per_sample_seconds = elapsed / n if n else 0.0
    metrics.avg_time_per_kb_seconds = elapsed / total_kb if total_kb else 0.0

    return results, metrics


def _evaluate_threaded(
    error_samples: List[Tuple[dict, int]],
    model: NameFixer,
    max_workers: int,
) -> List[SampleResult]:
    """Evaluate samples using :class:`ThreadPoolExecutor` for concurrent
    I/O-bound LLM API calls.

    libcst is thread-safe by design — no shared mutable state, no file cache."""
    return _run_threaded(error_samples, model, max_workers)


def _run_threaded(
    error_samples: List[Tuple[dict, int]],
    model: NameFixer,
    max_workers: int,
) -> List[SampleResult]:
    """Core of ``_evaluate_threaded`` — extracted so save/restore logic
    is separate from the executor machinery."""
    results: List[SampleResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_process_sample, s, i, model): (i, s)
            for s, i in error_samples
        }
        desc = f"Evaluating ({max_workers} threads)"
        with tqdm(total=len(futures), desc=desc, unit="sample") as pbar:
            for future in as_completed(futures):
                idx, sample = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    tqdm.write(f"Sample {idx} failed: {exc}")
                pbar.update(1)

    # Restore original ordering.
    results.sort(key=lambda r: r.sample_index)
    return results


def _evaluate_parallel(
    error_samples: List[Tuple[dict, int]],
    model_name: str,
    model_kwargs: dict,
    n_workers: int,
) -> List[SampleResult]:
    """Evaluate samples in parallel using :class:`ProcessPoolExecutor`."""
    payloads = [(s, idx, model_name, model_kwargs) for s, idx in error_samples]
    results: List[SampleResult] = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_eval_worker, p): p for p in payloads}
        with tqdm(total=len(payloads), desc="Evaluating (parallel)", unit="sample") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    sample, idx = futures[future][:2]
                    tqdm.write(f"Worker failed for sample {idx}: {exc}")
                pbar.update(1)

    # Restore original ordering (as_completed yields in finish order).
    results.sort(key=lambda r: r.sample_index)
    return results
