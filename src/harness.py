"""Evaluation harness for identifier name-fixing models.

Reads a JSONL dataset, runs a :class:`NameFixer` on each corrupted sample,
applies suggested fixes via Jedi's scope-aware refactoring, and reports
metrics.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from Levenshtein import distance as _lev_distance
from tqdm import tqdm

from .identifier_utils import apply_jedi_rename, extract_renameable_identifiers
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

    # Suppress SyntaxWarning from parso about invalid escape sequences in
    # real-world code.
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    # Extract identifiers from corrupted code.
    identifiers = extract_renameable_identifiers(corrupted)
    all_names = list(identifiers.keys())

    # Get model suggestions.
    fixes = model.fix_names(corrupted, all_names)

    # Apply fixes sequentially, re-extracting after every rename because
    # character offsets shift.  Each identifier may be defined in multiple
    # scopes — we rename them one scope at a time, re-extracting after each.
    predicted = corrupted
    for corrupted_name, fixed_name in fixes.items():
        if corrupted_name == fixed_name:
            continue
        # Keep re-extracting and renaming until no more definitions of this
        # name remain, or until we've tried every available position without
        # success (e.g., Jedi rejects all of them).
        tried = 0
        while True:
            current_ids = extract_renameable_identifiers(predicted)
            positions = current_ids.get(corrupted_name, [])
            if not positions:
                break
            if tried >= len(positions):
                break  # exhausted all positions without success.
            line, col = positions[tried]
            try:
                changed = apply_jedi_rename(predicted, line, col, fixed_name)
            except Exception:
                # Jedi may choke on edge cases (unresolvable type, stale
                # internal state, etc.).  Skip this position.
                tried += 1
                continue
            if changed:
                predicted = next(iter(changed.values()))
                tried = 0  # reset — code changed, positions need re-extraction.
                continue
            tried += 1  # rename returned empty result, try next position.

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


# Track whether this worker process has already initialised its own Jedi cache
# and the path for cleanup on exit.
_worker_cache_initialised = False
_worker_cache_dir: Optional[str] = None


def _eval_worker(payload: Tuple[dict, int, str, dict]) -> SampleResult:
    """Process one sample in a child process — free function for pickling.

    On Linux the default ``fork`` start method means child processes inherit
    the parent's ``jedi.settings.cache_directory``.  All workers would share
    the same path, corrupting each other's pickle files ("Ran out of input",
    "invalid load key").  To prevent this, each worker re-initialises the
    Jedi cache to a unique temp directory on its first call.
    """
    global _worker_cache_initialised, _worker_cache_dir
    if not _worker_cache_initialised:
        import atexit
        import shutil
        import tempfile
        import pathlib
        import jedi.settings
        _worker_cache_dir = tempfile.mkdtemp(prefix="jedi_eval_worker_")
        jedi.settings.cache_directory = pathlib.Path(_worker_cache_dir)
        atexit.register(shutil.rmtree, _worker_cache_dir, ignore_errors=True)
        _worker_cache_initialised = True

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
    if max_samples is not None:
        error_samples = error_samples[:max_samples]

    # Auto-detect worker count.
    n_workers = workers if workers > 0 else os.cpu_count() or 1

    # If the model supports parallel API requests, use a thread pool to
    # overlap I/O-bound HTTP calls.  Local models get max_parallel_requests=1
    # (serial) — cloud APIs benefit from 10–20 concurrent requests.
    max_parallel = getattr(model, "max_parallel_requests", 1)

    if max_parallel > 1:
        # Thread-pool path — concurrent HTTP calls, Jedi ops serialised by GIL.
        results = _evaluate_threaded(error_samples, model, max_parallel)
    elif n_workers <= 1:
        # Serial path — no pickling overhead, simpler debugging.
        results: List[SampleResult] = []
        for sample, idx in tqdm(error_samples, desc="Evaluating", unit="sample"):
            results.append(_process_sample(sample, idx, model))
    else:
        # Parallel path — each worker creates its own model copy.
        model_name = model.name  # type: ignore[attr-defined]
        # Collect any extra constructor kwargs the model needs to be rebuilt
        # in worker processes (e.g. GECToRFixer needs model_dir).
        model_kwargs = getattr(model, "_init_kwargs", {})
        results = _evaluate_parallel(error_samples, model_name, model_kwargs, n_workers)

    metrics = compute_metrics(results)
    return results, metrics


def _evaluate_threaded(
    error_samples: List[Tuple[dict, int]],
    model: NameFixer,
    max_workers: int,
) -> List[SampleResult]:
    """Evaluate samples using :class:`ThreadPoolExecutor` for concurrent
    I/O-bound LLM API calls.

    Jedi is **not thread-safe** by default: ``fast_parser`` reuses mutable
    module objects across calls, and the parso file cache is shared across
    threads (leading to corrupted pickle files).  Both must be disabled
    for the duration of the threaded evaluation."""
    import jedi.settings

    # Save and disable Jedi's thread-unsafe features.
    _prev_cache = jedi.settings.cache_directory
    _prev_fast = jedi.settings.fast_parser
    jedi.settings.cache_directory = None
    jedi.settings.fast_parser = False

    try:
        results = _run_threaded(error_samples, model, max_workers)
    finally:
        jedi.settings.cache_directory = _prev_cache
        jedi.settings.fast_parser = _prev_fast

    return results


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
                    results.append(future.result())
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
                    results.append(future.result())
                except Exception as exc:
                    sample, idx = futures[future][:2]
                    tqdm.write(f"Worker failed for sample {idx}: {exc}")
                pbar.update(1)

    # Restore original ordering (as_completed yields in finish order).
    results.sort(key=lambda r: r.sample_index)
    return results
