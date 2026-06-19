#!/usr/bin/env -S uv run python
"""Run evaluation benchmarks across models and datasets, caching results to disk.

Usage::

    uv run src/benchmark.py                    # run all missing benchmarks
    uv run src/benchmark.py --force            # re-run everything
    uv run src/benchmark.py --save-samples     # also save per-sample results
    uv run src/benchmark.py --max-samples 50   # limit to 50 samples for quick test

Datasets and models are defined explicitly in this script (see ``DATASETS``
and ``MODELS`` below).  Results are written to ``benchmarks/`` as JSON files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Ensure src/ is importable even when run from project root.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.harness import evaluate, EvalMetrics, SampleResult  # noqa: E402
from src.models import make_model  # noqa: E402

# ------------------------------------------------------------------ #
# Datasets — only evaluation-only datasets (not used for training)
# ------------------------------------------------------------------ #

DATASET_PATHS = {
    # "demo": str(_project_root / "data" / "demo.jsonl"),
    # "github_python": str(_project_root / "data" / "github_python" / "test.jsonl"),
    "quicktest": str(_project_root / "data" / "quicktest.jsonl"),
}

# ------------------------------------------------------------------ #
# Models — explicit list with any extra constructor kwargs
# ------------------------------------------------------------------ #

GECTOR_MODEL_DIR = str(_project_root / "models" / "gector-codebert" / "best")

MODELS: List[Dict[str, Any]] = [
    {
        "key": "spellcheck",
        "display": "SpellCheck",
        "kwargs": {},
    },
    {
        "key": "typos",
        "display": "Typos",
        "kwargs": {},
    },
    {
        "key": "gector",
        "display": "GECToR (CodeBERT)",
        "kwargs": {"model_dir": GECTOR_MODEL_DIR},
    },
    {
        "key": "llm_api",
        "display": "Llama 3.2 1B",
        "kwargs": {"preset": "llama-3.2-1b"},
    },
    {
        "key": "llm_api",
        "display": "Qwen 3.5 0.8B",
        "kwargs": {"preset": "qwen-3.5-0.8b"},
    },
    {
        "key": "llm_api",
        "display": "Gemma 4 E2B",
        "kwargs": {"preset": "gemma4-e2b"},
    },
    {
        "key": "llm_api",
        "display": "Qwen 3.5 9B",
        "kwargs": {"preset": "qwen3.5-9B"},
    },
    {
        "key": "llm_api",
        "display": "WordStorm",
        "kwargs": {"preset": "wordstorm"},
    },
]

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

OUTPUT_DIR = _project_root / "benchmarks"


def _slug(model_entry: Dict[str, Any]) -> str:
    """Short filesystem-safe key for a model."""
    key = model_entry["key"]
    if key == "llm_api":
        preset = model_entry["kwargs"].get("preset", "unknown")
        return f"llm_{preset}"
    if key == "gector":
        return "gector"
    return key


def _metrics_to_dict(m: EvalMetrics) -> Dict[str, Any]:
    return {
        "total_samples": m.total_samples,
        "exact_match_rate": m.exact_match_rate,
        "identifier_precision": m.identifier_precision,
        "identifier_recall": m.identifier_recall,
        "identifier_f1": m.identifier_f1,
        "avg_normalized_edit_distance": m.avg_normalized_edit_distance,
        "clean_samples_total": m.clean_samples_total,
        "clean_false_positive_count": m.clean_false_positive_count,
        "clean_false_positive_rate": m.clean_false_positive_rate,
        "total_time_seconds": m.total_time_seconds,
        "avg_time_per_sample_seconds": m.avg_time_per_sample_seconds,
        "avg_time_per_kb_seconds": m.avg_time_per_kb_seconds,
    }


def _sample_to_dict(r: SampleResult) -> Dict[str, Any]:
    return {
        "index": r.sample_index,
        "exact_match": r.exact_match,
        "corrupted_code": r.corrupted_code,
        "predicted_code": r.predicted_code,
        "ground_truth_code": r.ground_truth_code,
        "model_fixes": r.model_fixes,
        "gt_original_names": r.gt_original_names,
        "gt_corrupted_names": r.gt_corrupted_names,
    }


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_project_root,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _warmup(model) -> None:
    """Warm up the model with one trivial sample.

    This absorbs model-loading latency (disk I/O for GECToR weights,
    server-side lazy load for LLM APIs, CUDA JIT compilations) so the
    timed evaluation reflects steady-state throughput.
    """
    try:
        model.fix_names("x = 1 + 2", ["x"])
    except Exception:
        pass  # If warmup fails, carry on — the real eval will report.


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def run_benchmark(
    force: bool = False,
    save_samples: bool = False,
    max_samples: Optional[int] = None,
) -> None:
    """Run all model × dataset combinations, caching to ``benchmarks/``."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    git_commit = _get_git_commit()

    total = len(MODELS) * len(DATASET_PATHS)
    run_count = 0

    for model_entry in MODELS:
        for ds_key, ds_path in DATASET_PATHS.items():
            slug = _slug(model_entry)
            out_path = OUTPUT_DIR / f"{slug}_{ds_key}.json"
            display = model_entry["display"]

            if out_path.exists() and not force:
                print(f"[SKIP] {display} on {ds_key} — cached at {out_path.name}")
                continue

            run_count += 1
            print(f"\n{'=' * 60}")
            print(f"[{run_count}/{total}] {display} on {ds_key}")
            print(f"{'=' * 60}")

            try:
                model = make_model(model_entry["key"], **model_entry["kwargs"])

                # Warmup: process one trivial sample so model-loading
                # overhead (disk I/O, server-side lazy load, CUDA JIT)
                # is not counted in timing.  The result is discarded.
                _warmup(model)

                t0 = time.perf_counter()
                results, metrics = evaluate(
                    model, ds_path,
                    max_samples=max_samples,
                    workers=1,
                )
                elapsed = time.perf_counter() - t0

                result_dict: Dict[str, Any] = {
                    "model": display,
                    "model_key": model_entry["key"],
                    "dataset": ds_key,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "git_commit": git_commit,
                    "elapsed_wall_seconds": elapsed,
                    "metrics": _metrics_to_dict(metrics),
                }

                if save_samples:
                    result_dict["per_sample"] = [
                        _sample_to_dict(r) for r in results
                    ]

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)

                em = metrics.exact_match_rate
                f1 = metrics.identifier_f1
                ms_per_kb = metrics.avg_time_per_kb_seconds * 1000
                print(f"Done: EM={em:.1%}  F1={f1:.1%}  "
                      f"time={elapsed:.1f}s  {ms_per_kb:.0f} ms/kB")
                print(f"Saved to {out_path}")

            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                # Save error record so downstream tools know this failed.
                error_dict = {
                    "model": display,
                    "model_key": model_entry["key"],
                    "dataset": ds_key,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "git_commit": git_commit,
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(error_dict, f, indent=2)
                print(f"Error record saved to {out_path}")

    print(f"\nAll done.  {run_count} benchmark(s) run, results in {OUTPUT_DIR}/")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks, caching results to disk.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if cached result exists.",
    )
    parser.add_argument(
        "--save-samples", action="store_true",
        help="Also save per-sample results (larger files).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit to first N error samples (for quick testing).",
    )
    args = parser.parse_args(argv)
    run_benchmark(
        force=args.force,
        save_samples=args.save_samples,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
