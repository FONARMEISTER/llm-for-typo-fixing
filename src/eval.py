"""CLI for running evaluation of name-fixing models.

Usage::

    uv run src/eval.py --model spellcheck --dataset data/demo.jsonl
    uv run src/eval.py --model spellcheck --dataset data/demo.jsonl --output results.json
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from typing import Optional, Sequence

from .harness import SampleResult, evaluate
from .models import MODEL_REGISTRY, make_model


def _run(
    model_name: str,
    dataset_path: str,
    output_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    workers: int = 1,
    preset: Optional[str] = None,
    llm_config: str = "config/llm_presets.toml",
) -> None:
    # Build model kwargs.
    model_kwargs: dict = {}
    display_name = model_name
    if model_name == "llm_api":
        if preset:
            model_kwargs["preset"] = preset
            model_kwargs["config_path"] = llm_config
            display_name = f"llm_api ({preset})"
        else:
            print("Error: --preset is required when --model llm_api")
            return

    model = make_model(model_name, **model_kwargs)
    print(f"Model: {display_name}")
    print(f"Dataset: {dataset_path}")

    max_parallel = getattr(model, "max_parallel_requests", 1)
    if max_parallel > 1:
        print(f"Parallelism: {max_parallel} threads (API concurrency)")
    else:
        n_workers = workers if workers > 0 else __import__("os").cpu_count() or 1
        print(f"Workers: {n_workers}")

    if max_samples:
        print(f"Max samples: {max_samples}")

    results, metrics = evaluate(
        model, dataset_path, max_samples=max_samples, workers=workers,
    )

    # Print metrics to stdout.
    print()
    print("=" * 60)
    print(f"Total samples evaluated: {metrics.total_samples}")
    print(f"Exact match rate:       {metrics.exact_match_rate:.4f}")
    print(f"Identifier precision:   {metrics.identifier_precision:.4f}")
    print(f"Identifier recall:      {metrics.identifier_recall:.4f}")
    print(f"Identifier F1:          {metrics.identifier_f1:.4f}")
    print(f"Avg norm. edit distance:{metrics.avg_normalized_edit_distance:.4f}")
    print("=" * 60)

    if output_path:
        _save_results(results, metrics, output_path)
        print(f"\nResults saved to {output_path}")


def _save_results(
    results: list[SampleResult], metrics, path: str,
) -> None:
    """Save per-sample results and aggregate metrics to a JSON file."""
    per_sample = [
        {
            "index": r.sample_index,
            "exact_match": r.exact_match,
            "corrupted_code": r.corrupted_code,
            "predicted_code": r.predicted_code,
            "ground_truth_code": r.ground_truth_code,
            "model_fixes": r.model_fixes,
            "gt_original_names": r.gt_original_names,
            "gt_corrupted_names": r.gt_corrupted_names,
        }
        for r in results
    ]
    output = {
        "metrics": asdict(metrics),
        "per_sample": per_sample,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = ArgumentParser(description="Evaluate name-fixing models.")
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY), required=True, help="Model to evaluate."
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to JSONL dataset."
    )
    parser.add_argument(
        "--output", default=None, help="Optional JSON path for detailed output."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit to first N error samples."
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Worker processes (0=auto-detect CPU count, 1=serial).",
    )
    parser.add_argument(
        "--preset", default=None,
        help="Inference preset name (required when --model llm_api).",
    )
    parser.add_argument(
        "--llm-config", default="config/llm_presets.toml",
        help="Path to TOML preset config file (default: config/llm_presets.toml).",
    )
    args = parser.parse_args(argv)
    _run(
        args.model, args.dataset, args.output, args.max_samples, args.workers,
        preset=args.preset, llm_config=args.llm_config,
    )


if __name__ == "__main__":
    main()
