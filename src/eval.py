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
) -> None:
    model = make_model(model_name)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    if max_samples:
        print(f"Max samples: {max_samples}")

    results, metrics = evaluate(model, dataset_path, max_samples=max_samples)

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
    args = parser.parse_args(argv)
    _run(args.model, args.dataset, args.output, args.max_samples)


if __name__ == "__main__":
    main()
