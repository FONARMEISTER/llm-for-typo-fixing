#!/usr/bin/env -S uv run python
"""Visualise cached benchmark results as charts.

Usage::

    uv run src/benchmark_viz.py                # plot all cached results
    uv run src/benchmark_viz.py --no-heatmap   # skip the heatmap

Reads JSON result files from ``benchmarks/`` and outputs charts to
``benchmarks/plots/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Headless — safe even without $DISPLAY.

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

BENCHMARKS_DIR = _project_root / "benchmarks"
PLOTS_DIR = BENCHMARKS_DIR / "plots"

# Marker shapes for distinct models.
MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "h", "*"]

# Palette — enough colours for up to 10 models.
COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #


def load_benchmarks() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load all JSON result files from ``benchmarks/``.

    Returns:
        ``{dataset_key: {model_display: {metrics...}}}`` — only datasets
        that have at least one valid (non-error) result.
    """
    results: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for path in sorted(BENCHMARKS_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "error" in data:
            # Skip error records — model failed on this dataset.
            continue
        if "metrics" not in data:
            continue

        ds = data.get("dataset", "unknown")
        model = data.get("model", path.stem)
        results[ds][model] = data["metrics"]

    return dict(results)


def model_order(models: List[str]) -> List[str]:
    """Sort models putting baselines (spellcheck, typos) first."""
    baselines = ["SpellCheck", "Typos"]
    ordered = [m for m in baselines if m in models]
    ordered += sorted(m for m in models if m not in baselines)
    return ordered


# ------------------------------------------------------------------ #
# Per-dataset grouped bar charts
# ------------------------------------------------------------------ #


def _bar_chart(
    ax: plt.Axes,
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str = "",
    ylim: Tuple[float, float] = (0.0, 1.0),
    fmt: str = ".1%",
) -> None:
    """Draw a vertical grouped bar chart on *ax*."""
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.55, color=COLOURS[:len(labels)], edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Value annotations.
    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * (ylim[1] - ylim[0]),
                f"{val:{fmt}}", ha="center", va="bottom", fontsize=7)


def plot_per_dataset(results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """For each dataset, produce a single figure with 4 subplots:
    EM, Precision, Recall, F1 — one bar per model.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for ds_key, models_dict in results.items():
        ordered = model_order(list(models_dict.keys()))
        em_vals = [models_dict[m]["exact_match_rate"] for m in ordered]
        prec_vals = [models_dict[m]["identifier_precision"] for m in ordered]
        rec_vals = [models_dict[m]["identifier_recall"] for m in ordered]
        f1_vals = [models_dict[m]["identifier_f1"] for m in ordered]

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle(f"Dataset: {ds_key}", fontsize=13, fontweight="bold")

        _bar_chart(axes[0], ordered, em_vals, "Exact Match")
        _bar_chart(axes[1], ordered, prec_vals, "Precision")
        _bar_chart(axes[2], ordered, rec_vals, "Recall")
        _bar_chart(axes[3], ordered, f1_vals, "F1")

        fig.tight_layout()
        out_path = PLOTS_DIR / f"bars_{ds_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


# ------------------------------------------------------------------ #
# Time (ms/kB) chart
# ------------------------------------------------------------------ #


def plot_time(results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """Produce a grouped bar chart of ms/kB per model per dataset."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = sorted(results.keys())
    all_models: set[str] = set()
    for ds in datasets:
        all_models.update(results[ds].keys())
    ordered = model_order(list(all_models))

    x = np.arange(len(datasets))
    width = 0.8 / len(ordered)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Throughput — ms per kB of code (lower is better)", fontweight="bold")
    ax.set_ylabel("ms / kB")
    ax.set_xticks(x + width * (len(ordered) - 1) / 2)
    ax.set_xticklabels(datasets, fontsize=10)

    for i, model in enumerate(ordered):
        vals = [
            results[ds].get(model, {}).get("avg_time_per_kb_seconds", 0) * 1000
            for ds in datasets
        ]
        offset = x + i * width
        bars = ax.bar(offset, vals, width, label=model,
                      color=COLOURS[i % len(COLOURS)], edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=6)

    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = PLOTS_DIR / "time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ------------------------------------------------------------------ #
# Heatmap
# ------------------------------------------------------------------ #


def plot_heatmap(results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """Produce two heatmaps: F1 score and ms/kB across model × dataset."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = sorted(results.keys())
    all_models = set()
    for ds in datasets:
        all_models.update(results[ds].keys())
    models = model_order(list(all_models))

    n_rows = len(models)
    n_cols = len(datasets)

    for metric_key, title, cmap, fmt in [
        ("identifier_f1", "F1 Score", "RdYlGn", ".3f"),
        ("avg_time_per_kb_seconds", "ms / kB", "YlOrRd_r", ".0f"),
    ]:
        matrix = np.zeros((n_rows, n_cols))
        mask = np.zeros((n_rows, n_cols), dtype=bool)

        for j, ds in enumerate(datasets):
            for i, model in enumerate(models):
                m = results[ds].get(model, {})
                if metric_key in m:
                    val = m[metric_key]
                    if metric_key == "avg_time_per_kb_seconds":
                        val *= 1000  # Convert to ms.
                    matrix[i, j] = val
                else:
                    mask[i, j] = True

        fig, ax = plt.subplots(figsize=(n_cols * 1.8, n_rows * 0.7))
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0,
                       vmax=1.0 if metric_key == "identifier_f1" else None)

        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(models, fontsize=10)
        ax.set_title(f"Model × Dataset — {title}", fontweight="bold")

        # Annotate cells.
        for i in range(n_rows):
            for j in range(n_cols):
                if mask[i, j]:
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=7, color="grey")
                else:
                    ax.text(j, i, f"{matrix[i, j]:{fmt}}", ha="center",
                            va="center", fontsize=7,
                            color="white" if matrix[i, j] > 0.5 and metric_key == "identifier_f1" else "black")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(title)

        fig.tight_layout()
        safe_name = metric_key.split("_")[0] if "f1" in metric_key else "time_heatmap"
        out_path = PLOTS_DIR / f"heatmap_{safe_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualise cached benchmark results.",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Skip heatmap generation.",
    )
    args = parser.parse_args(argv)

    if not BENCHMARKS_DIR.exists():
        print(f"Benchmarks directory not found: {BENCHMARKS_DIR}")
        print("Run 'uv run src/benchmark.py' first.")
        sys.exit(1)

    results = load_benchmarks()
    if not results:
        print("No valid benchmark results found.")
        sys.exit(1)

    datasets = list(results.keys())
    models_seen = sorted({m for ds in results.values() for m in ds})
    print(f"Loaded {len(datasets)} dataset(s): {', '.join(datasets)}")
    print(f"Loaded {len(models_seen)} model(s): {', '.join(models_seen)}")

    plot_per_dataset(results)
    plot_time(results)
    if not args.no_heatmap:
        plot_heatmap(results)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
