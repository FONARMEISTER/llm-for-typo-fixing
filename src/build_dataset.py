"""Build a code-grammar-error dataset.

Usage::

    python -m src.build_dataset --source demo --out data/demo.jsonl
    python -m src.build_dataset --source mbpp --out data/mbpp/ --workers 8
    python -m src.build_dataset --source magicoder --out data/magicoder/
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List

from tqdm import tqdm

from .sources import iter_source, _KNOWN_SPLIT_SIZES
from .typo_injector import inject_typos


def _build_kwargs(args: argparse.Namespace, split: str) -> dict:
    kwargs: dict = {}
    if args.max_samples is not None:
        kwargs["max_samples"] = args.max_samples
    if args.source in ("mbpp", "code_search_net", "magicoder", "codealpaca"):
        kwargs["split"] = split
        kwargs["split"] = split
    return kwargs


def _estimate_total(source: str, split: str, max_samples: int | None) -> int | None:
    """Best-effort estimate of the number of input snippets, for the progress bar."""
    known = _KNOWN_SPLIT_SIZES.get(source, {}).get(split)
    if known is None:
        # code_search_net is streamed and huge; only show a total if the user capped it
        return max_samples
    if max_samples is not None:
        return min(known, max_samples)
    return known


def _emit(snippet: str, rng: random.Random, args: argparse.Namespace) -> Iterable[dict]:
    """Yield 1 negative (clean) + N positive (corrupted) samples per snippet.

    Schema per sample:
      * ``code``       — the input the model will see at inference time.
      * ``has_errors`` — boolean label (detection target).
      * ``fixed``      — the corrected version of ``code``. For clean samples this
                         is ``None`` (the code is already correct, there's nothing
                         to fix). For corrupted samples it is the original snippet.
      * ``edits``      — span-level supervision (empty for clean samples).
    """
    # negative example — guarantees the classifier sees both classes
    yield {
        "code": snippet,
        "has_errors": False,
        "fixed": None,
        "edits": [],
        "language": "python",
    }

    for _ in range(args.variants_per_snippet):
        result = inject_typos(
            snippet,
            rng=rng,
            max_edits=args.max_edits,
            p_edit=args.p_edit,
        )
        if not result.has_errors:
            continue
        yield {
            "code": result.corrupted,
            "has_errors": True,
            "fixed": result.original,
            "edits": [e.to_dict() for e in result.edits],
            "language": "python",
        }


def _worker(payload: tuple) -> tuple[str, List[dict]]:
    """Process a single snippet — runs in a child process."""
    snippet, seed, args_dict = payload

    # Suppress SyntaxWarning from parso about invalid escape sequences in
    # real-world code (e.g., "\i" in non-raw strings).
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    # Disable Jedi/parso disk cache to avoid multi-process cache corruption.
    import jedi.settings
    jedi.settings.cache_directory = None

    rng = random.Random(seed)
    args = argparse.Namespace(**args_dict)
    samples = list(_emit(snippet, rng, args))
    return snippet, samples


def _process_split(
    source: str,
    split: str,
    out_path: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> None:
    """Process a single split and write to output file."""
    total = _estimate_total(source, split, args.max_samples)

    # Collect all snippets upfront (needed for parallel distribution).
    snippets = list(iter_source(source, **_build_kwargs(args, split)))
    n_in = len(snippets)

    # Pre-generate seeds so runs are deterministic regardless of worker count.
    seeds = [rng.randint(0, 2**31 - 1) for _ in snippets]
    args_dict = vars(args)

    workers = getattr(args, "workers", 1) or multiprocessing.cpu_count() or 1

    n_neg = 0
    n_pos = 0
    n_skipped = 0  # snippets that produced no corrupted variant (all attempts failed)

    progress = tqdm(
        total=n_in,
        desc=f"building {os.path.basename(out_path)}",
        unit="snip",
        dynamic_ncols=True,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        if workers <= 1:
            # Serial path — simpler, avoids pickling overhead.
            for snippet, seed in zip(snippets, seeds):
                _, samples = _worker((snippet, seed, args_dict))
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    if sample["has_errors"]:
                        n_pos += 1
                    else:
                        n_neg += 1
                # Check if any positive samples were generated.
                if sum(1 for s in samples if s["has_errors"]) == 0:
                    n_skipped += 1
                progress.set_postfix(
                    clean=n_neg, corrupted=n_pos, no_typo=n_skipped, refresh=False,
                )
                progress.update(1)
        else:
            # Parallel path.
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(_worker, (snip, seed, args_dict))
                    for snip, seed in zip(snippets, seeds)
                ]
                for fut in as_completed(futures):
                    _, samples = fut.result()
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        if sample["has_errors"]:
                            n_pos += 1
                        else:
                            n_neg += 1
                    if sum(1 for s in samples if s["has_errors"]) == 0:
                        n_skipped += 1
                    progress.set_postfix(
                        clean=n_neg, corrupted=n_pos, no_typo=n_skipped, refresh=False,
                    )
                    progress.update(1)

    progress.close()
    print(
        f"read {n_in} snippets -> wrote {n_neg + n_pos} samples "
        f"({n_neg} clean, {n_pos} corrupted, {n_skipped} snippets had no usable identifiers) "
        f"to {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="demo",
        choices=["demo", "mbpp", "code_search_net", "magicoder", "codealpaca", "github_python"],
    )
    parser.add_argument("--out", required=True, help="output directory or file path")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="cap on number of source snippets to read per split",
    )
    parser.add_argument(
        "--variants-per-snippet", type=int, default=1,
        help="how many corrupted variants to produce per snippet",
    )
    parser.add_argument(
        "--max-edits", type=int, default=2,
        help="max distinct identifiers renamed per variant",
    )
    parser.add_argument(
        "--p-edit", type=float, default=0.8,
        help="probability of renaming each candidate identifier",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="number of parallel workers (0 = auto-detect CPU count, 1 = serial)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Determine if output is a directory or single file.
    out_dir = args.out
    if not out_dir.endswith("/"):
        if os.path.splitext(out_dir)[1]:  # Has extension → treat as file.
            # Pick the first available split (some sources only have test).
            known_splits = _KNOWN_SPLIT_SIZES.get(args.source, {})
            file_split = "train"
            for candidate in ("test", "validation", "train"):
                if known_splits.get(candidate, 0) > 0:
                    file_split = candidate
                    break
            splits = [file_split]
            out_base = out_dir
            out_dir = os.path.dirname(out_dir) or "."
            os.makedirs(out_dir, exist_ok=True)
        else:
            # Directory mode - process all splits
            splits = ["train", "validation", "test"]
            out_base = None
            os.makedirs(out_dir, exist_ok=True)
    else:
        # Explicit directory mode
        splits = ["train", "validation", "test"]
        out_base = None
        os.makedirs(out_dir, exist_ok=True)

    for split in splits:
        split_size = _KNOWN_SPLIT_SIZES.get(args.source, {}).get(split, 0)
        if split_size == 0:
            print(f"Skipping {split} split for {args.source} (no data available)")
            continue

        if out_base:
            # Single file mode
            out_path = out_base
        else:
            # Directory mode
            out_path = os.path.join(out_dir, f"{split}.jsonl")

        print(f"\nProcessing {split} split...")
        _process_split(args.source, split, out_path, rng, args)

    print(f"\n✓ All splits processed successfully!")


if __name__ == "__main__":
    main()
