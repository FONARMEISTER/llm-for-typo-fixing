"""Build a code-grammar-error dataset.

Usage::

    python -m src.build_dataset --source demo --out data/demo.jsonl
    python -m src.build_dataset --source mbpp --split train --out data/mbpp.jsonl
    python -m src.build_dataset --source code_search_net --split train \
        --out data/csn.jsonl --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Iterable

from tqdm import tqdm

from src.sources import iter_source
from src.typo_injector import inject_typos


# Known sizes (rows per split) for non-streaming sources. Used purely to give
# the progress bar a `total` so it can show % and ETA. Approximate values are
# fine — tqdm just stops at whatever actually arrives.
_KNOWN_SPLIT_SIZES = {
    "demo": {"train": 8},
    "mbpp": {"train": 374, "validation": 90, "test": 500, "prompt": 10},
}


def _build_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {}
    if args.max_samples is not None:
        kwargs["max_samples"] = args.max_samples
    if args.source in ("mbpp", "code_search_net"):
        kwargs["split"] = args.split
    return kwargs


def _estimate_total(args: argparse.Namespace) -> int | None:
    """Best-effort estimate of the number of input snippets, for the progress bar."""
    known = _KNOWN_SPLIT_SIZES.get(args.source, {}).get(args.split)
    if known is None:
        # code_search_net is streamed and huge; only show a total if the user capped it
        return args.max_samples
    if args.max_samples is not None:
        return min(known, args.max_samples)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="demo", choices=["demo", "mbpp", "code_search_net"])
    parser.add_argument("--split", default="train", help="HuggingFace split (mbpp / code_search_net)")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="cap on number of source snippets to read")
    parser.add_argument("--variants-per-snippet", type=int, default=1,
                        help="how many corrupted variants to produce per snippet")
    parser.add_argument("--max-edits", type=int, default=2,
                        help="max distinct identifiers renamed per variant")
    parser.add_argument("--p-edit", type=float, default=0.8,
                        help="probability of renaming each candidate identifier")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    total = _estimate_total(args)
    n_in = 0
    n_neg = 0
    n_pos = 0
    n_skipped = 0  # snippets that produced no corrupted variant (all attempts failed)

    progress = tqdm(
        iter_source(args.source, **_build_kwargs(args)),
        desc=f"building {os.path.basename(args.out)}",
        unit="snip",
        total=total,
        dynamic_ncols=True,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        for snippet in progress:
            n_in += 1
            produced_positive = False
            for sample in _emit(snippet, rng, args):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if sample["has_errors"]:
                    n_pos += 1
                    produced_positive = True
                else:
                    n_neg += 1
            if not produced_positive:
                n_skipped += 1

            # update once per snippet to keep overhead negligible
            progress.set_postfix(
                clean=n_neg,
                corrupted=n_pos,
                no_typo=n_skipped,
                refresh=False,
            )

    progress.close()
    print(
        f"read {n_in} snippets -> wrote {n_neg + n_pos} samples "
        f"({n_neg} clean, {n_pos} corrupted, {n_skipped} snippets had no usable identifiers) "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
