# Code Grammar Error Dataset

Tooling to build a dataset for training a model that detects and fixes **grammar** errors
in source code (typos in identifiers — variable / function / class names), **not** syntax
or logical errors.

## Idea

For every source snippet we produce two kinds of samples:

| field | description |
|---|---|
| `code` | the source code the model sees as input |
| `has_errors` | `true` if `code` contains injected identifier typos, else `false` |
| `fixed` | the corrected version of `code`; `null` when `has_errors=false` (nothing to fix) |
| `edits` | list of `{original_name, corrupted_name, num_occurrences}` (empty when `has_errors=false`) |
| `language` | e.g. `"python"` |

Clean sample (negative): `code` is the unmodified snippet, `fixed=null`.
Corrupted sample (positive): `code` has typos, `fixed` is the original clean snippet.

This supports both downstream tasks from one file:

- **Detection**: train a binary classifier on `code → has_errors`.
- **Correction**: train a seq2seq on `code → fixed` (filter to `has_errors=true`, or
  teach the model to copy `code` unchanged when it's already correct).

## Pipeline

1. **Download** code snippets from a public dataset (default: `mbpp`).
2. **Parse** them with Python's `tokenize` module to locate identifier occurrences.
3. **Filter** out keywords, builtins, imported names, attribute names, and strings.
4. **Inject typos** consistently across all occurrences of a chosen identifier.
5. **Write** JSONL with original / corrupted / fixed / labels.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build a small demo dataset (no network needed — uses bundled fallback snippets)
python -m src.build_dataset --source demo --out data/demo.jsonl --max-samples 50

# Build from MBPP (downloads via HuggingFace datasets)
python -m src.build_dataset --source mbpp --split train --out data/mbpp_grammar.jsonl

# Build from CodeSearchNet Python
python -m src.build_dataset --source code_search_net --split train \
    --out data/csn_python_grammar.jsonl --max-samples 5000
```

## Layout

```
src/
  typo_injector.py     # identifier-aware typo generator (Python via tokenize)
  build_dataset.py     # CLI: download -> corrupt -> write JSONL
  sources.py           # dataset loaders (mbpp, code_search_net, demo)
tests/
  test_typo_injector.py
data/                  # generated output (gitignored)
```
