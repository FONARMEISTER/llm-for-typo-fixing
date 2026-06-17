# AGENTS.md — Project knowledge for humans and AI agents

## Architecture overview

```
src/
  identifier_utils.py   — Shared Jedi helpers (name extraction, rename).
  text_utils.py          — CamelCase / snake_case splitting and reassembly.
  typo_injector.py       — Dataset generator: injects textual typos into Python code.
  sources.py             — Dataset sources (demo snippets, MBPP, etc.).
  build_dataset.py       — CLI that glues sources + injector → JSONL dataset.
  harness.py             — Evaluation pipeline: JSONL → model → Jedi rename → metrics.
  eval.py                — CLI for running evaluation.
  viewer.py              — Local HTML viewer for manual dataset inspection.
  models/
    __init__.py           — MODEL_REGISTRY, make_model().
    base.py               — NameFixer (ABC).
    spellcheck.py         — SpellChecker baseline (pyspellchecker).
```

Data flows:

1. **Dataset generation**: `sources.py` → `typo_injector.inject_typos()` → JSONL (`data/*.jsonl`).
2. **Evaluation**: JSONL → `harness.evaluate(model, path)` → metrics.

Both pipelines use `identifier_utils.extract_renameable_identifiers()` and
`identifier_utils.apply_jedi_rename()` — scope-aware refactoring via Jedi.

## Jedi API — crucial gotchas

### The `path=""` slowdown

```python
# ❌ SLOW — adds ~1.9 s overhead per call:
jedi.Script(code=source, path="")

# ✅ Fast — omit path or pass None:
jedi.Script(code=source)
jedi.Script(code=source, path=None)
```

Only pass a real filesystem path when you need multi-file refactoring resolution.

### Multiprocess cache corruption

Parallel `ProcessPoolExecutor` workers share `~/.cache/parso/` and will
corrupt each other's pickle files.  Inside any worker function that calls
Jedi, do **before the first Jedi call**:

```python
import jedi.settings
jedi.settings.cache_directory = None
```

If you see `EOFError: Ran out of input` from `parso/cache.py`, delete
`~/.cache/parso/` and add the setting above.

### Name position vs definition start position

```python
n = some_jedi_name
# ✅ Use these for rename():
line = n.line        # position of the NAME token (e.g., the 'c' in 'compute')
col  = n.column

# ❌ DON'T use this for rename():
pos = n.get_definition_start_position()  # points to 'def' keyword, not the name
```

`get_definition_start_position()` returns the start of the enclosing statement
(`def`, `class`, etc.), **not** the identifier itself.  Renaming at that
position will fail with `RefactoringError` on already-modified code.

### Multi-file refactoring

`jedi.Script.rename()` returns a `Refactoring` whose `get_changed_files()`
returns `dict[path, ChangedFile]`.  Always iterate **all** entries — never
assume a single file.  Our `apply_jedi_rename()` reflects this (returns
`dict[str, str]`), and single-file callers wrap it.

### Internal Jedi errors on edge cases

Some code triggers internal Jedi bugs (e.g., `ValueError: too many values to
unpack` in type inference).  Our `inject_typos` catches all `Exception` around
`_apply_rename` and skips the offending edit.  If you add new Jedi call sites,
wrap them similarly.

## Dataset format (JSONL)

```json
{
  "code":       "...",     // corrupted code (or original if has_errors=false)
  "fixed":      "...",     // ground-truth clean code
  "has_errors": true,
  "edits": [{"original_name": "number", "corrupted_name": "nubmer", "num_occurrences": 2}],
  "language": "python"
}
```

## How the typo injector works

- `inject_typos(source, rng, max_edits, p_edit, corrupt_comments, p_comment_word)`
- Identifier corruption: extracts Jedi definitions, picks
  **one random definition position per name** (even if the name is defined in
  multiple scopes), calls Jedi `rename()` for scope-aware refactoring.
- Comment corruption: tokenizes source, finds `COMMENT` tokens, corrupts
  random words inside them (skipping markers like `TODO`, `FIXME`).
- `make_typo()`: randomly applies one of 5 operations (swap, delete, duplicate,
  keyboard-substitute, case-flip).  Returns `None` if no valid typo can be
  produced.
- Protected names: keywords, soft keywords, builtins, `self`, `cls`,
  dunders — never renamed.  Defined in `identifier_utils._PROTECTED_NAMES`.
- Coordinates are re-extracted after each rename to avoid staleness when
  two names share a line.

## How the harness works

- `evaluate(model, dataset_path, max_samples)`: reads JSONL, processes
  only `has_errors=true` samples.
- Per sample: extracts identifiers from corrupted code, calls
  `model.fix_names(code, names)`, applies each suggested fix via Jedi rename
  (re-extracting positions after each rename because offsets may shift).
- Metrics: exact match, identifier-level precision/recall/F1, normalised
  Levenshtein edit distance (via `Levenshtein` package).

## Model interface

```python
class NameFixer(ABC):
    def fix_names(self, code: str, names: list[str]) -> dict[str, str]:
        """Return {corrupted_name: fixed_name} for identifiers to fix.
        Names NOT in the result are treated as already correct."""
```

To add a new model:
1. Create `src/models/<name>.py` with a class inheriting `NameFixer`.
2. Register a factory in `src/models/__init__.py` → `MODEL_REGISTRY`.

## Spellchecker baseline

- Uses `pyspellchecker` (PyPI: `pyspellchecker`).
- Splits identifiers via `text_utils.split_identifier()` (CamelCase + snake_case).
- Reassembles via `text_utils.reassemble_identifier()` (preserves case).
- Skips names shorter than 3 characters.

## Identifier splitting (text_utils)

- `split_identifier("myCamel_Snake")` → `["my", "Camel", "Snake"]`.
- Handles leading underscores: `"_private"` → `["_", "private"]`.
- Reassembly must match word count exactly, or it falls back to the original.

## Build & test commands

```bash
make test              # all tests (pytest)
make build-demo        # regenerate data/demo.jsonl
make build-all         # regenerate all datasets (parallel by default)
make eval              # spellchecker on demo dataset
make viewer            # HTML dataset viewer at localhost:8765
make clean             # delete generated data
```

Always use `uv run python -m src.<module>` for scripts (relative imports).

## Parallel dataset building

`build_dataset.py` supports `--workers N` (default 0 = auto-detect CPU count).
Uses `ProcessPoolExecutor`.  Seeds are pre-computed for determinism regardless
of worker count.

## Testing conventions

- Tests use seeded `random.Random(seed)` for determinism.
- `test_harness.py` has `PerfectFixer` (ground-truth oracle) and `NoopFixer`
  (baseline that changes nothing) for integration tests.
- Snippet constants in tests use raw triple-quoted strings (`r"""..."""`).
- The project uses Jupytext: `.ipynb` + `.py` pairs — edit the `.py` file,
  ignore the `.ipynb`.

## Dependencies (pyproject.toml)

- `jedi>=0.20.0` — scope-aware Python refactoring.
- `pyspellchecker>=0.8.0` — baseline spellchecker.
- `Levenshtein>=0.27.0` — C-accelerated edit distance (not `python-Levenshtein`).
- `datasets>=2.14.0` — HuggingFace datasets (MBPP, etc.).
- `tqdm>=4.65.0` — progress bars.
