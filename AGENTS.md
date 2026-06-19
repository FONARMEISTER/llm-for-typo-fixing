# AGENTS.md — Project knowledge for humans and AI agents

## Architecture overview

```
src/
  identifier_utils.py   — LibCST-based helpers (name extraction, rename).
  text_utils.py          — CamelCase / snake_case splitting and reassembly.
  typo_injector.py       — Dataset generator: injects textual typos into Python code.
  sources.py             — Dataset sources (demo snippets, MBPP, Magicoder, CodeAlpaca,
                            GitHub Python).
  build_dataset.py       — CLI that glues sources + injector → JSONL dataset.
  harness.py             — Evaluation pipeline: JSONL → model → rename → metrics.
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

Both pipelines use `identifier_utils.extract_renameable_identifiers()` for
scope-aware extraction and `identifier_utils.apply_rename()` for batch rename.
`apply_rename_single()` provides position-based single-identifier rename for
the typo injector.

## LibCST scope analysis

``identifier_utils.py`` uses libcst's ``ScopeProvider`` to find all
renameable identifiers (functions, classes, variables, parameters) while
filtering out keywords, builtins, imports, ``self``/``cls``, and dunders.

``apply_rename(source, rename_map)`` renames all identifiers in a single
CST traversal — no re-extraction needed.  ``apply_rename_single()`` renames
a single identifier at a specific position (used during typo injection where
one name is corrupted at a time).

libcst is thread-safe by design: no shared mutable state, no file cache.

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
- Identifier corruption: extracts libcst definitions, picks
  **one random definition position per name** (even if the name is defined in
  multiple scopes), calls ``apply_rename_single()`` for scope-aware rename.
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

- `evaluate(model, dataset_path, max_samples, workers)`: reads JSONL, processes
  only `has_errors=true` samples.
- Per sample: extracts identifiers from corrupted code, calls
  `model.fix_names(code, names)`, applies all fixes at once via
  ``apply_rename()`` (single libcst pass — no re-extraction needed).
- ``model_fixes`` in :class:`SampleResult` reflects the model's **raw
  suggestions** (not just the successfully applied ones), so identifier-level
  metrics evaluate the model rather than the rename harness.
- Metrics: exact match, identifier-level precision/recall/F1, normalised
  Levenshtein edit distance (via `Levenshtein` package).
- Parallel: ``workers=0`` auto-detects CPU count; workers re-create the model
  by calling ``make_model(model.name)``.  The serial path (``workers=1``) avoids
  pickling overhead and is the default.
- Workers use libcst which is thread-safe — no cache isolation needed.

## Model interface

```python
class NameFixer(ABC):
    name: str = "unknown"  # registry key for parallel worker re-creation
    def fix_names(self, code: str, names: list[str]) -> dict[str, str]:
        """Return {corrupted_name: fixed_name} for identifiers to fix.
        Names NOT in the result are treated as already correct."""
```

To add a new model:
1. Create `src/models/<name>.py` with a class inheriting `NameFixer`.
2. Set ``name = "your-model"`` on the class.
3. Register a factory in `src/models/__init__.py` → `MODEL_REGISTRY`.

## GECToR architecture

- **Encoder**: `microsoft/codebert-base` (default; any HuggingFace encoder works).
- **Heads**: `tag_head` (Linear → vocab_size) + `detect_head` (Linear → 2).
- **Dual loss**: tag CE + detect CE + optional REINFORCE Levenshtein auxiliary.
- **Iterative inference**: model applies tags, re-encodes the corrected code,
  and repeats until convergence or `max_iter` passes.

### Tag vocabulary modes

**Character-edit mode** (default, `--vocab-type char-edit`):
- Static vocabulary (~4000 tags), data-independent.
- Generalises to **any identifier** — including ones never seen during training.
- Five operation families mirror `make_typo()`:
  - `$SWAP_<pos>` — swap characters at positions *pos* and *pos+1*
  - `$DEL_<pos>` — delete the character at *pos*
  - `$INS_<pos>_<char>` — insert *char* before position *pos*
  - `$SUB_<pos>_<char>` — substitute the character at *pos* with *char*
  - `$CASE_<pos>` — flip the case of the character at *pos*
- `compute_char_edit_tag(corrupted, original)` determines the tag;
  `apply_char_edit(token, tag)` applies it at inference time.
- One tag per code token per iteration; multi-typo names are fixed across
  successive iterative passes.

**Replace mode** (legacy, `--vocab-type replace`):
- `$REPLACE_<name>` tags built from training data.
- Cannot generalise to unseen identifiers (OOV problem).

### Training

```bash
make train-gector                # train on MBPP with defaults
make train-gector GECTOR_ENCODER=microsoft/codebert-base  # explicit
make train-gector-all            # train on all datasets merged
```

CLI flags: `--vocab-type`, `--lev-weight`, `--detect-weight`, etc.

## Spellchecker baseline

- Uses `pyspellchecker` (PyPI: `pyspellchecker`).
- Splits identifiers via `text_utils.split_identifier()` (CamelCase + snake_case).
- Reassembles via `text_utils.reassemble_identifier()` (preserves case).
- Skips names shorter than 3 characters.

## Typos baseline

- Uses the ``typos`` CLI tool (source-code-aware spell corrector) via subprocess.
- Feeds the full source code to ``typos - --write-changes`` and maps corrected
  identifiers back to the original names by position matching (±5 columns).
- High precision, low recall (curated dictionary misses many typos).

## LLM API baseline

- Uses the ``openai`` Python package to call any OpenAI-compatible chat
  completions API (local llama.cpp, llama-swap, Ollama, vLLM, OpenAI, etc.).
- ``response_format={"type": "json_object"}`` for structured JSON output.
- Inference presets are configured in ``config/llm_presets.toml`` (TOML).
  Each preset specifies: ``base_url``, ``model``, ``api_key_env`` (or ``""``),
  ``max_tokens``, ``temperature``, ``system_prompt``.
- Supports both reasoning models (Gemma 4, Qwen 3.5 with ``reasoning_content``)
  and classic models.
- Exponential backoff retry on transient errors (429, 5xx, timeout,
  connection): up to 5 retries (configurable via ``max_retries`` and
  ``retry_base_delay``).  Respects ``Retry-After`` headers.
- CLI usage: ``make eval-llm PRESET=gemma4-26b`` or:
  ``uv run python -m src.eval --model llm_api --preset gemma4-26b --dataset data/demo.jsonl``
- Demo results (Gemma 4 26B Q6K): 84.6% EM, 100% precision, 89.5% recall, 94.5% F1.

## Identifier splitting (text_utils)

- `split_identifier("myCamel_Snake")` → `["my", "Camel", "Snake"]`.
- Handles leading underscores: `"_private"` → `["_", "private"]`.
- Reassembly must match word count exactly, or it falls back to the original.

## Build & test commands

```bash
make test              # all tests (pytest)
make build-demo        # regenerate data/demo.jsonl
make build-github-python  # build test dataset from real GitHub Python files
make build-all         # regenerate all datasets (parallel by default)
make eval              # spellchecker on demo dataset
make eval-llm PRESET=gemma4-26b  # LLM model via llama-swap on demo dataset
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

## Commit message conventions

- **Class names** in square brackets: ``[LLMAPIFixer]``, ``[NameFixer]``.
- **Functions, methods, modules, files** in backticks with ``()``:
  `` `fix_names()` ``, `` `_process_sample()` ``, `` `evaluate()` ``.
- Qualify method with class when ambiguous: `` `LLMAPIFixer.from_preset()` ``.
- Always parentheses after function/method name.
- **Don't** append test-count footnotes (e.g. «All 57 tests pass») — every
  commit is expected to keep the suite green unless stated otherwise.
- **Don't** repeat in prose what the diff already says.  Focus on *why* and
  *impact*: what changed behaviourally, what edge case was fixed, what new
  capability is available.  Write for a future reader (yourself, another
  agent, or a teammate) who is skimming ``git log`` to understand the
  project's history.

## Dependencies (pyproject.toml)

- ``libcst>=1.5.0`` — scope-aware Python refactoring (CST-based).
- `pyspellchecker>=0.8.0` — baseline spellchecker.
- `Levenshtein>=0.27.0` — C-accelerated edit distance (not `python-Levenshtein`).
- `datasets>=2.14.0` — HuggingFace datasets (MBPP, etc.).
- `openai` — OpenAI-compatible client for LLM baseline.
- `torch>=2.0.0`, `transformers>=4.30.0` — optional (``[ml]`` extra) for `typo_datasets.py` PyTorch data loaders.
