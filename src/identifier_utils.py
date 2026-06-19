"""LibCST-based identifier utilities — shared by :mod:`typo_injector` (dataset
generation) and :mod:`harness` (model evaluation).

Migrated from Jedi to libcst for:
- **Thread safety** — no shared mutable state, no file cache.
- **Single-pass multi-rename** — ``apply_rename(source, rename_map)`` renames
  all identifiers in one CST traversal.
- **No filesystem** — works on strings directly.
- **About 10× faster** than the Jedi-based implementation.
"""

from __future__ import annotations

from src.identifier_utils_libcst import (
    _PROTECTED_NAMES,
    apply_jedi_rename,
    apply_rename,
    extract_renameable_identifiers,
    is_protected_name,
)

__all__ = [
    "_PROTECTED_NAMES",
    "apply_jedi_rename",
    "apply_rename",
    "extract_renameable_identifiers",
    "is_protected_name",
]
