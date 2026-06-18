"""Shared Jedi-based identifier utilities.

Used by both :mod:`typo_injector` (dataset generation) and :mod:`harness`
(model evaluation).
"""

from __future__ import annotations

import builtins
import keyword
import warnings
from typing import Dict, List, Optional, Tuple

import jedi

# Real-world code often contains invalid escape sequences (e.g., "\i" in
# non-raw strings).  parso emits SyntaxWarning for these, which is just noise
# for our refactoring use case.
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Names we never treat as renameable: language keywords, soft keywords,
# builtins, and idiomatic "fixed" names.
_PROTECTED_NAMES: set[str] = (
    set(keyword.kwlist)
    | set(getattr(keyword, "softkwlist", []))
    | set(dir(builtins))
    | {"self", "cls", "__init__", "__name__", "__main__", "__file__", "__doc__"}
)


def is_protected_name(name: str) -> bool:
    return name in _PROTECTED_NAMES or (name.startswith("__") and name.endswith("__"))


def extract_renameable_identifiers(
    source: str,
) -> Dict[str, List[Tuple[int, int]]]:
    """Return ``name -> [(line, col), ...]`` of definition positions.

    Uses Jedi for scope-aware extraction.  A name may appear multiple
    times if it is defined in different scopes (e.g. ``result`` in both
    ``outer()`` and ``inner()``).

    We include ``statement``, ``function``, ``class``, and ``param``
    definitions.

    We skip keywords, builtins, protected names, names shorter than 3
    characters, and dunder names.
    """
    script = jedi.Script(code=source)
    names = script.get_names(all_scopes=True, definitions=True, references=False)

    result: Dict[str, List[Tuple[int, int]]] = {}
    for n in names:
        name = n.name
        if is_protected_name(name):
            continue

        if n.type not in ("statement", "function", "class", "param"):
            continue

        line = n.line
        col = n.column
        if line is None or col is None:
            continue

        result.setdefault(name, []).append((line, col))

    return result


def apply_jedi_rename(
    source: str, line: int, col: int, new_name: str,
    path: Optional[str] = None,
) -> dict[str, str]:
    """Rename the identifier at ``(line, col)`` to ``new_name``.

    Returns a mapping ``{file_path: new_source}`` for **all** files changed
    by the refactoring.  For single-file usage the only key is typically
    ``""`` (the empty string).

    Raises ``jedi.api.exceptions.RefactoringError`` if there is no
    identifier under the cursor.
    """
    kwargs: dict = {"code": source}
    if path is not None:
        kwargs["path"] = path
    script = jedi.Script(**kwargs)  # type: ignore[arg-type]
    refactoring = script.rename(line=line, column=col, new_name=new_name)
    result: dict[str, str] = {}
    if refactoring is not None:
        for file_path, change in refactoring.get_changed_files().items():
            result[file_path] = change.get_new_code()
    return result
