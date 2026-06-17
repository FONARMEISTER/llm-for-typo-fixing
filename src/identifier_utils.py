"""Shared Jedi-based identifier utilities.

Used by both :mod:`typo_injector` (dataset generation) and :mod:`harness`
(model evaluation).
"""

from __future__ import annotations

import atexit
import builtins
import keyword
import pathlib
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import jedi
import jedi.settings

# Give this process its own private parso cache directory.
#
# Setting cache_directory=None makes parso fall back to its *default* shared
# path (~/Library/Caches/Parso on macOS).  In a multiprocessing scenario that
# shared path is written to by every worker simultaneously, corrupting the
# pickle files and causing "EOFError: Ran out of input".
#
# Using a per-process temp directory means each process has a fully isolated
# cache — no cross-process writes, no corruption.  The directory is removed
# automatically when the process exits via atexit.
_jedi_cache_dir = tempfile.mkdtemp(prefix="jedi_proc_")
jedi.settings.cache_directory = pathlib.Path(_jedi_cache_dir)
atexit.register(shutil.rmtree, _jedi_cache_dir, True)

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
    try:
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
    except Exception:
        # Jedi may fail due to internal errors or edge cases.
        # Return empty result to indicate failure.
        return {}
