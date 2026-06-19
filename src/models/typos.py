"""typos-based baseline model.

Uses `typos <https://github.com/crate-ci/typos>`_ — a source-code-aware
spelling corrector — to fix corrupted identifier names.  The full source
code is fed to ``typos - --write-changes`` via subprocess, and identifier
names are mapped between original and corrected code by position.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Dict, List, Tuple

from .base import NameFixer
from ..identifier_utils import extract_renameable_identifiers, UnparseableCodeError


class TyposFixer(NameFixer):
    """Baseline: correct identifiers using the ``typos`` CLI tool."""

    name = "typos"

    def __init__(self, timeout: float = 30.0) -> None:
        self._binary = shutil.which("typos")
        if self._binary is None:
            # Under `uv run` the binary lives in the project venv.
            from pathlib import Path
            candidate = Path(__file__).resolve().parent.parent.parent / ".venv" / "bin" / "typos"
            if candidate.is_file():
                self._binary = str(candidate)
            else:
                raise RuntimeError(
                    "typos binary not found.  Install it with:  uv add typos"
                )
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    # NameFixer interface
    # ------------------------------------------------------------------ #

    def fix_names(self, code: str, names: List[str]) -> Dict[str, str]:
        """Run ``typos`` on the full source, then map corrections back to names.

        Only the ``code`` is used — ``names`` is treated as the set of
        identifiers the *caller* wants fixes for.  We extract identifiers
        from the corrected output and match them to the requested names by
        position.
        """
        corrected = self._run_typos(code)
        if corrected == code:
            return {}

        # Build a position → corrected-name lookup from the corrected code.
        try:
            corr_idents = extract_renameable_identifiers(corrected)
        except UnparseableCodeError:
            return {}
        pos_to_name: Dict[Tuple[int, int], str] = {}
        for cname, positions in corr_idents.items():
            for pos in positions:
                pos_to_name[pos] = cname

        # Extract identifiers from the original code to get their positions.
        try:
            orig_idents = extract_renameable_identifiers(code)
        except UnparseableCodeError:
            return {}

        result: Dict[str, str] = {}
        for name in names:
            if name not in orig_idents:
                continue
            orig_positions = orig_idents[name]
            for oline, ocol in orig_positions:
                # typos edits are in-place (same line, nearby column).
                # Column may shift by a couple of characters if an earlier
                # correction on the same line changed lengths.
                for delta in range(-5, 6):
                    candidate = pos_to_name.get((oline, ocol + delta))
                    if candidate is not None and candidate != name:
                        result[name] = candidate
                        break
                if name in result:
                    break

        return result

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_typos(self, code: str) -> str:
        """Feed *code* to ``typos - --write-changes`` and return corrected text."""
        proc = subprocess.run(
            [self._binary, "-", "--write-changes"],
            input=code.encode("utf-8"),
            capture_output=True,
            timeout=self._timeout,
        )
        # Exit code 0 = no typos found, code 2 = typos found and fixed.
        # Anything else is an error (e.g. a config problem).
        if proc.returncode not in (0, 2):
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"typos exited with code {proc.returncode}: {stderr}"
            )
        return proc.stdout.decode("utf-8")
