"""Abstract interface for identifier name-fixing models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class NameFixer(ABC):
    """A model that fixes textual errors in identifiers inside source code.

    The model receives the (possibly corrupted) code and a list of
    identifier names found in that code.  It returns a dictionary mapping
    corrupted names to their suggested corrections.  Names that the model
    considers correct should be omitted from the result dictionary.
    """

    @abstractmethod
    def fix_names(self, code: str, names: List[str]) -> Dict[str, str]:
        """Return ``{corrupted_name: fixed_name}`` for identifiers to fix.

        Args:
            code: Full Python source code, possibly containing typos.
            names: List of all renameable identifiers in ``code`` (as
                   extracted by :func:`src.identifier_utils.extract_renameable_identifiers`).

        Returns:
            A mapping from corrupted identifier names to their corrected
            forms.  Only entries that the model *would* change should be
            included — names not in the dict are treated as already correct.
        """
        ...
