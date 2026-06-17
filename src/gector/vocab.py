"""Tag vocabulary for the GECToR code-typo tagger.

The vocabulary maps edit-operation strings to integer indices and back.

Special tags
------------
``$KEEP``     — index 0, leave token unchanged
``$DELETE``   — index 1, delete token (rarely needed for identifier typos)
``$UNK``      — index 2, unknown tag (used as a fallback during inference)

Data-derived tags
-----------------
``$REPLACE_<tok>`` — replace the current token with ``<tok>``.

The vocabulary is built by scanning a JSONL dataset produced by
:mod:`src.build_dataset` and collecting all ``(corrupted_name,
original_name)`` pairs from the ``edits`` field.  Each such pair
contributes a ``$REPLACE_<original_name>`` tag.

Persistence
-----------
The vocabulary is saved as a plain-text file (one tag per line, index =
line number) so it can be inspected and version-controlled easily.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

KEEP_TAG = "$KEEP"
DELETE_TAG = "$DELETE"
UNK_TAG = "$UNK"

_SPECIAL_TAGS: List[str] = [KEEP_TAG, DELETE_TAG, UNK_TAG]

KEEP_IDX = 0
DELETE_IDX = 1
UNK_IDX = 2


def replace_tag(token: str) -> str:
    """Return the ``$REPLACE_<token>`` tag string for *token*."""
    return f"$REPLACE_{token}"


def is_replace_tag(tag: str) -> bool:
    return tag.startswith("$REPLACE_")


def replacement_token(tag: str) -> str:
    """Extract the target token from a ``$REPLACE_<token>`` tag."""
    return tag[len("$REPLACE_"):]


# ------------------------------------------------------------------ #
# Vocabulary class
# ------------------------------------------------------------------ #


class TagVocab:
    """Bidirectional mapping between tag strings and integer indices.

    Parameters
    ----------
    tags:
        Ordered list of tag strings.  The first three entries must be
        ``$KEEP``, ``$DELETE``, ``$UNK`` (in that order).
    """

    def __init__(self, tags: List[str]) -> None:
        assert tags[:3] == _SPECIAL_TAGS, (
            f"First three tags must be {_SPECIAL_TAGS}, got {tags[:3]}"
        )
        self._tags = list(tags)
        self._tag2idx: Dict[str, int] = {t: i for i, t in enumerate(tags)}

    # ---------------------------------------------------------------- #
    # Properties
    # ---------------------------------------------------------------- #

    @property
    def size(self) -> int:
        return len(self._tags)

    @property
    def tags(self) -> List[str]:
        return list(self._tags)

    # ---------------------------------------------------------------- #
    # Lookup
    # ---------------------------------------------------------------- #

    def tag2idx(self, tag: str) -> int:
        """Return the index for *tag*, falling back to ``UNK_IDX``."""
        return self._tag2idx.get(tag, UNK_IDX)

    def idx2tag(self, idx: int) -> str:
        if 0 <= idx < len(self._tags):
            return self._tags[idx]
        return UNK_TAG

    def __len__(self) -> int:
        return self.size

    def __contains__(self, tag: str) -> bool:
        return tag in self._tag2idx

    # ---------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        """Write one tag per line to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._tags) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TagVocab":
        """Load a vocabulary saved with :meth:`save`."""
        path = Path(path)
        tags = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
        tags = [t for t in tags if t]  # drop blank lines
        return cls(tags)

    # ---------------------------------------------------------------- #
    # Factory
    # ---------------------------------------------------------------- #

    @classmethod
    def build_from_jsonl(
        cls,
        paths: Iterable[str | Path],
        min_freq: int = 1,
        max_replace_tags: Optional[int] = None,
    ) -> "TagVocab":
        """Build a vocabulary by scanning JSONL dataset files.

        For every sample with ``has_errors=true`` the ``edits`` list is
        inspected.  Each edit contributes a ``$REPLACE_<original_name>``
        tag (the correction the model should predict for the corrupted
        token).

        Parameters
        ----------
        paths:
            One or more paths to ``.jsonl`` files produced by
            :mod:`src.build_dataset`.
        min_freq:
            Minimum number of times a ``$REPLACE_*`` tag must appear in
            the dataset to be included in the vocabulary.
        max_replace_tags:
            If set, keep only the *max_replace_tags* most frequent
            ``$REPLACE_*`` tags (after applying *min_freq*).

        Returns
        -------
        TagVocab
        """
        from collections import Counter

        replace_counts: Counter[str] = Counter()

        for path in paths:
            path = Path(path)
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    if not sample.get("has_errors"):
                        continue
                    for edit in sample.get("edits", []):
                        orig = edit.get("original_name", "")
                        if orig:
                            replace_counts[replace_tag(orig)] += 1

        # Filter by frequency.
        filtered = {t: c for t, c in replace_counts.items() if c >= min_freq}

        # Sort by frequency descending, then alphabetically for determinism.
        sorted_tags = sorted(filtered.keys(), key=lambda t: (-filtered[t], t))

        if max_replace_tags is not None:
            sorted_tags = sorted_tags[:max_replace_tags]

        all_tags = _SPECIAL_TAGS + sorted_tags
        return cls(all_tags)

    @classmethod
    def minimal(cls) -> "TagVocab":
        """Return a minimal vocabulary with only the three special tags.

        Useful for unit tests and smoke tests that don't need real data.
        """
        return cls(list(_SPECIAL_TAGS))
