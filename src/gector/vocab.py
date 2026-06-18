"""Tag vocabulary for the GECToR code-typo tagger.

The vocabulary maps edit-operation strings to integer indices and back.

Special tags
------------
``$KEEP``     — index 0, leave token unchanged
``$DELETE``   — index 1, delete token (rarely needed for identifier typos)
``$UNK``      — index 2, unknown tag (used as a fallback during inference)

Tag modes
---------
**Replace mode** (legacy):
    ``$REPLACE_<tok>`` — replace the current token with ``<tok>``.
    Vocabulary is data-derived; cannot generalise to unseen identifiers.

**Character-edit mode** (recommended):
    Tags encode single-character edit operations that are *identifier-
    agnostic* and generalise to any token the model encounters at inference
    time.  The five operation families mirror the typo operations used by
    :func:`~src.typo_injector.make_typo`:

    ``$SWAP_<pos>``         — swap characters at positions *pos* and *pos+1*
    ``$DEL_<pos>``          — delete the character at *pos*
    ``$INS_<pos>_<char>``   — insert *char* before position *pos*
    ``$SUB_<pos>_<char>``   — substitute the character at *pos* with *char*
    ``$CASE_<pos>``         — flip the case of the character at *pos*

    The vocabulary is *static* — built by enumerating all combinations up
    to ``MAX_CHAR_POS`` positions and ``EDIT_ALPHABET`` characters.

Persistence
-----------
The vocabulary is saved as a plain-text file (one tag per line, index =
line number) so it can be inspected and version-controlled easily.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


# ------------------------------------------------------------------ #
# Replace-mode helpers (legacy)
# ------------------------------------------------------------------ #


def replace_tag(token: str) -> str:
    """Return the ``$REPLACE_<token>`` tag string for *token*."""
    return f"$REPLACE_{token}"


def is_replace_tag(tag: str) -> bool:
    return tag.startswith("$REPLACE_")


def replacement_token(tag: str) -> str:
    """Extract the target token from a ``$REPLACE_<token>`` tag."""
    return tag[len("$REPLACE_"):]


# ------------------------------------------------------------------ #
# Character-edit constants and helpers
# ------------------------------------------------------------------ #

#: Maximum character position index supported by the vocabulary.
#: Identifiers longer than this can still be partially corrected for
#: positions within range; positions beyond this limit fall back to $UNK.
MAX_CHAR_POS: int = 30

#: Characters that can appear in ``$INS`` and ``$SUB`` tags.
#: Covers lowercase, uppercase, digits, and underscore — the full set
#: of characters valid in Python identifiers (excluding Unicode).
EDIT_ALPHABET: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"

# Tag prefixes for character-edit operations.
_SWAP_PREFIX = "$SWAP_"
_DEL_PREFIX = "$DEL_"
_INS_PREFIX = "$INS_"
_SUB_PREFIX = "$SUB_"
_CASE_PREFIX = "$CASE_"

_CHAR_EDIT_PREFIXES: Tuple[str, ...] = (
    _SWAP_PREFIX, _DEL_PREFIX, _INS_PREFIX, _SUB_PREFIX, _CASE_PREFIX,
)


def is_char_edit_tag(tag: str) -> bool:
    """Return ``True`` if *tag* is a character-edit operation tag."""
    return any(tag.startswith(p) for p in _CHAR_EDIT_PREFIXES)


def _swap_tag(pos: int) -> str:
    return f"$SWAP_{pos}"


def _del_tag(pos: int) -> str:
    return f"$DEL_{pos}"


def _ins_tag(pos: int, char: str) -> str:
    return f"$INS_{pos}_{char}"


def _sub_tag(pos: int, char: str) -> str:
    return f"$SUB_{pos}_{char}"


def _case_tag(pos: int) -> str:
    return f"$CASE_{pos}"


def compute_char_edit_tag(corrupted: str, original: str) -> Optional[str]:
    """Compute the single character-edit tag that transforms *corrupted* → *original*.

    Returns ``None`` if the edit cannot be expressed as a single character
    operation or if positions exceed ``MAX_CHAR_POS``.

    The function tries edits in the following order:
    1. Same length → substitution, case-flip, or adjacent swap
    2. Corrupted is 1 char longer → delete a character (reverses duplication)
    3. Corrupted is 1 char shorter → insert a character (reverses deletion)
    """
    len_c, len_o = len(corrupted), len(original)

    if len_c == len_o:
        # Find differing positions.
        diffs = [i for i in range(len_c) if corrupted[i] != original[i]]

        if len(diffs) == 1:
            pos = diffs[0]
            if pos >= MAX_CHAR_POS:
                return None
            # Case flip?
            if corrupted[pos].lower() == original[pos].lower():
                return _case_tag(pos)
            # Substitution.
            if original[pos] in EDIT_ALPHABET:
                return _sub_tag(pos, original[pos])
            return None

        if len(diffs) == 2:
            i, j = diffs
            # Adjacent swap?
            if j == i + 1 and corrupted[i] == original[j] and corrupted[j] == original[i]:
                if i >= MAX_CHAR_POS:
                    return None
                return _swap_tag(i)
            return None

        return None  # More than 2 diffs — not a single edit.

    if len_c == len_o + 1:
        # Corrupted has one extra character → delete it to get original.
        # Find the position where the extra char was inserted.
        for pos in range(len_c):
            candidate = corrupted[:pos] + corrupted[pos + 1:]
            if candidate == original:
                if pos >= MAX_CHAR_POS:
                    return None
                return _del_tag(pos)
        return None

    if len_c == len_o - 1:
        # Corrupted is missing one character → insert it to get original.
        # Find which character was deleted and where.
        for pos in range(len_o):
            candidate = corrupted[:pos] + original[pos] + corrupted[pos:]
            if candidate == original:
                if pos >= MAX_CHAR_POS:
                    return None
                char = original[pos]
                if char in EDIT_ALPHABET:
                    return _ins_tag(pos, char)
                return None
        return None

    return None  # Length difference > 1 — not a single edit.


def apply_char_edit(token: str, tag: str) -> Optional[str]:
    """Apply a character-edit *tag* to *token* and return the result.

    Returns ``None`` if the tag is malformed or the position is out of
    range for the given token.
    """
    try:
        if tag.startswith(_SWAP_PREFIX):
            pos = int(tag[len(_SWAP_PREFIX):])
            if pos < 0 or pos + 1 >= len(token):
                return None
            chars = list(token)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return "".join(chars)

        if tag.startswith(_DEL_PREFIX):
            pos = int(tag[len(_DEL_PREFIX):])
            if pos < 0 or pos >= len(token):
                return None
            return token[:pos] + token[pos + 1:]

        if tag.startswith(_INS_PREFIX):
            rest = tag[len(_INS_PREFIX):]
            sep = rest.index("_")
            pos = int(rest[:sep])
            char = rest[sep + 1:]
            if pos < 0 or pos > len(token) or len(char) != 1:
                return None
            return token[:pos] + char + token[pos:]

        if tag.startswith(_SUB_PREFIX):
            rest = tag[len(_SUB_PREFIX):]
            sep = rest.index("_")
            pos = int(rest[:sep])
            char = rest[sep + 1:]
            if pos < 0 or pos >= len(token) or len(char) != 1:
                return None
            return token[:pos] + char + token[pos + 1:]

        if tag.startswith(_CASE_PREFIX):
            pos = int(tag[len(_CASE_PREFIX):])
            if pos < 0 or pos >= len(token):
                return None
            chars = list(token)
            chars[pos] = chars[pos].swapcase()
            return "".join(chars)

    except (ValueError, IndexError):
        return None

    return None


def _enumerate_char_edit_tags() -> List[str]:
    """Return all possible character-edit tags in a deterministic order."""
    tags: List[str] = []
    for pos in range(MAX_CHAR_POS):
        tags.append(_swap_tag(pos))
    for pos in range(MAX_CHAR_POS):
        tags.append(_del_tag(pos))
    for pos in range(MAX_CHAR_POS + 1):  # can insert *after* last char
        for char in EDIT_ALPHABET:
            tags.append(_ins_tag(pos, char))
    for pos in range(MAX_CHAR_POS):
        for char in EDIT_ALPHABET:
            tags.append(_sub_tag(pos, char))
    for pos in range(MAX_CHAR_POS):
        tags.append(_case_tag(pos))
    return tags


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

    @property
    def is_char_edit(self) -> bool:
        """``True`` if this vocabulary uses character-edit tags."""
        # A char-edit vocab contains at least one $SWAP_ tag.
        return any(t.startswith(_SWAP_PREFIX) for t in self._tags[:50])

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
    # Tag computation helpers
    # ---------------------------------------------------------------- #

    def tag_for_edit(self, corrupted: str, original: str) -> int:
        """Return the tag index for transforming *corrupted* → *original*.

        In char-edit mode, computes the character-level edit tag.
        In replace mode, returns the ``$REPLACE_<original>`` tag index.
        Falls back to ``UNK_IDX`` if the tag is not in the vocabulary.
        """
        if self.is_char_edit:
            tag = compute_char_edit_tag(corrupted, original)
            if tag is None:
                return UNK_IDX
            return self.tag2idx(tag)
        else:
            return self.tag2idx(replace_tag(original))

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
        """Build a *replace-mode* vocabulary by scanning JSONL dataset files.

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
    def build_char_edit(cls) -> "TagVocab":
        """Build a *character-edit* vocabulary (static, data-independent).

        The vocabulary enumerates all single-character edit operations up
        to :data:`MAX_CHAR_POS` positions and :data:`EDIT_ALPHABET`
        characters.  This vocabulary generalises to any identifier —
        including ones never seen during training.

        Returns
        -------
        TagVocab
        """
        all_tags = _SPECIAL_TAGS + _enumerate_char_edit_tags()
        return cls(all_tags)

    @classmethod
    def minimal(cls) -> "TagVocab":
        """Return a minimal vocabulary with only the three special tags.

        Useful for unit tests and smoke tests that don't need real data.
        """
        return cls(list(_SPECIAL_TAGS))
