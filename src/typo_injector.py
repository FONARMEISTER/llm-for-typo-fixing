"""Inject textual (grammar) typos into Python source code.

Typos are injected into:
  * Identifiers (variable / function / class / parameter names).
  * Comments.

The key requirement: corrupted code MUST remain semantically equivalent to
the original. We use Jedi_ for scope-aware refactoring, so that renaming a
variable in one scope does not accidentally shadow or clobber a variable
with the same name in another scope.

.. _Jedi: https://jedi.readthedocs.io/en/latest/

The public entry point is :func:`inject_typos`.
"""

from __future__ import annotations

import io
import keyword
import random
import re
import tokenize
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .identifier_utils import (
    _PROTECTED_NAMES,
    apply_jedi_rename,
    extract_renameable_identifiers as _extract_renameable_identifiers,
)

# Rough QWERTY adjacency map used for "fat-finger" substitutions.
_KEYBOARD_NEIGHBORS: Dict[str, str] = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class Edit:
    """A single identifier rename applied to a snippet."""

    original_name: str
    corrupted_name: str
    num_occurrences: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorruptionResult:
    original: str
    corrupted: str
    edits: List[Edit] = field(default_factory=list)
    corrupted_comments: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return (
            len(self.edits) > 0
            or len(self.corrupted_comments) > 0
        ) and self.corrupted != self.original


# --------------------------------------------------------------------------- #
# Typo operations (same for identifiers and comment words)
# --------------------------------------------------------------------------- #


def _swap_adjacent(name: str, rng: random.Random) -> Optional[str]:
    if len(name) < 4:
        return None
    candidates = [
        i for i in range(1, len(name) - 1)
        if name[i] != name[i + 1] and name[i] != "_" and name[i + 1] != "_"
    ]
    if not candidates:
        return None
    i = rng.choice(candidates)
    return name[:i] + name[i + 1] + name[i] + name[i + 2:]


def _duplicate_char(name: str, rng: random.Random) -> Optional[str]:
    if len(name) < 3:
        return None
    candidates = [i for i in range(1, len(name)) if name[i] != "_"]
    if not candidates:
        return None
    i = rng.choice(candidates)
    return name[:i] + name[i] + name[i:]


def _substitute_neighbor(name: str, rng: random.Random) -> Optional[str]:
    if len(name) < 3:
        return None
    idxs = [i for i in range(1, len(name)) if name[i].lower() in _KEYBOARD_NEIGHBORS]
    if not idxs:
        return None
    i = rng.choice(idxs)
    neighbors = _KEYBOARD_NEIGHBORS[name[i].lower()]
    replacement = rng.choice(neighbors)
    if name[i].isupper():
        replacement = replacement.upper()
    return name[:i] + replacement + name[i + 1:]


def _case_flip(name: str, rng: random.Random) -> Optional[str]:
    """E.g. ``my_value`` -> ``my_Value`` or ``MyClass`` -> ``Myclass``."""
    idxs = [i for i, ch in enumerate(name) if ch.isalpha() and i > 0]
    if not idxs:
        return None
    i = rng.choice(idxs)
    ch = name[i]
    flipped = ch.lower() if ch.isupper() else ch.upper()
    if flipped == ch:
        return None
    return name[:i] + flipped + name[i + 1:]


def _delete_char_no_underscore(name: str, rng: random.Random) -> Optional[str]:
    """Like :func:`_delete_char` but never deletes an underscore."""
    if len(name) < 4:
        return None
    candidates = [i for i in range(1, len(name)) if name[i] != "_"]
    if not candidates:
        return None
    i = rng.choice(candidates)
    return name[:i] + name[i + 1:]


_OPERATIONS = (
    _swap_adjacent,
    _delete_char_no_underscore,
    _duplicate_char,
    _substitute_neighbor,
    _case_flip,
)


def make_typo(name: str, rng: random.Random, max_tries: int = 8) -> Optional[str]:
    """Return a typo of ``name``, or ``None`` if we can't produce a valid one."""
    for _ in range(max_tries):
        op = rng.choice(_OPERATIONS)
        candidate = op(name, rng)
        if not candidate or candidate == name:
            continue
        if not _is_valid_identifier(candidate):
            continue
        if candidate in _PROTECTED_NAMES:
            continue
        return candidate
    return None


def _is_valid_identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


# Re-exported from ``identifier_utils`` for backward compatibility.
# _extract_renameable_identifiers is imported above.
# _apply_rename is a thin wrapper over apply_jedi_rename below.


def _apply_rename(source: str, line: int, col: int, new_name: str) -> str:
    """Single-file wrapper around ``apply_jedi_rename``."""
    changed = apply_jedi_rename(source, line, col, new_name)
    if changed:
        return next(iter(changed.values()))
    return source


def _count_name_occurrences(source: str, name: str) -> int:
    """Count how many times ``name`` appears as an identifier token."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return 0
    return sum(1 for tok in tokens if tok.type == tokenize.NAME and tok.string == name)


# --------------------------------------------------------------------------- #
# Comment corruption
# --------------------------------------------------------------------------- #

# Minimal set of English stop-words and common code-like tokens we skip when
# corrupting comment text.  This is a heuristic — we don't want to turn
# ``# TODO: fix bug`` into ``# TODO: fx bgu`` where ``TODO`` is mangled.
_COMMENT_SKIP_WORDS = frozenset({
    "todo", "fixme", "note", "hack", "xxx", "bug", "warning", "attention",
    "yes", "no", "true", "false", "none", "is", "in", "or", "and", "not",
    "if", "else", "elif", "for", "while", "with", "as", "from", "import",
    "def", "class", "return", "pass", "break", "continue", "try", "except",
    "finally", "raise", "yield", "the", "a", "an", "this", "that", "it",
})


def _corrupt_word(word: str, rng: random.Random) -> Optional[str]:
    """Corrupt a single comment word, or return ``None`` if no suitable typo."""
    if len(word) < 3:
        return None
    if word.lower() in _COMMENT_SKIP_WORDS:
        return None
    if not word.isalpha():
        return None
    return make_typo(word, rng)


def _corrupt_comment_text(text: str, rng: random.Random, p_word: float = 0.25) -> str:
    """Corrupt a portion of words inside a single comment line."""
    # Split on word boundaries while preserving non-word chunks (spaces, punctuation).
    parts = re.split(r"(\W+)", text)
    corrupted_parts: List[str] = []
    for part in parts:
        if part and part[0].isalpha():
            if rng.random() < p_word:
                typo = _corrupt_word(part, rng)
                if typo:
                    corrupted_parts.append(typo)
                    continue
        corrupted_parts.append(part)
    return "".join(corrupted_parts)


def _corrupt_comments_in_source(
    source: str, rng: random.Random, p_word: float = 0.25,
) -> Tuple[str, List[str]]:
    """Corrupt words inside comments in ``source``.

    Returns ``(new_source, list_of_corrupted_comment_texts)``.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source, []

    # Build a mapping: (line_number, column) -> new_comment_text.
    comment_map: Dict[Tuple[int, int], str] = {}
    corrupted_texts: List[str] = []
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        new_text = _corrupt_comment_text(tok.string, rng, p_word=p_word)
        if new_text != tok.string:
            key = (tok.start[0], tok.start[1])
            comment_map[key] = new_text
            corrupted_texts.append(new_text)

    if not comment_map:
        return source, []

    # Rebuild source line-by-line, swapping corrupted comments.
    lines = source.splitlines(keepends=True)
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip("\n\r")
        suffix = line[len(stripped):]
        comment_pos = stripped.find("#")
        if comment_pos >= 0:
            key = (i, comment_pos)
            if key in comment_map:
                lines[i - 1] = stripped[:comment_pos] + comment_map[key] + suffix

    return "".join(lines), corrupted_texts


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def inject_typos(
    source: str,
    *,
    rng: Optional[random.Random] = None,
    max_edits: int = 2,
    p_edit: float = 0.8,
    corrupt_comments: bool = True,
    p_comment_word: float = 0.25,
) -> CorruptionResult:
    """Inject textual typos into ``source``.

    Identifier typos are injected via Jedi's scope-aware refactoring so that
    scoped variables (shadowed names in nested functions) are handled
    correctly and the corrupted code stays semantically equivalent.

    Comment typos are injected directly into ``COMMENT`` tokens by corrupting
    random words inside the comment text.

    Args:
        source: Original Python source code.
        rng: Optional ``random.Random`` for reproducibility.
        max_edits: Maximum number of **distinct identifier names** to corrupt.
        p_edit: Probability of picking each candidate identifier name.
        corrupt_comments: If True, also corrupt words inside comments.
        p_comment_word: Probability of corrupting an individual word in a comment.

    Returns:
        :class:`CorruptionResult` with the original and corrupted code plus
        metadata about what was changed.
    """
    rng = rng or random.Random()

    # ---- Identifier corruption ----
    try:
        def_positions = _extract_renameable_identifiers(source)
    except Exception:
        # Jedi internal error (parser cache corruption, type-inference
        # edge case in third-party imports, etc.) — return clean.
        return CorruptionResult(
            original=source,
            corrupted=source,
            edits=[],
            corrupted_comments=[],
        )

    # Group names: pick which NAMES to corrupt (not individual positions).
    candidate_names = sorted(def_positions.keys())
    rng.shuffle(candidate_names)

    chosen_renames: List[Tuple[str, str, List[Tuple[int, int]]]] = []
    #  (original_name, typo_name, [(line, col)])  — only one position per name.
    used_typos: set = set()
    for name in candidate_names:
        if len(chosen_renames) >= max_edits:
            break
        if rng.random() > p_edit:
            continue
        typo = make_typo(name, rng)
        if not typo:
            continue
        if typo in def_positions or typo in used_typos:
            continue
        used_typos.add(typo)
        # If the name is defined in multiple scopes, pick one at random — a human
        # might typo the name in one place but not in another.
        pos = rng.choice(def_positions[name])
        chosen_renames.append((name, typo, [pos]))

    # Apply renames one by one, re-extracting positions before each
    # rename to avoid stale coordinates (an earlier rename may shift
    # columns for identifiers on the same line).  Jedi internal errors
    # (e.g. on type-inference edge cases) are caught and the offending
    # rename is skipped.
    corrupted = source
    applied_names: set = set()
    for original_name, typo_name, _unused_positions in chosen_renames:
        # Re-extract from the current source to get correct coordinates.
        # Jedi may fail on already-corrupted code (internal cache bugs,
        # type-inference edge cases) — skip the rename if it does.
        try:
            fresh = _extract_renameable_identifiers(corrupted)
        except Exception:
            continue
        if original_name not in fresh:
            continue
        line, col = rng.choice(fresh[original_name])
        try:
            corrupted = _apply_rename(corrupted, line, col, typo_name)
        except Exception:
            continue
        applied_names.add(original_name)

    # ---- Comment corruption ----
    corrupted_comments: List[str] = []
    if corrupt_comments:
        corrupted, corrupted_comments = _corrupt_comments_in_source(
            corrupted, rng, p_word=p_comment_word,
        )

    # ---- Build result ----
    edits = []
    for original_name, typo_name, positions in chosen_renames:
        if original_name not in applied_names:
            continue
        try:
            count = _count_name_occurrences(source, original_name)
        except Exception:
            count = len(positions)
        edits.append(
            Edit(
                original_name=original_name,
                corrupted_name=typo_name,
                num_occurrences=count,
            )
        )

    return CorruptionResult(
        original=source,
        corrupted=corrupted,
        edits=edits,
        corrupted_comments=corrupted_comments,
    )
