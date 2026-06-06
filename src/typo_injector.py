"""Inject grammar-style typos into source-code identifiers.

A "grammar error" here means a typo in an identifier (variable / function / class name)
that keeps the code syntactically valid Python in *most* cases, but breaks semantics
(NameError at runtime). We do NOT touch keywords, builtins, string literals, comments,
numeric literals or attribute accesses (`obj.attr` — `attr` is left alone because it
typically refers to something defined elsewhere).

The public entry point is :func:`inject_typos`.
"""

from __future__ import annotations

import builtins
import io
import keyword
import random
import string
import tokenize
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple


# Identifiers we never rename: language keywords, soft keywords, builtins, and a few
# names that appear in idiomatic code but are conventionally treated as "fixed".
_PROTECTED_NAMES = (
    set(keyword.kwlist)
    | set(getattr(keyword, "softkwlist", []))
    | set(dir(builtins))
    | {"self", "cls", "__init__", "__name__", "__main__", "__file__", "__doc__"}
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

    @property
    def has_errors(self) -> bool:
        return len(self.edits) > 0 and self.corrupted != self.original


# --------------------------------------------------------------------------- #
# Typo operations
# --------------------------------------------------------------------------- #


def _swap_adjacent(name: str, rng: random.Random) -> Optional[str]:
    if len(name) < 4:  # avoid corrupting too-short names
        return None
    # don't touch the leading underscore / first char (keeps it lexically an identifier)
    # and never swap an underscore with a neighbor (that effectively moves the
    # word boundary — reads as a naming choice, not a typo).
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
    # don't duplicate an underscore (e.g. ``my_var -> my__var`` is suspicious but
    # not a typical typo; keep duplication on letters only).
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


# Note: we intentionally do NOT implement an "underscore drop" op
# (e.g. ``test_list -> testlist``). Removing underscores reads as a
# different naming convention rather than a typo, and adds label noise.


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


# --------------------------------------------------------------------------- #
# Identifier extraction
# --------------------------------------------------------------------------- #


def _extract_renameable_identifiers(source: str) -> Dict[str, int]:
    """Return a mapping ``name -> occurrence_count`` for identifiers we may rename.

    We skip:
      * keywords / builtins / protected names
      * the name immediately following ``.`` (attribute access)
      * the name immediately following ``import`` / ``from`` / ``as`` (module names)
      * single-character names (too risky — high collision rate after typo)
      * dunder names
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        return {}

    skip_next = False  # set True after `.`, `import`, `from`, `as`
    counts: Dict[str, int] = {}

    prev_string: Optional[str] = None
    for tok in tokens:
        if tok.type != tokenize.NAME:
            if tok.type == tokenize.OP and tok.string == ".":
                skip_next = True
            prev_string = tok.string if tok.type == tokenize.OP else None
            continue

        name = tok.string

        if skip_next:
            skip_next = False
            prev_string = name
            continue

        if name in ("import", "from", "as"):
            skip_next = True
            prev_string = name
            continue

        if (
            name in _PROTECTED_NAMES
            or len(name) < 3
            or (name.startswith("__") and name.endswith("__"))
        ):
            prev_string = name
            continue

        counts[name] = counts.get(name, 0) + 1
        prev_string = name

    return counts


def _replace_identifier(source: str, original: str, replacement: str) -> str:
    """Replace every *identifier* occurrence of ``original`` (not in strings / comments
    / attribute access) with ``replacement``.

    We rebuild the source via :func:`tokenize.untokenize` to avoid touching string
    literals and comments.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        return source

    new_tokens: List[tokenize.TokenInfo] = []
    skip_next = False
    for i, tok in enumerate(tokens):
        if tok.type != tokenize.NAME:
            if tok.type == tokenize.OP and tok.string == ".":
                skip_next = True
            new_tokens.append(tok)
            continue

        name = tok.string

        if skip_next:
            skip_next = False
            new_tokens.append(tok)
            continue

        if name in ("import", "from", "as"):
            skip_next = True
            new_tokens.append(tok)
            continue

        if name == original:
            new_tokens.append(tok._replace(string=replacement))
        else:
            new_tokens.append(tok)

    try:
        return tokenize.untokenize(new_tokens)
    except ValueError:
        # untokenize is picky about column/row consistency; fall back to a
        # word-boundary regex replace as a best-effort.
        import re

        pattern = re.compile(rf"\b{original}\b")
        return pattern.sub(replacement, source)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def inject_typos(
    source: str,
    *,
    rng: Optional[random.Random] = None,
    max_edits: int = 2,
    p_edit: float = 0.8,
) -> CorruptionResult:
    """Inject up to ``max_edits`` identifier typos into ``source``.

    Args:
        source: original code snippet.
        rng: optional ``random.Random`` for reproducibility.
        max_edits: upper bound on the number of distinct identifiers to corrupt.
        p_edit: probability of picking each candidate identifier for corruption.
    """
    rng = rng or random.Random()
    candidates = _extract_renameable_identifiers(source)

    if not candidates:
        return CorruptionResult(original=source, corrupted=source, edits=[])

    # pick at most `max_edits` identifiers, biased toward the more frequent ones
    names = list(candidates.keys())
    rng.shuffle(names)

    chosen: List[Tuple[str, str, int]] = []
    used_replacements = set()
    for name in names:
        if len(chosen) >= max_edits:
            break
        if rng.random() > p_edit:
            continue
        typo = make_typo(name, rng)
        if not typo:
            continue
        # avoid collisions with existing identifiers or other typos in this snippet
        if typo in candidates or typo in used_replacements:
            continue
        used_replacements.add(typo)
        chosen.append((name, typo, candidates[name]))

    corrupted = source
    for original, typo, _count in chosen:
        corrupted = _replace_identifier(corrupted, original, typo)

    edits = [
        Edit(original_name=o, corrupted_name=t, num_occurrences=c)
        for o, t, c in chosen
    ]
    return CorruptionResult(original=source, corrupted=corrupted, edits=edits)
