"""Code-aware tokenization and subword alignment for GECToR.

The GECToR tagger operates at the level of *code tokens* (the atomic units
produced by Python's ``tokenize`` module), not raw characters or subwords.
However, the Transformer encoder expects *subword* tokens (BPE/WordPiece).

This module provides:

1. :func:`code_tokens_from_source` — split Python source into a flat list
   of ``CodeToken`` objects (one per Python token, excluding whitespace and
   encoding markers).

2. :func:`align_to_subwords` — given a list of code tokens and a
   HuggingFace tokenizer, produce the subword ``input_ids`` tensor and a
   ``word_ids`` list that maps each subword position back to its code-token
   index (``None`` for special tokens such as ``[CLS]`` / ``[SEP]``).

3. :func:`first_subword_mask` — boolean mask that is ``True`` only at the
   *first* subword of each code token.  The GECToR tag head prediction is
   taken from these positions.

Design notes
------------
* We use Python's built-in ``tokenize`` module so that the token boundaries
  exactly match what the rename infrastructure uses.
* Whitespace, ``NEWLINE``, ``NL``, ``INDENT``, ``DEDENT``, ``ENCODING``,
  ``ENDMARKER`` tokens are dropped — they carry no identifier information.
* ``COMMENT`` tokens are kept as single opaque tokens (the model can learn
  to ``$KEEP`` them).
* If the source cannot be tokenized (``SyntaxError``, ``IndentationError``,
  ``tokenize.TokenError``) we fall back to whitespace splitting.
"""

from __future__ import annotations

import io
import tokenize as _tokenize
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Token types we drop entirely.
_SKIP_TYPES = frozenset({
    _tokenize.NEWLINE,
    _tokenize.NL,
    _tokenize.INDENT,
    _tokenize.DEDENT,
    _tokenize.ENCODING,
    _tokenize.ENDMARKER,
    _tokenize.ERRORTOKEN,
})


@dataclass(frozen=True)
class CodeToken:
    """A single Python source token with its position.

    Attributes
    ----------
    text:
        The raw token string (e.g. ``"calculate"``, ``"("``, ``"42"``).
    row:
        1-based line number of the token start.
    col:
        0-based column of the token start.
    tok_type:
        The ``tokenize`` token-type integer (e.g. ``tokenize.NAME``).
    """

    text: str
    row: int
    col: int
    tok_type: int

    @property
    def is_name(self) -> bool:
        return self.tok_type == _tokenize.NAME


def code_tokens_from_source(source: str) -> List[CodeToken]:
    """Tokenize *source* into a list of :class:`CodeToken` objects.

    Whitespace and structural tokens (``NEWLINE``, ``INDENT``, etc.) are
    dropped.  Returns an empty list if tokenization fails.
    """
    try:
        tokens = list(_tokenize.generate_tokens(io.StringIO(source).readline))
    except (_tokenize.TokenError, IndentationError, SyntaxError):
        # Fall back: split on whitespace, assign dummy positions.
        words = source.split()
        return [
            CodeToken(text=w, row=1, col=i * 10, tok_type=_tokenize.NAME)
            for i, w in enumerate(words)
        ]

    result: List[CodeToken] = []
    for tok in tokens:
        if tok.type in _SKIP_TYPES:
            continue
        if not tok.string:
            continue
        result.append(CodeToken(
            text=tok.string,
            row=tok.start[0],
            col=tok.start[1],
            tok_type=tok.type,
        ))
    return result


# ------------------------------------------------------------------ #
# Subword alignment
# ------------------------------------------------------------------ #


def align_to_subwords(
    code_tokens: List[CodeToken],
    tokenizer,  # transformers PreTrainedTokenizerFast
    max_length: int = 512,
) -> Tuple[List[int], List[Optional[int]]]:
    """Encode *code_tokens* with *tokenizer* and return alignment info.

    Parameters
    ----------
    code_tokens:
        List of :class:`CodeToken` objects from :func:`code_tokens_from_source`.
    tokenizer:
        A HuggingFace ``PreTrainedTokenizerFast`` (must support
        ``is_fast=True`` for ``word_ids()``).
    max_length:
        Maximum total subword sequence length (including special tokens).
        Sequences are truncated to this length.

    Returns
    -------
    input_ids : List[int]
        Subword token IDs including special tokens (``[CLS]``, ``[SEP]``
        or ``<s>``, ``</s>`` depending on the model).
    word_ids : List[Optional[int]]
        For each position in *input_ids*, the index into *code_tokens* of
        the code token that produced this subword, or ``None`` for special
        tokens.
    """
    texts = [ct.text for ct in code_tokens]

    # Tokenize as a list of "words" so the tokenizer tracks word boundaries.
    encoding = tokenizer(
        texts,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids: List[int] = encoding["input_ids"]
    word_ids: List[Optional[int]] = encoding.word_ids(batch_index=0)
    return input_ids, word_ids


def first_subword_mask(word_ids: List[Optional[int]]) -> List[bool]:
    """Return a boolean mask that is ``True`` at the first subword of each word.

    Special-token positions (``word_ids[i] is None``) are ``False``.

    Example
    -------
    >>> word_ids = [None, 0, 0, 1, 2, 2, None]
    >>> first_subword_mask(word_ids)
    [False, True, False, True, True, False, False]
    """
    mask: List[bool] = []
    seen: set = set()
    for wid in word_ids:
        if wid is None:
            mask.append(False)
        elif wid in seen:
            mask.append(False)
        else:
            seen.add(wid)
            mask.append(True)
    return mask


def apply_replace_tags(
    code_tokens: List[CodeToken],
    tags: List[str],
) -> str:
    """Apply a list of edit tags to *code_tokens* and reconstruct source.

    This is a *character-level reconstruction* used during iterative
    inference.  It is intentionally simple: tokens are joined with a
    single space.  The harness then uses libcst rename to apply the actual
    scope-aware refactoring.

    Parameters
    ----------
    code_tokens:
        Original code tokens.
    tags:
        One tag per code token (same length as *code_tokens*).  Tags not
        in ``{$KEEP, $DELETE, $REPLACE_*}`` are treated as ``$KEEP``.

    Returns
    -------
    str
        Reconstructed source with replacements applied.
    """
    from .vocab import is_replace_tag, replacement_token, DELETE_TAG

    assert len(code_tokens) == len(tags), (
        f"len(code_tokens)={len(code_tokens)} != len(tags)={len(tags)}"
    )

    out_tokens: List[str] = []
    for ct, tag in zip(code_tokens, tags):
        if tag == DELETE_TAG:
            continue
        elif is_replace_tag(tag):
            out_tokens.append(replacement_token(tag))
        else:
            out_tokens.append(ct.text)

    return " ".join(out_tokens)


def name_tag_from_edit(corrupted: str, original: str) -> str:
    """Return the tag that transforms *corrupted* into *original*.

    Tries a character-edit tag first (``$SWAP_*``, ``$DEL_*``, etc.).
    Falls back to ``$REPLACE_<original>`` if no single character edit
    can express the transformation.
    """
    from .vocab import compute_char_edit_tag, replace_tag

    char_tag = compute_char_edit_tag(corrupted, original)
    if char_tag is not None:
        return char_tag
    return replace_tag(original)
