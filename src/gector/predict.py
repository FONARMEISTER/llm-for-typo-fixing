"""Iterative inference engine for the GECToR code-typo tagger.

The engine applies the model's predicted edit tags to the source code
iteratively: after each pass the corrected code is fed back into the model
until no more ``$REPLACE`` tags are predicted or the iteration limit is
reached.

The final step maps predicted token replacements back to the
``{corrupted_name: fixed_name}`` dictionary expected by the
:class:`~src.models.base.NameFixer` interface.

Design
------
* Tokenization is done with Python's ``tokenize`` module (code-aware).
* Subword alignment uses the HuggingFace tokenizer stored alongside the
  model checkpoint.
* The model runs on CPU by default; pass ``device`` to use GPU/MPS.
* A *confidence threshold* on the detect head can gate predictions: only
  tokens where ``detect_prob >= min_detect_prob`` are considered for
  replacement.  Set to 0.0 to disable gating.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .tokenize_code import (
    CodeToken,
    code_tokens_from_source,
    align_to_subwords,
    first_subword_mask,
)
from .vocab import (
    KEEP_IDX,
    is_replace_tag, replacement_token,
    is_char_edit_tag, apply_char_edit,
)


# ------------------------------------------------------------------ #
# Core prediction
# ------------------------------------------------------------------ #


def predict_tags_for_source(
    source: str,
    model,           # GECToRModel
    tokenizer,       # HuggingFace PreTrainedTokenizerFast
    device: torch.device,
    max_length: int = 512,
    min_detect_prob: float = 0.0,
) -> Tuple[List[CodeToken], List[str]]:
    """Run one forward pass and return per-code-token tag strings.

    Parameters
    ----------
    source:
        Python source code (possibly containing typos).
    model:
        A :class:`~src.gector.model.GECToRModel` in eval mode.
    tokenizer:
        The HuggingFace tokenizer matching the model's encoder.
    device:
        Torch device to run inference on.
    max_length:
        Maximum subword sequence length (truncation).
    min_detect_prob:
        Minimum detection probability to accept a ``$REPLACE`` prediction.
        Tokens below this threshold are treated as ``$KEEP``.

    Returns
    -------
    code_tokens : List[CodeToken]
        The code tokens extracted from *source*.
    tags : List[str]
        One tag string per code token.
    """
    code_tokens = code_tokens_from_source(source)
    if not code_tokens:
        return [], []

    # Encode.
    input_ids, word_ids = align_to_subwords(code_tokens, tokenizer, max_length=max_length)
    fsw_mask = first_subword_mask(word_ids)

    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask_tensor = torch.ones_like(ids_tensor)

    # Forward pass.
    model.eval()
    with torch.no_grad():
        tag_preds, detect_probs = model.predict_tags(ids_tensor, mask_tensor)

    tag_preds_seq = tag_preds[0].tolist()       # [seq_len]
    detect_probs_seq = detect_probs[0].tolist() # [seq_len]

    # Map subword predictions back to code tokens (first-subword rule).
    n_code = len(code_tokens)
    code_tag_indices = [KEEP_IDX] * n_code
    code_detect_probs = [0.0] * n_code

    for pos, (is_first, wid) in enumerate(zip(fsw_mask, word_ids)):
        if is_first and wid is not None and wid < n_code:
            code_tag_indices[wid] = tag_preds_seq[pos]
            code_detect_probs[wid] = detect_probs_seq[pos]

    # Convert indices to tag strings, applying detect threshold.
    vocab = model.vocab
    tags: List[str] = []
    for i, (tag_idx, det_prob) in enumerate(zip(code_tag_indices, code_detect_probs)):
        tag = vocab.idx2tag(tag_idx)
        # Gate: if detect prob is below threshold, force $KEEP.
        if is_replace_tag(tag) and det_prob < min_detect_prob:
            tag = vocab.idx2tag(KEEP_IDX)
        tags.append(tag)

    return code_tokens, tags


# ------------------------------------------------------------------ #
# Iterative correction
# ------------------------------------------------------------------ #


def iterative_correct(
    source: str,
    model,
    tokenizer,
    device: torch.device,
    max_iter: int = 5,
    max_length: int = 512,
    min_detect_prob: float = 0.0,
) -> Tuple[str, Dict[str, str]]:
    """Apply GECToR iteratively until convergence or *max_iter* passes.

    Each iteration:
    1. Predict tags for the current source.
    2. Apply ``$REPLACE_*`` tags to NAME tokens.
    3. If no changes were made, stop early.

    Parameters
    ----------
    source:
        Original (possibly corrupted) Python source code.
    model:
        :class:`~src.gector.model.GECToRModel` in eval mode.
    tokenizer:
        Matching HuggingFace tokenizer.
    device:
        Torch device.
    max_iter:
        Maximum number of correction passes.
    max_length:
        Maximum subword sequence length.
    min_detect_prob:
        Detection probability threshold (0.0 = disabled).

    Returns
    -------
    corrected_source : str
        Source after applying all predicted corrections.
    name_fixes : Dict[str, str]
        Mapping ``{corrupted_name: fixed_name}`` for all NAME tokens that
        were replaced.  This is the format expected by
        :class:`~src.models.base.NameFixer`.
    """
    current = source
    all_fixes: Dict[str, str] = {}

    for _iteration in range(max_iter):
        code_tokens, tags = predict_tags_for_source(
            current, model, tokenizer, device,
            max_length=max_length,
            min_detect_prob=min_detect_prob,
        )
        if not code_tokens:
            break

        # Collect replacements for NAME tokens only.
        iter_fixes: Dict[str, str] = {}
        for ct, tag in zip(code_tokens, tags):
            if not ct.is_name:
                continue
            new_name: Optional[str] = None
            if is_replace_tag(tag):
                new_name = replacement_token(tag)
            elif is_char_edit_tag(tag):
                new_name = apply_char_edit(ct.text, tag)
            if new_name is not None and new_name != ct.text:
                iter_fixes[ct.text] = new_name

        if not iter_fixes:
            break  # Converged.

        # Apply fixes: simple string replacement of NAME tokens.
        # The harness will later use Jedi for scope-aware renaming.
        new_tokens: List[str] = []
        for ct, tag in zip(code_tokens, tags):
            if ct.is_name:
                if is_replace_tag(tag):
                    new_tokens.append(replacement_token(tag))
                    continue
                if is_char_edit_tag(tag):
                    edited = apply_char_edit(ct.text, tag)
                    if edited is not None:
                        new_tokens.append(edited)
                        continue
            new_tokens.append(ct.text)

        # Reconstruct source preserving original whitespace as much as
        # possible by replacing token text in-place.
        current = _reconstruct_source(source if _iteration == 0 else current,
                                      code_tokens, new_tokens)

        # Accumulate fixes (later iterations may refine earlier ones).
        all_fixes.update(iter_fixes)

    return current, all_fixes


def _reconstruct_source(
    original_source: str,
    code_tokens: List[CodeToken],
    new_token_texts: List[str],
) -> str:
    """Reconstruct source by replacing token texts while preserving whitespace.

    Works by iterating over the original source character by character and
    substituting token text at the known (row, col) positions.

    Parameters
    ----------
    original_source:
        The source that was tokenized to produce *code_tokens*.
    code_tokens:
        Original code tokens (with position info).
    new_token_texts:
        Replacement text for each token (same length as *code_tokens*).
    """
    lines = original_source.splitlines(keepends=True)
    # Pad lines list so 1-based indexing works.
    lines = [""] + lines  # lines[1] = first line

    # Build a list of (row, col, old_text, new_text) substitutions,
    # sorted in reverse order so we can apply them without offset drift.
    subs = []
    for ct, new_text in zip(code_tokens, new_token_texts):
        if new_text != ct.text:
            subs.append((ct.row, ct.col, ct.text, new_text))

    # Apply substitutions in reverse line/col order.
    subs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for row, col, old_text, new_text in subs:
        if row >= len(lines):
            continue
        line = lines[row]
        end = col + len(old_text)
        if line[col:end] == old_text:
            lines[row] = line[:col] + new_text + line[end:]

    return "".join(lines[1:])  # drop the padding empty string


# ------------------------------------------------------------------ #
# Name-fix extraction (for NameFixer interface)
# ------------------------------------------------------------------ #


def extract_name_fixes(
    source: str,
    model,
    tokenizer,
    device: torch.device,
    max_iter: int = 5,
    max_length: int = 512,
    min_detect_prob: float = 0.0,
) -> Dict[str, str]:
    """High-level entry point: return ``{corrupted_name: fixed_name}`` dict.

    This is the function called by :class:`~src.gector.fixer.GECToRFixer`.

    Only NAME tokens are considered; punctuation, operators, literals, and
    comments are ignored.

    Parameters
    ----------
    source:
        Python source code (possibly corrupted).
    model, tokenizer, device:
        Model, tokenizer, and device as returned by
        :func:`~src.gector.fixer.GECToRFixer._load`.
    max_iter:
        Maximum iterative correction passes.
    max_length:
        Maximum subword sequence length.
    min_detect_prob:
        Detection probability threshold.

    Returns
    -------
    Dict[str, str]
        ``{corrupted_name: fixed_name}`` — only entries where the model
        suggests a change.
    """
    _, name_fixes = iterative_correct(
        source, model, tokenizer, device,
        max_iter=max_iter,
        max_length=max_length,
        min_detect_prob=min_detect_prob,
    )
    return name_fixes
