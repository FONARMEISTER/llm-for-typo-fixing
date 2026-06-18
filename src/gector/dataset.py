"""PyTorch Dataset for GECToR code-typo training.

Each JSONL sample is converted into a sequence of (input_ids, tag_labels,
detect_labels) tensors aligned to the subword tokenization of the corrupted
code.

Label assignment
----------------
For each code token we assign:

* ``tag_label``    — index into :class:`~src.gector.vocab.TagVocab`.
  - ``KEEP_IDX``   for tokens that are correct.
  - ``vocab.tag2idx("$REPLACE_<original>")`` for corrupted tokens.
  - ``UNK_IDX``    for tokens whose correction is not in the vocabulary.

* ``detect_label`` — binary (0 = correct, 1 = erroneous).

Only the *first subword* of each code token carries a meaningful label;
continuation subwords are masked with ``LABEL_IGNORE = -100`` (PyTorch's
default ignore index for ``CrossEntropyLoss``).

Samples with ``has_errors=false`` are included as negative examples: all
tokens get ``KEEP_IDX`` / ``detect=0``.

Collation
---------
:func:`collate_fn` pads a batch of variable-length sequences to the same
length using the tokenizer's ``pad_token_id`` for ``input_ids`` and
``LABEL_IGNORE`` for labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import torch

from ..typo_datasets import BaseTypoDataset
from .vocab import TagVocab, KEEP_IDX
from .tokenize_code import (
    code_tokens_from_source,
    align_to_subwords,
    first_subword_mask,
)

LABEL_IGNORE: int = -100  # CrossEntropyLoss ignore_index


class GECToRDataset(BaseTypoDataset):
    """Dataset that converts JSONL samples to GECToR training tensors.

    Inherits file loading, ``__len__``, and ``__getitem__`` from
    :class:`~src.typo_datasets.BaseTypoDataset` and implements
    :meth:`_process_item` for the GECToR-specific dual-label format.

    Parameters
    ----------
    paths:
        One or more paths to ``.jsonl`` files.
    vocab:
        Tag vocabulary (built from the same dataset via
        :meth:`~src.gector.vocab.TagVocab.build_from_jsonl`).
    tokenizer:
        A HuggingFace ``PreTrainedTokenizerFast``.
    max_length:
        Maximum subword sequence length (sequences are truncated).
    include_clean:
        If ``True`` (default), include ``has_errors=false`` samples as
        negative examples.  Set to ``False`` to train only on errorful
        samples (not recommended — hurts precision).
    """

    def __init__(
        self,
        paths: Union[str, List[Union[str, Path]]],
        vocab: TagVocab,
        tokenizer,
        max_length: int = 512,
        include_clean: bool = True,
    ) -> None:
        # Normalise paths to plain strings for BaseTypoDataset.
        if isinstance(paths, (str, Path)):
            norm_paths: List[str] = [str(paths)]
        else:
            norm_paths = [str(p) for p in paths]

        # BaseTypoDataset handles all file I/O and populates self.data.
        super().__init__(norm_paths, tokenizer, max_length)

        self.vocab = vocab

        # Optionally drop clean (error-free) samples.
        if not include_clean:
            self.data = [s for s in self.data if s.get("has_errors")]

    # ---------------------------------------------------------------- #
    # Encoding  (called by BaseTypoDataset.__getitem__)
    # ---------------------------------------------------------------- #

    def _process_item(self, sample: dict) -> Dict[str, torch.Tensor]:
        """Convert one JSONL sample to a dict of tensors.

        Returns
        -------
        dict with keys:
            ``input_ids``      — LongTensor [seq_len]
            ``attention_mask`` — LongTensor [seq_len]
            ``tag_labels``     — LongTensor [seq_len]  (LABEL_IGNORE at non-first subwords)
            ``detect_labels``  — LongTensor [seq_len]  (LABEL_IGNORE at non-first subwords)
        """
        code = sample["code"]
        edits: List[dict] = sample.get("edits", [])

        # Build a mapping: corrupted_name → original_name
        fix_map: Dict[str, str] = {
            e["corrupted_name"]: e["original_name"]
            for e in edits
            if e.get("corrupted_name") and e.get("original_name")
        }

        # Tokenize source into code tokens.
        code_toks = code_tokens_from_source(code)
        if not code_toks:
            # Empty / unparseable source — return a minimal dummy tensor.
            return self._empty_item()

        # Subword alignment.
        input_ids, word_ids = align_to_subwords(
            code_toks, self.tokenizer, max_length=self.max_length
        )
        seq_len = len(input_ids)

        # Build per-code-token labels.
        n_code = len(code_toks)
        code_tag_labels = [KEEP_IDX] * n_code
        code_detect_labels = [0] * n_code

        for i, ct in enumerate(code_toks):
            if ct.is_name and ct.text in fix_map:
                orig = fix_map[ct.text]
                code_tag_labels[i] = self.vocab.tag_for_edit(ct.text, orig)
                code_detect_labels[i] = 1

        # Expand to subword level: first subword gets the label, rest get LABEL_IGNORE.
        tag_labels = [LABEL_IGNORE] * seq_len
        detect_labels = [LABEL_IGNORE] * seq_len
        fsw_mask = first_subword_mask(word_ids)

        for pos, (is_first, wid) in enumerate(zip(fsw_mask, word_ids)):
            if is_first and wid is not None and wid < n_code:
                tag_labels[pos] = code_tag_labels[wid]
                detect_labels[pos] = code_detect_labels[wid]

        attention_mask = [1] * seq_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "tag_labels": torch.tensor(tag_labels, dtype=torch.long),
            "detect_labels": torch.tensor(detect_labels, dtype=torch.long),
            # Raw strings kept for sequence-level auxiliary losses in train.py.
            # They are plain Python strings, not tensors — collate_fn handles them.
            "raw_code": sample["code"],
            "raw_fixed": sample.get("fixed") or sample["code"],
        }

    def _empty_item(self) -> Dict[str, torch.Tensor]:
        """Return a minimal valid item for empty/unparseable sources."""
        pad_id = self.tokenizer.pad_token_id or 0
        # [CLS] [SEP]
        cls_id = getattr(self.tokenizer, "cls_token_id", None) or pad_id
        sep_id = getattr(self.tokenizer, "sep_token_id", None) or pad_id
        input_ids = [cls_id, sep_id]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "tag_labels": torch.tensor([LABEL_IGNORE, LABEL_IGNORE], dtype=torch.long),
            "detect_labels": torch.tensor([LABEL_IGNORE, LABEL_IGNORE], dtype=torch.long),
            "raw_code": "",
            "raw_fixed": "",
        }


# ------------------------------------------------------------------ #
# Collation
# ------------------------------------------------------------------ #


def collate_fn(
    batch: List[Dict],
    pad_token_id: int = 1,
) -> Dict:
    """Pad a batch of variable-length items to the same length.

    Parameters
    ----------
    batch:
        List of dicts as returned by :meth:`GECToRDataset.__getitem__`.
    pad_token_id:
        Token ID used to pad ``input_ids`` (should match the tokenizer's
        ``pad_token_id``).

    Returns
    -------
    dict with tensor keys (``input_ids``, ``attention_mask``, ``tag_labels``,
    ``detect_labels``) each padded to ``[batch, max_seq_len]``, plus plain
    list keys ``raw_code`` and ``raw_fixed`` for auxiliary losses.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    padded: Dict[str, List[torch.Tensor]] = {
        "input_ids": [],
        "attention_mask": [],
        "tag_labels": [],
        "detect_labels": [],
    }

    raw_codes: List[str] = []
    raw_fixeds: List[str] = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        padded["input_ids"].append(
            torch.cat([item["input_ids"],
                       torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        padded["attention_mask"].append(
            torch.cat([item["attention_mask"],
                       torch.zeros(pad_len, dtype=torch.long)])
        )
        padded["tag_labels"].append(
            torch.cat([item["tag_labels"],
                       torch.full((pad_len,), LABEL_IGNORE, dtype=torch.long)])
        )
        padded["detect_labels"].append(
            torch.cat([item["detect_labels"],
                       torch.full((pad_len,), LABEL_IGNORE, dtype=torch.long)])
        )
        raw_codes.append(item.get("raw_code", ""))
        raw_fixeds.append(item.get("raw_fixed", ""))

    result = {k: torch.stack(v) for k, v in padded.items()}
    result["raw_code"] = raw_codes    # List[str] — not a tensor
    result["raw_fixed"] = raw_fixeds  # List[str] — not a tensor
    return result


def make_collate_fn(pad_token_id: int):
    """Return a collate function bound to *pad_token_id*."""
    def _collate(batch):
        return collate_fn(batch, pad_token_id=pad_token_id)
    return _collate
