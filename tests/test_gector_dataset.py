"""Unit tests for :class:`~src.gector.dataset.GECToRDataset`.

Tests cover:
- JSONL loading and ``include_clean`` filtering
- Output keys and tensor shapes
- ``LABEL_IGNORE`` (-100) placement at special-token positions
- ``KEEP_IDX`` / ``detect=0`` assignment for clean samples
- Correct char-edit tag index and ``detect=1`` for corrupted tokens
- Multi-file path merging
- Empty / unparseable source safety
- Both char-edit and replace vocabulary modes
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

import torch

from tests.tokenizer_cache import get_codebert_tokenizer

from src.gector.dataset import GECToRDataset, LABEL_IGNORE
from src.gector.vocab import (
    TagVocab, KEEP_IDX, UNK_IDX,
    replace_tag, compute_char_edit_tag,
)


# ------------------------------------------------------------------ #
# Shared fixtures
# ------------------------------------------------------------------ #

_CORRUPTED_SAMPLE: Dict[str, Any] = {
    "code": "def calcluate(x):\n    return x",
    "fixed": "def calculate(x):\n    return x",
    "has_errors": True,
    "edits": [{"corrupted_name": "calcluate", "original_name": "calculate"}],
}

_CLEAN_SAMPLE: Dict[str, Any] = {
    "code": "def calculate(x):\n    return x",
    "fixed": "def calculate(x):\n    return x",
    "has_errors": False,
    "edits": [],
}

_EMPTY_SAMPLE: Dict[str, Any] = {
    "code": "",
    "fixed": "",
    "has_errors": False,
    "edits": [],
}

MOCK_DATA: List[Dict[str, Any]] = [_CORRUPTED_SAMPLE, _CLEAN_SAMPLE]


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #

def _write_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# ------------------------------------------------------------------ #
# Tests (char-edit vocabulary — default)
# ------------------------------------------------------------------ #

class GECToRDatasetTests(unittest.TestCase):
    """Tests for GECToRDataset using a char-edit vocabulary."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = get_codebert_tokenizer()

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.mock_file = os.path.join(cls.temp_dir.name, "mock_gector.jsonl")
        _write_jsonl(cls.mock_file, MOCK_DATA)

        # Use the char-edit vocabulary (static, data-independent).
        cls.vocab = TagVocab.build_char_edit()

        cls.dataset = GECToRDataset(
            [cls.mock_file], cls.vocab, cls.tokenizer, max_length=64
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    # ---- Loading --------------------------------------------------------

    def test_dataset_length(self) -> None:
        """Both samples (corrupted + clean) are loaded."""
        self.assertEqual(len(self.dataset), 2)

    def test_output_keys(self) -> None:
        """Each item contains the four expected keys."""
        sample = self.dataset[0]
        self.assertIn("input_ids", sample)
        self.assertIn("attention_mask", sample)
        self.assertIn("tag_labels", sample)
        self.assertIn("detect_labels", sample)

    # ---- Tensor shapes --------------------------------------------------

    def test_tensor_shapes_1d(self) -> None:
        """All output tensors are 1-D and share the same length."""
        sample = self.dataset[0]
        seq_len = sample["input_ids"].shape[0]
        self.assertEqual(sample["input_ids"].dim(), 1)
        self.assertEqual(sample["attention_mask"].dim(), 1)
        self.assertEqual(sample["tag_labels"].dim(), 1)
        self.assertEqual(sample["detect_labels"].dim(), 1)
        self.assertEqual(sample["attention_mask"].shape[0], seq_len)
        self.assertEqual(sample["tag_labels"].shape[0], seq_len)
        self.assertEqual(sample["detect_labels"].shape[0], seq_len)

    def test_tensor_dtypes_long(self) -> None:
        """All tensors are integer (LongTensor)."""
        sample = self.dataset[0]
        for key in ("input_ids", "attention_mask", "tag_labels", "detect_labels"):
            self.assertEqual(sample[key].dtype, torch.long, f"{key} should be long")

    # ---- LABEL_IGNORE placement -----------------------------------------

    def test_special_tokens_ignored(self) -> None:
        """[CLS] and [SEP] positions must have LABEL_IGNORE in both label tensors."""
        sample = self.dataset[0]
        input_ids = sample["input_ids"]
        tag_labels = sample["tag_labels"]
        detect_labels = sample["detect_labels"]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        for pos, tok_id in enumerate(input_ids.tolist()):
            if tok_id in (cls_id, sep_id):
                self.assertEqual(
                    tag_labels[pos].item(), LABEL_IGNORE,
                    f"Position {pos} (special token) must have tag_label=LABEL_IGNORE"
                )
                self.assertEqual(
                    detect_labels[pos].item(), LABEL_IGNORE,
                    f"Position {pos} (special token) must have detect_label=LABEL_IGNORE"
                )

    def test_label_ignore_present(self) -> None:
        """LABEL_IGNORE (-100) appears in both label tensors (continuation subwords)."""
        sample = self.dataset[0]
        self.assertTrue((sample["tag_labels"] == LABEL_IGNORE).any().item())
        self.assertTrue((sample["detect_labels"] == LABEL_IGNORE).any().item())

    # ---- Clean sample ---------------------------------------------------

    def test_clean_sample_all_keep(self) -> None:
        """Clean sample: every non-ignored position has KEEP_IDX and detect=0."""
        sample = self.dataset[1]  # clean sample
        tag_labels = sample["tag_labels"]
        detect_labels = sample["detect_labels"]

        valid_tag = tag_labels[tag_labels != LABEL_IGNORE]
        valid_detect = detect_labels[detect_labels != LABEL_IGNORE]

        self.assertTrue(
            (valid_tag == KEEP_IDX).all().item(),
            "Clean sample must have only KEEP_IDX in tag_labels"
        )
        self.assertTrue(
            (valid_detect == 0).all().item(),
            "Clean sample must have only 0 in detect_labels"
        )

    # ---- Corrupted sample -----------------------------------------------

    def test_corrupted_sample_has_error_token(self) -> None:
        """Corrupted sample: at least one position has detect_label=1."""
        sample = self.dataset[0]
        valid_detect = sample["detect_labels"][sample["detect_labels"] != LABEL_IGNORE]
        self.assertTrue(
            (valid_detect == 1).any().item(),
            "Corrupted sample must have at least one detect_label=1"
        )

    def test_corrupted_sample_char_edit_tag_index(self) -> None:
        """The corrupted token's first subword gets the correct char-edit tag index.

        'calcluate' → 'calculate': positions 4 and 5 are swapped (l↔u), so
        the expected tag is $SWAP_4.
        """
        expected_tag = compute_char_edit_tag("calcluate", "calculate")
        self.assertIsNotNone(expected_tag, "Should compute a char-edit tag for calcluate→calculate")
        self.assertEqual(expected_tag, "$SWAP_4")

        expected_tag_idx = self.vocab.tag2idx(expected_tag)
        self.assertNotEqual(expected_tag_idx, UNK_IDX,
                            "$SWAP_4 should be in the char-edit vocabulary")

        sample = self.dataset[0]
        tag_labels = sample["tag_labels"]
        self.assertTrue(
            (tag_labels == expected_tag_idx).any().item(),
            f"Expected tag index {expected_tag_idx} ($SWAP_4) in tag_labels"
        )

    def test_error_count_matches_edits(self) -> None:
        """Number of detect=1 positions equals the number of corrupted code tokens."""
        sample = self.dataset[0]
        valid_detect = sample["detect_labels"][sample["detect_labels"] != LABEL_IGNORE]
        n_errors = int((valid_detect == 1).sum().item())
        # "calcluate" is a single token → exactly 1 error position expected.
        self.assertEqual(n_errors, 1)

    # ---- include_clean --------------------------------------------------

    def test_include_clean_false_drops_clean(self) -> None:
        """include_clean=False keeps only errorful samples."""
        ds = GECToRDataset(
            [self.mock_file], self.vocab, self.tokenizer,
            max_length=64, include_clean=False
        )
        self.assertEqual(len(ds), 1)

    def test_include_clean_true_keeps_all(self) -> None:
        """include_clean=True (default) keeps all samples including clean ones."""
        ds = GECToRDataset(
            [self.mock_file], self.vocab, self.tokenizer,
            max_length=64, include_clean=True
        )
        self.assertEqual(len(ds), 2)

    # ---- Multi-file loading ---------------------------------------------

    def test_multi_file_merging(self) -> None:
        """Passing two JSONL files concatenates their samples."""
        extra_file = os.path.join(self.temp_dir.name, "extra.jsonl")
        _write_jsonl(extra_file, [_CLEAN_SAMPLE])

        ds = GECToRDataset(
            [self.mock_file, extra_file], self.vocab, self.tokenizer, max_length=64
        )
        self.assertEqual(len(ds), 3)  # 2 from mock + 1 from extra

    # ---- Edge cases -----------------------------------------------------

    def test_empty_source_no_crash(self) -> None:
        """Empty source code returns a minimal valid item without raising."""
        empty_file = os.path.join(self.temp_dir.name, "empty.jsonl")
        _write_jsonl(empty_file, [_EMPTY_SAMPLE])

        ds = GECToRDataset(
            [empty_file], self.vocab, self.tokenizer, max_length=64
        )
        sample = ds[0]
        self.assertIn("input_ids", sample)
        self.assertGreater(sample["input_ids"].shape[0], 0)

    def test_string_path_accepted(self) -> None:
        """A single string path (not a list) is accepted without error."""
        ds = GECToRDataset(
            self.mock_file, self.vocab, self.tokenizer, max_length=64
        )
        self.assertEqual(len(ds), 2)

    def test_max_length_truncation(self) -> None:
        """Sequences are truncated to max_length."""
        ds = GECToRDataset(
            [self.mock_file], self.vocab, self.tokenizer, max_length=8
        )
        sample = ds[0]
        self.assertLessEqual(sample["input_ids"].shape[0], 8)


# ------------------------------------------------------------------ #
# Tests (replace vocabulary — legacy mode)
# ------------------------------------------------------------------ #

class GECToRDatasetReplaceVocabTests(unittest.TestCase):
    """Tests for GECToRDataset using the legacy replace vocabulary."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = get_codebert_tokenizer()

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.mock_file = os.path.join(cls.temp_dir.name, "mock_gector.jsonl")
        _write_jsonl(cls.mock_file, MOCK_DATA)

        # Use the replace vocabulary (data-derived).
        cls.vocab = TagVocab.build_from_jsonl([cls.mock_file])

        cls.dataset = GECToRDataset(
            [cls.mock_file], cls.vocab, cls.tokenizer, max_length=64
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_corrupted_sample_replace_tag_index(self) -> None:
        """The corrupted token gets $REPLACE_calculate tag index."""
        expected_tag_idx = self.vocab.tag2idx(replace_tag("calculate"))
        sample = self.dataset[0]
        tag_labels = sample["tag_labels"]
        self.assertTrue(
            (tag_labels == expected_tag_idx).any().item(),
            f"Expected tag index {expected_tag_idx} ($REPLACE_calculate) in tag_labels"
        )

    def test_clean_sample_all_keep(self) -> None:
        """Clean sample: every non-ignored position has KEEP_IDX."""
        sample = self.dataset[1]
        tag_labels = sample["tag_labels"]
        valid_tag = tag_labels[tag_labels != LABEL_IGNORE]
        self.assertTrue(
            (valid_tag == KEEP_IDX).all().item(),
            "Clean sample must have only KEEP_IDX in tag_labels"
        )


if __name__ == "__main__":
    unittest.main()
