"""Unit tests for :class:`~src.gector.model.GECToRModel`.

Tests cover:
- Construction from encoder.
- Forward pass (training mode) returns loss tensors.
- Forward pass (inference mode) returns logits.
- ``predict_tags`` returns correct shapes.
- ``save_pretrained`` / ``from_pretrained`` roundtrip.
- ``from_encoder`` and ``from_pretrained`` constructors.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Dict

import torch

from src.gector.model import GECToRModel, GECToRConfig
from src.gector.vocab import TagVocab, KEEP_IDX
from src.gector.dataset import LABEL_IGNORE
from src.gector.tokenize_code import (
    code_tokens_from_source,
    align_to_subwords,
    first_subword_mask,
)

from tests.tokenizer_cache import get_codebert_tokenizer

try:
    HAVE_TOKENIZER = True
except ImportError:
    HAVE_TOKENIZER = False


# ------------------------------------------------------------------ #
# Test helper
# ------------------------------------------------------------------ #

def _make_dummy_input(
    tokenizer,
    source: str = "def foo(x):\n    return x",
    max_length: int = 64,
) -> Dict[str, torch.Tensor]:
    """Build a batch of one with dummy tag/detect labels (all KEEP)."""
    code_tokens = code_tokens_from_source(source)
    input_ids, word_ids = align_to_subwords(code_tokens, tokenizer, max_length=max_length)
    fsw_mask = first_subword_mask(word_ids)
    seq_len = len(input_ids)

    # All KEEP for simplicity.
    tag_labels = [LABEL_IGNORE] * seq_len
    detect_labels = [LABEL_IGNORE] * seq_len
    for pos, (is_first, _wid) in enumerate(zip(fsw_mask, word_ids)):
        if is_first:
            tag_labels[pos] = KEEP_IDX
            detect_labels[pos] = 0

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "tag_labels": torch.tensor([tag_labels], dtype=torch.long),
        "detect_labels": torch.tensor([detect_labels], dtype=torch.long),
    }


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

@unittest.skipUnless(HAVE_TOKENIZER, "transformers not available")
class GECToRModelTests(unittest.TestCase):
    """Tests for :class:`GECToRModel` construction and forward pass."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.vocab = TagVocab.build_char_edit()
        cls.tokenizer = get_codebert_tokenizer()
        # Use a tiny encoder for speed.
        cls.model = GECToRModel.from_encoder(
            "microsoft/codebert-base", cls.vocab,
            hidden_dropout=0.1, detect_weight=0.5,
        )
        cls.model = cls.model.to("cpu")

    def test_model_device(self) -> None:
        self.assertEqual(
            next(self.model.parameters()).device.type, "cpu"
        )

    def test_forward_with_labels_returns_loss(self) -> None:
        batch = _make_dummy_input(self.tokenizer)
        out = self.model(**batch)
        self.assertIn("loss", out)
        self.assertIn("tag_loss", out)
        self.assertIn("detect_loss", out)
        self.assertIsInstance(out["loss"].item(), float)

    def test_forward_without_labels_returns_logits_only(self) -> None:
        batch = _make_dummy_input(self.tokenizer)
        batch.pop("tag_labels")
        batch.pop("detect_labels")
        out = self.model(**batch)
        self.assertNotIn("loss", out)
        self.assertIn("tag_logits", out)
        self.assertIn("detect_logits", out)

    def test_tag_logits_shape(self) -> None:
        batch = _make_dummy_input(self.tokenizer)
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        B, L, V = out["tag_logits"].shape
        self.assertEqual(B, 1)
        self.assertEqual(V, self.vocab.size)

    def test_detect_logits_shape(self) -> None:
        batch = _make_dummy_input(self.tokenizer)
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        B, L, C = out["detect_logits"].shape
        self.assertEqual(B, 1)
        self.assertEqual(C, 2)

    def test_predict_tags(self) -> None:
        batch = _make_dummy_input(self.tokenizer)
        tag_preds, detect_probs = self.model.predict_tags(
            batch["input_ids"], batch["attention_mask"]
        )
        self.assertEqual(tag_preds.dim(), 2)
        self.assertEqual(tag_preds.shape, batch["input_ids"].shape)
        self.assertEqual(detect_probs.dim(), 2)

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "test_model")
            self.model.save_pretrained(ckpt)

            # Verify files exist.
            self.assertTrue(os.path.isfile(os.path.join(ckpt, "gector_config.json")))
            self.assertTrue(os.path.isfile(os.path.join(ckpt, "vocab.txt")))
            self.assertTrue(os.path.isfile(os.path.join(ckpt, "pytorch_model_full.bin")))
            self.assertTrue(os.path.isfile(os.path.join(ckpt, "config.json")))  # HF encoder config

            loaded = GECToRModel.from_pretrained(ckpt)
            self.assertEqual(loaded.vocab.size, self.vocab.size)

            # Forward pass with loaded model should produce same shape logits.
            batch = _make_dummy_input(self.tokenizer)
            out = loaded(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            self.assertEqual(out["tag_logits"].shape[-1], self.vocab.size)


@unittest.skipUnless(HAVE_TOKENIZER, "transformers not available")
class GECToRConfigTests(unittest.TestCase):
    """Tests for :class:`GECToRConfig` save/load."""

    def test_save_load(self) -> None:
        config = GECToRConfig(
            encoder_model="roberta-base",
            vocab_size=100,
            hidden_dropout=0.2,
            detect_weight=0.7,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_config.json")
            config.save(path)
            loaded = GECToRConfig.load(path)
            self.assertEqual(loaded.encoder_model, "roberta-base")
            self.assertEqual(loaded.vocab_size, 100)
            self.assertEqual(loaded.hidden_dropout, 0.2)
            self.assertEqual(loaded.detect_weight, 0.7)


if __name__ == "__main__":
    unittest.main()
