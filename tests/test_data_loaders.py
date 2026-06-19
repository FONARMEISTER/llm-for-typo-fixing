"""Unit tests for dataset loading and preprocessing logic.

This module uses the built-in unittest framework to verify that the
data loading pipelines correctly parse JSONL files and transform source
code into model-ready tensors for Seq2Seq, Causal LM, and Masked LM architectures.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from tests.tokenizer_cache import get_codebert_tokenizer

from src.typo_datasets import (
    CausalLMTypoDataset,
    MaskedLMTypoDataset,
    Seq2SeqTypoDataset,
)

# A shared list of dictionaries mimicking the output of the typo_injector pipeline.
MOCK_DATA: List[Dict[str, Any]] = [
    # Case 1: Corrupted snippet requiring a fix
    {
        "code": "def calc_sum(a, b):\n    return a + b",
        "has_errors": True,
        "fixed": "def calculate_sum(a, b):\n    return a + b",
        "edits": [{"original_name": "calculate_sum", "corrupted_name": "calc_sum", "num_occurrences": 1}]
    },
    # Case 2: Clean snippet (fixed is null/missing)
    {
        "code": "class ValidClass:\n    pass",
        "has_errors": False,
        "fixed": None,
        "edits": []
    }
]




class Seq2SeqTypoDatasetTests(unittest.TestCase):
    """Test suite for the Seq2Seq dataset loader (e.g., CodeT5, BART).
    
    Validates the correct parsing of JSONL files and the transformation of
    source code into input and label tensors suitable for Encoder-Decoder models.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up class-level resources before any tests are run.
        
        Initializes the CodeT5 tokenizer, creates a temporary JSONL file
        with mock data, and instantiates the dataset.
        """
        # NOTE: explicitly passing additional_special_tokens=[] to bypass the 
        # TokenizersBackend TypeError bug present in older HF model configs.
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/codet5-small", 
            additional_special_tokens=[]
        )
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.mock_file = os.path.join(cls.temp_dir.name, "mock_seq2seq.jsonl")

        with open(cls.mock_file, "w", encoding="utf-8") as f:
            for item in MOCK_DATA:
                f.write(json.dumps(item) + "\n")

        cls.dataset = Seq2SeqTypoDataset(cls.mock_file, cls.tokenizer, max_length=64)

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleans up class-level resources after all tests have completed.
        
        Securely deletes the temporary directory and all files within it.
        """
        cls.temp_dir.cleanup()

    def test_dataset_length(self) -> None:
        """Tests if the dataset correctly parses the total number of items."""
        self.assertEqual(len(self.dataset), 2, "Dataset should contain exactly 2 mock items.")

    def test_corrupted_sample_tensors(self) -> None:
        """Tests the tensor generation and padding masking for a corrupted code sample.
        
        Ensures that generated tensors match the specified max_length and that
        the padding tokens in the labels are replaced with -100 to prevent the
        model from computing loss on padding.
        """
        sample = self.dataset[0]
        
        # Verify tensor dimensions
        self.assertEqual(sample["input_ids"].shape, torch.Size([64]))
        self.assertEqual(sample["attention_mask"].shape, torch.Size([64]))
        self.assertEqual(sample["labels"].shape, torch.Size([64]))
        
        # Verify padding logic
        has_padding_id = (sample["labels"] == self.tokenizer.pad_token_id).any().item()
        self.assertFalse(has_padding_id, "Labels must not contain the original pad_token_id.")
        
        has_ignore_index = (sample["labels"] == -100).any().item()
        self.assertTrue(has_ignore_index, "Expected -100 masking tokens in labels.")

    def test_corrupted_sample_decoding(self) -> None:
        """Tests if the model target (labels) correctly reconstructs the fixed code."""
        sample = self.dataset[0]
        valid_labels = sample["labels"][sample["labels"] != -100]
        target_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)
        self.assertIn("calculate_sum", target_text, "Target text must contain the fixed identifier.")


class CausalLMTypoDatasetTests(unittest.TestCase):
    """Test suite for the Causal LM dataset loader (e.g., GPT, Qwen, CodeLlama).
    
    Validates instruction-tuning formatting and prompt masking techniques
    used for Decoder-only architectures.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up class-level resources before any tests are run.
        
        Initializes a GPT-2 tokenizer (as a proxy for Causal LMs), assigns a
        pad token, creates a temporary file, and loads the dataset.
        """
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.mock_file = os.path.join(cls.temp_dir.name, "mock_causal.jsonl")

        with open(cls.mock_file, "w", encoding="utf-8") as f:
            for item in MOCK_DATA:
                f.write(json.dumps(item) + "\n")

        cls.dataset = CausalLMTypoDataset(cls.mock_file, cls.tokenizer, max_length=64)

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleans up class-level resources after all tests have completed."""
        cls.temp_dir.cleanup()

    def test_tensors_shape(self) -> None:
        """Tests that all generated tensors have the expected dimensions."""
        sample = self.dataset[0]
        self.assertEqual(sample["input_ids"].shape, torch.Size([64]))
        self.assertEqual(sample["attention_mask"].shape, torch.Size([64]))
        self.assertEqual(sample["labels"].shape, torch.Size([64]))

    def test_prompt_masking_logic(self) -> None:
        """Critically tests that the instructional prompt is masked with -100.
        
        Ensures that the Causal LM loss function is only applied to the
        generation of the fixed code, ignoring the prompt.
        """
        sample = self.dataset[0]
        valid_labels = sample["labels"][sample["labels"] != -100]
        target_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)
        
        self.assertNotIn("Fix grammar", target_text, "Prompt was not masked out with -100.")
        self.assertIn("calculate_sum", target_text, "Fixed code is missing from unmasked labels.")

    def test_clean_sample_masking(self) -> None:
        """Tests if the masking works correctly even when there are no errors."""
        sample = self.dataset[1]
        valid_labels = sample["labels"][sample["labels"] != -100]
        target_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)
        
        self.assertNotIn("Fix grammar", target_text)
        self.assertIn("ValidClass", target_text)


class MaskedLMTypoDatasetTests(unittest.TestCase):
    """Test suite for the Masked LM dataset loader (e.g., CodeBERT).
    
    Validates the correct injection of [MASK] tokens at corrupted positions
    and the alignment of reconstruction targets.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up class-level resources before any tests are run.
        
        Initializes the CodeBERT tokenizer, creates a temporary file,
        and loads the dataset.
        """
        cls.tokenizer = get_codebert_tokenizer()
        
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.mock_file = os.path.join(cls.temp_dir.name, "mock_masked.jsonl")

        with open(cls.mock_file, "w", encoding="utf-8") as f:
            for item in MOCK_DATA:
                f.write(json.dumps(item) + "\n")

        cls.dataset = MaskedLMTypoDataset(cls.mock_file, cls.tokenizer, max_length=64)

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleans up class-level resources after all tests have completed."""
        cls.temp_dir.cleanup()

    def test_tensors_shape(self) -> None:
        """Tests that all generated tensors have the expected dimensions."""
        sample = self.dataset[0]
        self.assertEqual(sample["input_ids"].shape, torch.Size([64]))
        self.assertEqual(sample["attention_mask"].shape, torch.Size([64]))
        self.assertEqual(sample["labels"].shape, torch.Size([64]))

    def test_mask_injection_corrupted_sample(self) -> None:
        """Tests if [MASK] tokens are correctly injected into input_ids.
        
        Validates that corrupted identifiers are replaced with the mask token
        in the input, but are absent from the reconstruction target (labels).
        """
        sample = self.dataset[0]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        mask_token_id = self.tokenizer.mask_token_id
        
        has_mask_in_input = (input_ids == mask_token_id).any().item()
        self.assertTrue(has_mask_in_input, "Input IDs must contain the mask token for corrupted samples.")

        has_mask_in_labels = (labels == mask_token_id).any().item()
        self.assertFalse(has_mask_in_labels, "Labels must not contain the mask token.")

    def test_clean_sample_no_masks(self) -> None:
        """Tests if clean samples are left completely untouched.
        
        Validates that no [MASK] tokens are erroneously injected when there
        are no grammar errors present in the snippet.
        """
        sample = self.dataset[1]
        input_ids = sample["input_ids"]
        mask_token_id = self.tokenizer.mask_token_id
        
        has_mask = (input_ids == mask_token_id).any().item()
        self.assertFalse(has_mask, "Clean samples should not contain any mask tokens.")

    def test_labels_padding_masking(self) -> None:
        """Tests if padding tokens in labels are properly replaced with -100."""
        sample = self.dataset[0]
        has_ignore_index = (sample["labels"] == -100).any().item()
        self.assertTrue(has_ignore_index, "Labels must use -100 for padding.")
        
        valid_labels = sample["labels"][sample["labels"] != -100]
        target_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)
        self.assertIn("calculate_sum", target_text)

if __name__ == "__main__":
    unittest.main()