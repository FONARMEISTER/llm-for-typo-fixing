"""Unit tests for dataset loading and preprocessing logic.

This module uses the built-in unittest framework to verify that the
data loading pipelines correctly parse JSONL files and transform source
code into model-ready tensors.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch
from transformers import AutoTokenizer

from src.typo_datasets import Seq2SeqTypoDataset


class Seq2SeqTypoDatasetTests(unittest.TestCase):
    """Test suite for the Seq2Seq dataset loader.
    
    Validates the correct parsing of JSONL files and the transformation of
    source code into input and label tensors suitable for Hugging Face models.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up class-level resources before any tests are run.
        
        Initializes a small tokenizer for fast execution, creates a temporary
        JSONL file mimicking the data pipeline output, and loads the dataset.
        """
        # We use a small tokenizer for speed.
        # NOTE: explicitly passing additional_special_tokens=[] to bypass the 
        # TokenizersBackend TypeError bug present in older HF model configs.
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/codet5-small", 
            additional_special_tokens=[]
        )

        # Create a temporary directory and file
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.mock_file_path = os.path.join(cls.temp_dir.name, "mock_dataset.jsonl")

        # Define edge-case data snippets
        mock_data = [
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

        # Write mock data to the temporary file
        with open(cls.mock_file_path, "w", encoding="utf-8") as f:
            for item in mock_data:
                f.write(json.dumps(item) + "\n")
                
        # Initialize the dataset once for all test methods
        cls.dataset = Seq2SeqTypoDataset(
            jsonl_paths=cls.mock_file_path,
            tokenizer=cls.tokenizer,
            max_length=64
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleans up class-level resources after all tests have completed."""
        # This securely deletes the temporary directory and all files within it
        cls.temp_dir.cleanup()

    def test_dataset_length(self) -> None:
        """Tests if the dataset correctly parses the total number of items."""
        self.assertEqual(len(self.dataset), 2, "Dataset should contain exactly 2 mock items.")

    def test_corrupted_sample_tensors(self) -> None:
        """Tests the tensor generation and masking for a corrupted code sample."""
        sample = self.dataset[0]

        # 1. Assert tensor shapes
        expected_shape = torch.Size([64])
        self.assertEqual(sample["input_ids"].shape, expected_shape)
        self.assertEqual(sample["attention_mask"].shape, expected_shape)
        self.assertEqual(sample["labels"].shape, expected_shape)

        # 2. Assert padding tokens are correctly replaced with -100 in labels
        pad_token_id = self.tokenizer.pad_token_id
        
        # Ensure no label equals the pad_token_id
        has_padding_id = (sample["labels"] == pad_token_id).any().item()
        self.assertFalse(has_padding_id, "Labels must not contain the original pad_token_id.")
        
        # Ensure -100 is present (since our code is way shorter than max_length=64)
        has_ignore_index = (sample["labels"] == -100).any().item()
        self.assertTrue(has_ignore_index, "Expected -100 masking tokens in labels.")

    def test_corrupted_sample_decoding(self) -> None:
        """Tests if the model target (labels) reconstructs the fixed code."""
        sample = self.dataset[0]

        # Filter out the -100 values to allow the tokenizer to decode
        valid_labels = sample["labels"][sample["labels"] != -100]
        target_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)

        # The target should be the corrected code
        self.assertIn("calculate_sum", target_text)

    def test_clean_sample_fallback(self) -> None:
        """Tests if clean samples fallback to the original source code as target."""
        sample = self.dataset[1]  # The clean sample

        valid_labels = sample["labels"][sample["labels"] != -100]
        target_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)

        # If 'fixed' is None, the target should remain exactly the source code
        self.assertIn("ValidClass", target_text)


if __name__ == "__main__":
    unittest.main()