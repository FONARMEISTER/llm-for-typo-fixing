"""GECToRFixer — NameFixer implementation backed by a trained GECToR model.

This module bridges the GECToR inference engine with the existing
:class:`~src.models.base.NameFixer` interface so that a trained GECToR
checkpoint can be evaluated with the standard harness::

    uv run python -m src.eval \\
        --model gector \\
        --dataset data/mbpp/test.jsonl \\
        --gector-model models/gector-roberta/best

The model checkpoint directory must contain:
    * ``config.json``           — :class:`~src.gector.model.GECToRConfig`
    * ``vocab.txt``             — :class:`~src.gector.vocab.TagVocab`
    * ``pytorch_model_full.bin``— full state dict
    * HuggingFace tokenizer files (``tokenizer.json``, etc.)

Lazy loading
------------
The model and tokenizer are loaded on the first call to
:meth:`GECToRFixer.fix_names` (not at construction time).  This avoids
loading large weights in the main process when the harness spawns worker
processes — each worker loads its own copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..models.base import NameFixer


class GECToRFixer(NameFixer):
    """Identifier typo fixer backed by a trained GECToR sequence tagger.

    Parameters
    ----------
    model_dir:
        Path to the checkpoint directory produced by
        :mod:`src.gector.train`.
    device:
        Torch device string (``"cuda"``, ``"mps"``, ``"cpu"``).
        Auto-detected if ``None``.
    max_iter:
        Maximum iterative correction passes per sample.
    max_length:
        Maximum subword sequence length.
    min_detect_prob:
        Detection probability threshold.  Predictions below this value
        are suppressed (treated as ``$KEEP``).  Set to 0.0 to disable.
    """

    name: str = "gector"

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        max_iter: int = 5,
        max_length: int = 512,
        min_detect_prob: float = 0.0,
    ) -> None:
        self.model_dir = str(model_dir)
        self._device_str = device
        self.max_iter = max_iter
        self.max_length = max_length
        self.min_detect_prob = min_detect_prob

        # Lazy-loaded.
        self._model = None
        self._tokenizer = None
        self._device: Optional[torch.device] = None

    # ---------------------------------------------------------------- #
    # NameFixer interface
    # ---------------------------------------------------------------- #

    def fix_names(self, code: str, names: List[str]) -> Dict[str, str]:
        """Return ``{corrupted_name: fixed_name}`` for identifiers to fix.

        Parameters
        ----------
        code:
            Full Python source code (possibly containing typos).
        names:
            List of renameable identifier names extracted from *code* by
            :func:`~src.identifier_utils.extract_renameable_identifiers`.
            The GECToR model uses the full *code* for context; *names* is
            used to filter the output to only identifiers that appear in
            the code.

        Returns
        -------
        Dict[str, str]
            Only entries where the model suggests a change are included.
        """
        model, tokenizer, device = self._load()

        from .predict import extract_name_fixes
        fixes = extract_name_fixes(
            code, model, tokenizer, device,
            max_iter=self.max_iter,
            max_length=self.max_length,
            min_detect_prob=self.min_detect_prob,
        )

        # Filter to only names that actually appear in the identifier list.
        names_set = set(names)
        return {k: v for k, v in fixes.items() if k in names_set and k != v}

    # ---------------------------------------------------------------- #
    # Lazy loading
    # ---------------------------------------------------------------- #

    def _load(self):
        """Load model and tokenizer on first call (lazy)."""
        if self._model is not None:
            return self._model, self._tokenizer, self._device

        from .model import GECToRModel
        from transformers import AutoTokenizer

        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"GECToR model directory not found: {model_path}\n"
                f"Train a model first with:  make train-gector"
            )

        # Determine device.
        if self._device_str:
            device = torch.device(self._device_str)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model = GECToRModel.from_pretrained(model_path)
        model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)

        self._model = model
        self._tokenizer = tokenizer
        self._device = device

        return self._model, self._tokenizer, self._device
