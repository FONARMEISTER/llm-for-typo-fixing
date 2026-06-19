"""ByT5-small Seq2Seq model fine-tuned for Python typo fixing.

The model takes corrupted Python source code prefixed with
``"Fix Python typo:\\n"`` and generates the corrected code.  Identifier-level
fixes are extracted by positional diffing (same approach as ``TyposFixer``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .base import NameFixer
from ..identifier_utils import extract_renameable_identifiers, UnparseableCodeError

_BYT5_CHECKPOINT_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "models" / "checkpoint-byt5"
)
_BYT5_TASK_PREFIX = "Fix Python typo:\n"
_BYT5_MAX_LENGTH = 1024


class ByT5Seq2SeqFixer(NameFixer):
    """ByT5-small fine-tuned to fix typo-corrupted identifiers."""

    name = "byt5"

    def __init__(
        self,
        checkpoint_dir: str = _BYT5_CHECKPOINT_DIR,
        device: Optional[str] = None,
    ) -> None:
        """Load ByT5 from *checkpoint_dir*.

        Args:
            checkpoint_dir: Path to the fine-tuned checkpoint (default from
                ``models/checkpoint-byt5/``).
            device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).
                Auto-detected if *None*.
        """
        # Lazy import to keep transformers optional for the base package.
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = T5ForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
        ).to(self._device)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self._max_length = _BYT5_MAX_LENGTH

    # ------------------------------------------------------------------ #
    # NameFixer interface
    # ------------------------------------------------------------------ #

    def fix_names(self, code: str, names: List[str]) -> Dict[str, str]:
        """Generate corrected code with ByT5, then diff identifiers by position."""
        # Prepend the task prefix that was used during training.
        input_text = _BYT5_TASK_PREFIX + code
        corrected = self._generate(input_text)
        if corrected == code:
            return {}
        return _diff_by_position(code, corrected, names)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _generate(self, code: str) -> str:
        """Run seq2seq inference on the (prefixed) code."""
        inputs = self._tokenizer(
            code,
            return_tensors="pt",
            max_length=self._max_length,
            truncation=True,
        ).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_length=self._max_length,
                num_beams=1,
            )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _diff_by_position(
    original: str, corrected: str, names: List[str]
) -> Dict[str, str]:
    """Map corrected identifiers back to original names by line/column proximity.

    Same approach as :class:`TyposFixer` — extract identifiers from both
    versions, build a position→corrected_name map, and match within ±5
    columns.
    """
    try:
        corr_idents = extract_renameable_identifiers(corrected)
    except UnparseableCodeError:
        return {}

    pos_to_name: Dict[Tuple[int, int], str] = {}
    for cname, positions in corr_idents.items():
        for pos in positions:
            pos_to_name[pos] = cname

    try:
        orig_idents = extract_renameable_identifiers(original)
    except UnparseableCodeError:
        return {}

    result: Dict[str, str] = {}
    for name in names:
        if name not in orig_idents:
            continue
        orig_positions = orig_idents[name]
        for oline, ocol in orig_positions:
            for delta in range(-5, 6):
                candidate = pos_to_name.get((oline, ocol + delta))
                if candidate is not None and candidate != name:
                    result[name] = candidate
                    break
            if name in result:
                break

    return result
