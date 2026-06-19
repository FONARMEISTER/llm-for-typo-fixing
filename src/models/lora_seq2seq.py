"""
Fine-tuned decoder-only model (Q-LoRA) as a seq2seq typo fixer.

Loads a base causal LM plus a LoRA adapter, generates corrected code
via instruction-following prompt, and extracts identifier-level fixes.
"""

import re
from typing import Dict, List, Optional
import warnings

from .base import NameFixer
from .byt5_seq2seq import _diff_by_position


# Suppress per-file SyntaxWarning from parso (same as in identifier_utils.py).
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Path to the default LoRA checkpoint (relative to project root).
_DEFAULT_CKPT = "models/lora-qwen-coder"
_PROMPT_PREFIX = "Fix grammar and typos in this Python code:\n"
_PROMPT_SUFFIX = "\n\nCorrected code:\n"
# Strip trailing newline + prefix from generation output.
_CLEANUP_RE = re.compile(r"\n\s*$")
# Qwen 2.5-Coder fill-in-the-middle tokens that leak into output.
_FIM_RE = re.compile(
    r"<\|fim_(?:prefix|middle|suffix|pad)\|>|<\|repo_name\|>|<\|file_sep\|>"
)


class LoraSeq2SeqFixer(NameFixer):
    """Decoder-only model fine-tuned with Q-LoRA for code correction.

    Uses an instruction-following prompt format (matching the training data
    from :class:`CausalLMTypoDataset`) and generates the corrected code.
    Identifier-level fixes are extracted by positional diffing.
    """

    name = "lora_seq2seq"

    def __init__(
        self,
        checkpoint_dir: str = _DEFAULT_CKPT,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Load base model and LoRA adapter from *checkpoint_dir*.

        Args:
            checkpoint_dir: Directory containing adapter_config.json, adapter_model.safetensors,
                and tokenizer files.
            device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).
                Auto-detected if *None*.
            **kwargs: Forwarded to ``AutoModelForCausalLM.from_pretrained()``
                and ``.generate()``.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._generate_kwargs = kwargs

        # Load 4-bit quantized base model then attach LoRA adapter.
        # We rely on the adapter config stored in checkpoint_dir to specify
        # the base model name.
        import json
        adapter_config_path = f"{checkpoint_dir}/adapter_config.json"
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-0.5B")

        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=self._device,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        self._model.eval()

    def fix_names(self, code: str, names: List[str]) -> Dict[str, str]:
        """Generate corrected code and extract identifier fixes.

        Args:
            code: Corrupted Python source code.
            names: List of identifier names to potentially fix.
                Names not needing correction should be returned unchanged
                (or omitted entirely).

        Returns:
            Dict mapping ``{corrupted_name: fixed_name}`` for identifiers
            the model believes should be changed.
        """
        if not names:
            return {}

        import torch

        prefix = _PROMPT_PREFIX + code + _PROMPT_SUFFIX
        inputs = self._tokenizer(prefix, return_tensors="pt").to(self._device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                **self._generate_kwargs,
            )

        # Slice off the prompt tokens.
        generated_ids = outputs[0][prompt_len:]
        corrected = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        corrected = _FIM_RE.sub("", corrected)  # Strip Qwen FIM tokens.
        corrected = _CLEANUP_RE.sub("", corrected)

        if not corrected:
            return {}

        # Extract identifier fixes via positional diffing.
        return _diff_by_position(code, corrected, names)
