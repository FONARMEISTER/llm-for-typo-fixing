"""GECToR model: Transformer encoder + error-detection head + tag head.

Architecture
------------
::

    Input tokens (subwords)
         │
    ┌────▼────────────────────────────────┐
    │  Pre-trained Transformer encoder    │  (e.g. roberta-base)
    │  hidden states  [B, L, H]           │
    └────┬────────────────────────────────┘
         │
    ┌────▼──────────────┐   ┌────────────────────────┐
    │  Detect head      │   │  Tag head               │
    │  Linear(H, 2)     │   │  Linear(H, |vocab|)     │
    │  binary CE loss   │   │  multi-class CE loss    │
    └───────────────────┘   └────────────────────────┘

Both heads are applied to **every subword position**, but only the first
subword of each code token carries a meaningful label (continuation
subwords are masked with ``LABEL_IGNORE = -100``).

The total loss is::

    loss = tag_loss + detect_weight * detect_loss

where ``detect_weight`` defaults to 0.5.

Saving / loading
----------------
The full model (encoder + heads + vocab) is saved as a directory::

    <checkpoint_dir>/
        config.json          ← GECToRConfig (JSON)
        vocab.txt            ← TagVocab (one tag per line)
        pytorch_model.bin    ← state_dict (or model.safetensors)
        <encoder files>      ← HuggingFace model files

Use :meth:`GECToRModel.save_pretrained` and
:meth:`GECToRModel.from_pretrained`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .vocab import TagVocab
from .dataset import LABEL_IGNORE


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #


@dataclass
class GECToRConfig:
    """Hyper-parameters for :class:`GECToRModel`.

    Parameters
    ----------
    encoder_model:
        HuggingFace model name or local path for the Transformer encoder.
    vocab_size:
        Number of tags in the :class:`~src.gector.vocab.TagVocab`.
    hidden_dropout:
        Dropout probability applied to encoder hidden states before the
        classification heads.
    detect_weight:
        Weight of the error-detection loss relative to the tag loss.
    """

    encoder_model: str = "microsoft/codebert-base"
    vocab_size: int = 3          # will be overwritten from actual vocab
    hidden_dropout: float = 0.1
    detect_weight: float = 0.5

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GECToRConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #


class GECToRModel(nn.Module):
    """GECToR sequence tagger for code identifier typo correction.

    Parameters
    ----------
    config:
        :class:`GECToRConfig` instance.
    vocab:
        :class:`~src.gector.vocab.TagVocab` instance.
    """

    def __init__(self, config: GECToRConfig, vocab: TagVocab) -> None:
        super().__init__()
        self.config = config
        self.vocab = vocab

        # Encoder.
        enc_config = AutoConfig.from_pretrained(config.encoder_model)
        self.encoder = AutoModel.from_pretrained(config.encoder_model)
        hidden_size: int = enc_config.hidden_size

        # Heads.
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.tag_head = nn.Linear(hidden_size, vocab.size)
        self.detect_head = nn.Linear(hidden_size, 2)

        # Loss.
        self._tag_loss_fn = nn.CrossEntropyLoss(ignore_index=LABEL_IGNORE)
        self._detect_loss_fn = nn.CrossEntropyLoss(ignore_index=LABEL_IGNORE)

    # ---------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------- #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tag_labels: Optional[torch.Tensor] = None,
        detect_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass.

        Parameters
        ----------
        input_ids:
            LongTensor ``[B, L]``.
        attention_mask:
            LongTensor ``[B, L]``.
        tag_labels:
            Optional LongTensor ``[B, L]`` with ``LABEL_IGNORE`` at
            non-first-subword positions.  Required for training.
        detect_labels:
            Optional LongTensor ``[B, L]``.  Required for training.

        Returns
        -------
        dict with keys:
            ``tag_logits``     — FloatTensor ``[B, L, vocab_size]``
            ``detect_logits``  — FloatTensor ``[B, L, 2]``
            ``loss``           — scalar FloatTensor (only if labels given)
            ``tag_loss``       — scalar FloatTensor (only if labels given)
            ``detect_loss``    — scalar FloatTensor (only if labels given)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = self.dropout(outputs.last_hidden_state)  # [B, L, H]

        tag_logits = self.tag_head(hidden)        # [B, L, V]
        detect_logits = self.detect_head(hidden)  # [B, L, 2]

        result: Dict[str, torch.Tensor] = {
            "tag_logits": tag_logits,
            "detect_logits": detect_logits,
        }

        if tag_labels is not None and detect_labels is not None:
            B, L, V = tag_logits.shape
            tag_loss = self._tag_loss_fn(
                tag_logits.view(B * L, V),
                tag_labels.view(B * L),
            )
            detect_loss = self._detect_loss_fn(
                detect_logits.view(B * L, 2),
                detect_labels.view(B * L),
            )
            loss = tag_loss + self.config.detect_weight * detect_loss
            result["loss"] = loss
            result["tag_loss"] = tag_loss
            result["detect_loss"] = detect_loss

        return result

    # ---------------------------------------------------------------- #
    # Inference helpers
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def predict_tags(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return predicted tag indices and detection probabilities.

        Parameters
        ----------
        input_ids, attention_mask:
            Tensors ``[B, L]`` (batch of one is fine).

        Returns
        -------
        tag_preds : LongTensor ``[B, L]``
            Argmax over tag logits.
        detect_probs : FloatTensor ``[B, L]``
            Softmax probability of class 1 (erroneous) from detect head.
        """
        out = self.forward(input_ids, attention_mask)
        tag_preds = out["tag_logits"].argmax(dim=-1)
        detect_probs = out["detect_logits"].softmax(dim=-1)[..., 1]
        return tag_preds, detect_probs

    # ---------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------- #

    # Name of our config file — deliberately different from HuggingFace's
    # "config.json" so that encoder.save_pretrained() does not overwrite it.
    _GECTOR_CONFIG_FILE = "gector_config.json"

    def save_pretrained(self, directory: str | Path) -> None:
        """Save model, config, and vocab to *directory*.

        Creates the directory if it does not exist.

        File layout::

            <directory>/
                gector_config.json       ← GECToRConfig  (our config)
                vocab.txt                ← TagVocab
                config.json              ← HuggingFace encoder config
                pytorch_model_full.bin   ← full state dict (encoder + heads)
                tokenizer.json etc.      ← written by caller via tokenizer.save_pretrained()
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Our config — use a distinct filename to avoid collision with the
        # HuggingFace encoder config.json written below.
        self.config.save(directory / self._GECTOR_CONFIG_FILE)

        # Vocab.
        self.vocab.save(directory / "vocab.txt")

        # Encoder weights + HuggingFace config.json.
        self.encoder.save_pretrained(directory)

        # Full state dict (encoder + heads) for easy reloading.
        torch.save(self.state_dict(), directory / "pytorch_model_full.bin")

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "GECToRModel":
        """Load a model saved with :meth:`save_pretrained`.

        Parameters
        ----------
        directory:
            Path to the checkpoint directory.
        """
        directory = Path(directory)
        config = GECToRConfig.load(directory / cls._GECTOR_CONFIG_FILE)
        vocab = TagVocab.load(directory / "vocab.txt")

        # Point encoder at the local directory (contains HuggingFace config.json).
        config.encoder_model = str(directory)
        model = cls(config, vocab)

        # Load full state dict (heads + encoder).
        full_weights = directory / "pytorch_model_full.bin"
        if full_weights.exists():
            state = torch.load(full_weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state)

        return model

    @classmethod
    def from_encoder(
        cls,
        encoder_model: str,
        vocab: TagVocab,
        hidden_dropout: float = 0.1,
        detect_weight: float = 0.5,
    ) -> "GECToRModel":
        """Construct a fresh model from a pre-trained encoder.

        Parameters
        ----------
        encoder_model:
            HuggingFace model name (e.g. ``"roberta-base"``).
        vocab:
            Tag vocabulary.
        """
        config = GECToRConfig(
            encoder_model=encoder_model,
            vocab_size=vocab.size,
            hidden_dropout=hidden_dropout,
            detect_weight=detect_weight,
        )
        return cls(config, vocab)
