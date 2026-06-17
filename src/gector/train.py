"""Training loop for the GECToR code-typo tagger.

Usage
-----
::

    # Build vocab from dataset, then train:
    uv run python -m src.gector.train \\
        --train  data/mbpp/train.jsonl \\
        --val    data/mbpp/validation.jsonl \\
        --out    models/gector-roberta \\
        --encoder roberta-base \\
        --epochs 10 \\
        --batch-size 16 \\
        --lr 2e-5

    # Resume from checkpoint:
    uv run python -m src.gector.train \\
        --train  data/mbpp/train.jsonl \\
        --val    data/mbpp/validation.jsonl \\
        --out    models/gector-roberta \\
        --resume models/gector-roberta/checkpoint-epoch3

The script:
1. Builds (or loads) a :class:`~src.gector.vocab.TagVocab` from the
   training data.
2. Constructs a :class:`~src.gector.model.GECToRModel`.
3. Trains with AdamW + linear warmup + cosine decay.
4. Saves a checkpoint after every epoch and keeps the best by validation
   tag-loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import GECToRDataset, make_collate_fn
from .model import GECToRModel
from .vocab import TagVocab


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_tokenizer(encoder_model: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(encoder_model, use_fast=True)
    return tok


def _linear_warmup_cosine_decay(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Return a LambdaLR scheduler with linear warmup + cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #


def train(
    train_paths: List[str],
    val_paths: List[str],
    out_dir: str,
    encoder_model: str = "roberta-base",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 512,
    detect_weight: float = 0.5,
    hidden_dropout: float = 0.1,
    grad_clip: float = 1.0,
    resume: Optional[str] = None,
    device_str: Optional[str] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> None:
    """Full training run.

    Parameters
    ----------
    train_paths:
        List of training JSONL file paths.
    val_paths:
        List of validation JSONL file paths.
    out_dir:
        Directory where checkpoints and the final model are saved.
    encoder_model:
        HuggingFace model name or path for the Transformer encoder.
    epochs:
        Number of training epochs.
    batch_size:
        Per-device batch size.
    lr:
        Peak learning rate for AdamW.
    warmup_ratio:
        Fraction of total steps used for linear warmup.
    max_length:
        Maximum subword sequence length.
    detect_weight:
        Weight of the detection loss relative to the tag loss.
    hidden_dropout:
        Dropout on encoder hidden states.
    grad_clip:
        Gradient clipping norm.
    resume:
        Path to a checkpoint directory to resume from.
    device_str:
        Device string (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``).
        Auto-detected if ``None``.
    num_workers:
        DataLoader worker processes.
    seed:
        Random seed for reproducibility.
    """
    torch.manual_seed(seed)
    device = _get_device(device_str)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Encoder: {encoder_model}")
    print(f"Output: {out_path}")

    # ---- Tokenizer ----
    tokenizer = _build_tokenizer(encoder_model if resume is None else resume)
    pad_id = tokenizer.pad_token_id or 0

    # ---- Vocabulary ----
    vocab_path = out_path / "vocab.txt"
    if resume and (Path(resume) / "vocab.txt").exists():
        print(f"Loading vocab from {resume}/vocab.txt")
        vocab = TagVocab.load(Path(resume) / "vocab.txt")
    elif vocab_path.exists():
        print(f"Loading existing vocab from {vocab_path}")
        vocab = TagVocab.load(vocab_path)
    else:
        print("Building vocab from training data...")
        vocab = TagVocab.build_from_jsonl(train_paths)
        vocab.save(vocab_path)
        print(f"Vocab size: {vocab.size}  (saved to {vocab_path})")

    # ---- Datasets ----
    print("Loading datasets...")
    train_ds = GECToRDataset(
        [Path(p) for p in train_paths], vocab, tokenizer,
        max_length=max_length, include_clean=True,
    )
    val_ds = GECToRDataset(
        [Path(p) for p in val_paths], vocab, tokenizer,
        max_length=max_length, include_clean=True,
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")

    collate = make_collate_fn(pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers,
    )

    # ---- Model ----
    if resume:
        print(f"Resuming from {resume}")
        model = GECToRModel.from_pretrained(resume)
    else:
        model = GECToRModel.from_encoder(
            encoder_model, vocab,
            hidden_dropout=hidden_dropout,
            detect_weight=detect_weight,
        )
    model = model.to(device)

    # ---- Optimizer & Scheduler ----
    # Use different LR for encoder (fine-tuning) vs heads (training from scratch).
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.tag_head.parameters()) + list(model.detect_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr},
        {"params": head_params, "lr": lr * 10},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = _linear_warmup_cosine_decay(optimizer, warmup_steps, total_steps)

    # ---- Training state ----
    best_val_loss = float("inf")
    history = []

    # ---- Epoch loop ----
    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                tag_labels=batch["tag_labels"],
                detect_labels=batch["detect_labels"],
            )
            loss = out["loss"]
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(train_steps, 1)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    tag_labels=batch["tag_labels"],
                    detect_labels=batch["detect_labels"],
                )
                val_loss += out["loss"].item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}"
        )

        # -- Save checkpoint --
        ckpt_dir = out_path / f"checkpoint-epoch{epoch}"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        # -- Track best --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_dir = out_path / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  ✓ New best val_loss={best_val_loss:.4f} → saved to {best_dir}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })

    # ---- Save final model ----
    final_dir = out_path / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining complete.  Final model saved to {final_dir}")
    print(f"Best val_loss={best_val_loss:.4f} → {out_path / 'best'}")

    # Save training history.
    history_path = out_path / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training history saved to {history_path}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune GECToR for code identifier typo correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train", nargs="+", required=True, metavar="JSONL",
        help="Training JSONL file(s).",
    )
    parser.add_argument(
        "--val", nargs="+", required=True, metavar="JSONL",
        help="Validation JSONL file(s).",
    )
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="Output directory for checkpoints and final model.",
    )
    parser.add_argument(
        "--encoder", default="roberta-base", metavar="MODEL",
        help="HuggingFace encoder model name or path.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--detect-weight", type=float, default=0.5)
    parser.add_argument("--hidden-dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", default=None, metavar="CKPT_DIR",
                        help="Resume training from a checkpoint directory.")
    parser.add_argument("--device", default=None,
                        help="Device string (cuda / mps / cpu).  Auto-detected if omitted.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    train(
        train_paths=args.train,
        val_paths=args.val,
        out_dir=args.out,
        encoder_model=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        detect_weight=args.detect_weight,
        hidden_dropout=args.hidden_dropout,
        grad_clip=args.grad_clip,
        resume=args.resume,
        device_str=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
