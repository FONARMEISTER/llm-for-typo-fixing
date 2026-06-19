"""
Fine-tune a small decoder-only model with Q-LoRA for Python typo fixing.

Uses :class:`CausalLMTypoDataset` (instruction-following format):
"Fix grammar and typos in this Python code:\\n{code}\\n\\nCorrected code:\\n{fixed}"

Usage:
    uv run python scripts/train_lora.py
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Project imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.typo_datasets import CausalLMTypoDataset


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

_CKPT_DIR = Path(__file__).resolve().parent.parent / "models" / "lora-qwen-coder"
_CKPT_DIR.mkdir(parents=True, exist_ok=True)

_MAX_LENGTH = 768  # Tokens (covers ~p90 of train samples).
_BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q-LoRA typo fixer")
    parser.add_argument(
        "--base-model", default=_BASE_MODEL,
        help="HuggingFace model ID (default: Qwen 2.5 Coder 0.5B).",
    )
    parser.add_argument(
        "--max-length", type=int, default=_MAX_LENGTH,
        help="Maximum token length for inputs.",
    )
    parser.add_argument("--output-dir", default=str(_CKPT_DIR), help="Save directory.")
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lora-r", type=int, default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=0,
        help="Cap on training samples (0 = use all).",
    )
    return parser.parse_args(argv)


def _load_datasets(max_length: int, max_train: int):
    """Load training and validation datasets."""
    data_root = Path(__file__).resolve().parent.parent / "data"

    train_paths = [
        str(data_root / "mbpp" / "train.jsonl"),
        str(data_root / "codealpaca" / "train.jsonl"),
    ]
    val_paths = [
        str(data_root / "mbpp" / "validation.jsonl"),
        str(data_root / "codealpaca" / "validation.jsonl"),
    ]

    # We only use has_errors=True samples for seq2seq training.
    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = CausalLMTypoDataset(train_paths, tokenizer, max_length=max_length)
    val_ds = CausalLMTypoDataset(val_paths, tokenizer, max_length=max_length)

    # Filter to only corrupted samples.
    train_ds.data = [d for d in train_ds.data if d.get("has_errors") and d.get("fixed")]
    val_ds.data = [d for d in val_ds.data if d.get("has_errors") and d.get("fixed")]

    if max_train and len(train_ds.data) > max_train:
        train_ds.data = train_ds.data[:max_train]

    print(f"Train samples: {len(train_ds.data)}, Val samples: {len(val_ds.data)}")
    return train_ds, val_ds, tokenizer


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---------- Data ----------
    train_ds, val_ds, tokenizer = _load_datasets(args.max_length, args.max_train_samples)

    # ---------- Model ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------- Training ----------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=25,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        bf16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print(f"\nTraining for {args.epochs} epoch(s)...")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Steps per epoch: ~{len(train_ds) // (args.batch_size * args.grad_accum)}")

    trainer.train()

    # ---------- Save ----------
    print(f"\nSaving to {args.output_dir} ...")
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # Merge LoRA and save a full-precision checkpoint for fast inference.
    merged_dir = Path(args.output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"Merging LoRA into base model and saving to {merged_dir} ...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
