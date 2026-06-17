.PHONY: test test-verbose test-match build-demo build-mbpp build-magicoder build-codealpaca build-all eval eval-demo eval-mbpp eval-magicoder eval-codealpaca eval-all eval-quick eval-save eval-serial viewer lint clean train-gector eval-gector eval-gector-demo eval-gector-mbpp

# Number of parallel workers for dataset building (auto-detected).
WORKERS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Default model and dataset for evaluation
MODEL ?= spellcheck
DATASET ?= data/demo.jsonl
OUTPUT ?= results.json

# Run all tests.
test:
	uv run python -m pytest tests/ -v

# Run all tests with verbose output.
test-verbose:
	uv run python -m pytest tests/ -vv

# Run a specific test pattern.
# Usage: make test-match PATTERN=CommentCorruption
test-match:
	uv run python -m pytest tests/ -v -k "$(PATTERN)"

# Build the demo dataset only (fast).
build-demo:
	uv run python -m src.build_dataset \
		--source demo \
		--out data/demo.jsonl \
		--variants-per-snippet 5 \
		--max-edits 3 \
		--p-edit 0.8 \
		--seed 42

# Build the MBPP dataset.
build-mbpp:
	uv run python -m src.build_dataset \
		--source mbpp \
		--out data/mbpp/ \
		--variants-per-snippet 5 \
		--max-edits 3 \
		--p-edit 0.8 \
		--seed 42 \
		--workers $(WORKERS)

build-magicoder:
	uv run python -m src.build_dataset \
		--source magicoder \
		--out data/magicoder/ \
		--variants-per-snippet 5 \
		--max-edits 3 \
		--p-edit 0.8 \
		--seed 42 \
		--workers $(WORKERS)

build-codealpaca:
	uv run python -m src.build_dataset \
		--source codealpaca \
		--out data/codealpaca/ \
		--variants-per-snippet 5 \
		--max-edits 3 \
		--p-edit 0.8 \
		--seed 42 \
		--workers $(WORKERS)

# Build all datasets (demo, mbpp, magicoder, codealpaca).
build-all: build-demo build-mbpp build-magicoder build-codealpaca

# Run evaluation with the spellchecker baseline (parallel by default).
# Usage: make eval DATASET=data/demo.jsonl MODEL=spellcheck
eval:
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset $(DATASET) \
		--workers $(WORKERS)

# Run evaluation on demo dataset.
# Usage: make eval-demo MODEL=spellcheck
eval-demo:
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/demo.jsonl \
		--workers $(WORKERS)

# Run evaluation on MBPP dataset (all splits).
# Usage: make eval-mbpp MODEL=spellcheck
eval-mbpp:
	@echo "Evaluating MBPP train split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/mbpp/train.jsonl \
		--workers $(WORKERS)
	@echo "Evaluating MBPP validation split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/mbpp/validation.jsonl \
		--workers $(WORKERS)
	@echo "Evaluating MBPP test split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/mbpp/test.jsonl \
		--workers $(WORKERS)

# Run evaluation on Magicoder dataset (all splits).
# Usage: make eval-magicoder MODEL=spellcheck
eval-magicoder:
	@echo "Evaluating Magicoder train split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/magicoder/train.jsonl \
		--workers $(WORKERS)
	@echo "Evaluating Magicoder validation split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/magicoder/validation.jsonl \
		--workers $(WORKERS)
	@echo "Evaluating Magicoder test split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/magicoder/test.jsonl \
		--workers $(WORKERS)

# Run evaluation on CodeAlpaca dataset (all splits).
# Usage: make eval-codealpaca MODEL=spellcheck
eval-codealpaca:
	@echo "Evaluating CodeAlpaca train split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/codealpaca/train.jsonl \
		--workers $(WORKERS)
	@echo "Evaluating CodeAlpaca validation split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/codealpaca/validation.jsonl \
		--workers $(WORKERS)
	@echo "Evaluating CodeAlpaca test split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/codealpaca/test.jsonl \
		--workers $(WORKERS)


# Start the dataset viewer (opens browser).
viewer:
	uv run python -m src.viewer

# Lint Python sources.
lint:
	uv run python -m py_compile src/*.py

# Remove generated data files.
clean:
	rm -rf data/*.jsonl
	rm -rf data/mbpp/
	rm -rf data/magicoder/
	rm -rf data/codealpaca/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete

# ------------------------------------------------------------------ #
# GECToR targets
# ------------------------------------------------------------------ #

# GECToR checkpoint directory (override with: make train-gector GECTOR_MODEL=...)
GECTOR_MODEL ?= models/gector-roberta
# Encoder backbone (override with: make train-gector GECTOR_ENCODER=microsoft/codebert-base)
GECTOR_ENCODER ?= roberta-base
GECTOR_EPOCHS ?= 10
GECTOR_BATCH  ?= 16
GECTOR_LR     ?= 2e-5

# Fine-tune GECToR on MBPP only (train + val splits).
# Requires: make build-mbpp first.
# Usage:
#   make train-gector
#   make train-gector GECTOR_ENCODER=microsoft/codebert-base GECTOR_EPOCHS=20
train-gector:
	uv run python -m src.gector.train \
		--train data/mbpp/train.jsonl \
		--val   data/mbpp/validation.jsonl \
		--out   $(GECTOR_MODEL) \
		--encoder $(GECTOR_ENCODER) \
		--epochs $(GECTOR_EPOCHS) \
		--batch-size $(GECTOR_BATCH) \
		--lr $(GECTOR_LR)

# Fine-tune GECToR on ALL available datasets merged (mbpp + magicoder + codealpaca).
# Validation uses MBPP validation split.
# Requires: make build-all first.
# Usage:
#   make train-gector-all
#   make train-gector-all GECTOR_ENCODER=microsoft/codebert-base GECTOR_EPOCHS=15
train-gector-all:
	uv run python -m src.gector.train \
		--train data/mbpp/train.jsonl \
		        data/magicoder/train.jsonl \
		        data/codealpaca/train.jsonl \
		--val   data/mbpp/validation.jsonl \
		--out   $(GECTOR_MODEL)-all \
		--encoder $(GECTOR_ENCODER) \
		--epochs $(GECTOR_EPOCHS) \
		--batch-size $(GECTOR_BATCH) \
		--lr $(GECTOR_LR)

# Evaluate GECToR on the demo dataset.
# Usage: make eval-gector-demo GECTOR_MODEL=models/gector-roberta
eval-gector-demo:
	uv run python -m src.eval \
		--model gector \
		--dataset data/demo.jsonl \
		--gector-model $(GECTOR_MODEL)/best \
		--workers 1

# Evaluate GECToR on MBPP test split.
# Usage: make eval-gector-mbpp GECTOR_MODEL=models/gector-roberta
eval-gector-mbpp:
	uv run python -m src.eval \
		--model gector \
		--dataset data/mbpp/test.jsonl \
		--gector-model $(GECTOR_MODEL)/best \
		--workers 1

# Evaluate GECToR on Magicoder test split.
eval-gector-magicoder:
	uv run python -m src.eval \
		--model gector \
		--dataset data/magicoder/test.jsonl \
		--gector-model $(GECTOR_MODEL)/best \
		--workers 1

# Evaluate GECToR on CodeAlpaca test split.
eval-gector-codealpaca:
	uv run python -m src.eval \
		--model gector \
		--dataset data/codealpaca/test.jsonl \
		--gector-model $(GECTOR_MODEL)/best \
		--workers 1

# Evaluate GECToR on ALL dataset test splits (mbpp + magicoder + codealpaca).
# Usage: make eval-gector-all GECTOR_MODEL=models/gector-roberta
eval-gector-all: eval-gector-demo eval-gector-mbpp eval-gector-magicoder eval-gector-codealpaca

# Shorthand: evaluate best checkpoint on demo.
eval-gector: eval-gector-demo
