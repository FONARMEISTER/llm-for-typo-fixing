.PHONY: test test-verbose test-match build-demo build-mbpp build-magicoder build-codealpaca build-github-python build-quicktest build-all eval eval-demo eval-mbpp eval-magicoder eval-codealpaca eval-all eval-quick eval-save eval-serial viewer ruff ruff-fix lint coverage coverage-json coverage-html clean train-gector train-gector-all

# Number of parallel workers for dataset building (auto-detected).
WORKERS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
WORKES_EVAL := 1
# Default model and dataset for evaluation
MODEL ?= spellcheck
DATASET ?= data/demo.jsonl
OUTPUT ?= results.json

# GECToR checkpoint directory (override with: make train-gector GECTOR_MODEL=...)
GECTOR_MODEL ?= models/gector-codebert
# Encoder backbone (override with: make train-gector GECTOR_ENCODER=...)
GECTOR_ENCODER ?= microsoft/codebert-base
GECTOR_EPOCHS ?= 10
GECTOR_BATCH  ?= 16
GECTOR_LR     ?= 2e-5

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

# Build the GitHub Python dataset (real OSS code, test-only).
build-github-python:
	uv run python -m src.build_dataset \
		--source github_python \
		--out data/github_python/ \
		--variants-per-snippet 1 \
		--max-edits 2 \
		--p-edit 0.8 \
		--seed 42

# Small CodeAlpaca-based dataset for quick model smoketests (500 samples, seed 123).
build-quicktest:
	uv run python -m src.build_dataset \
		--source codealpaca \
		--out data/quicktest.jsonl \
		--variants-per-snippet 1 \
		--max-edits 2 \
		--p-edit 0.8 \
		--seed 123
	@head -500 data/quicktest.jsonl > data/quicktest.jsonl.tmp
	@mv data/quicktest.jsonl.tmp data/quicktest.jsonl

# Build all datasets (demo, mbpp, magicoder, codealpaca, github_python).
build-all: build-demo build-mbpp build-magicoder build-codealpaca build-github-python

# Run evaluation with the spellchecker baseline (parallel by default).
# Usage: make eval DATASET=data/demo.jsonl MODEL=spellcheck
# Usage: make eval DATASET=data/mbpp/test.jsonl MODEL=gector GECTOR_MODEL=models/gector-roberta
eval:
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset $(DATASET) \
		--gector-model $(GECTOR_MODEL)/best \
		--workers $(WORKES_EVAL)

# Run evaluation on demo dataset.
# Usage: make eval-demo MODEL=spellcheck
# Usage: make eval-demo MODEL=gector GECTOR_MODEL=models/gector-roberta
eval-demo:
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/demo.jsonl \
		--gector-model $(GECTOR_MODEL)/best \
		--workers $(WORKES_EVAL)

# Run evaluation on MBPP dataset (all splits).
# Usage: make eval-mbpp MODEL=spellcheck
# Usage: make eval-mbpp MODEL=gector GECTOR_MODEL=models/gector-roberta
eval-mbpp:
	@echo "Evaluating MBPP test split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/mbpp/test.jsonl \
		--gector-model $(GECTOR_MODEL)/best \
		--workers $(WORKES_EVAL)

# Run evaluation on Magicoder dataset (all splits).
# Usage: make eval-magicoder MODEL=spellcheck
# Usage: make eval-magicoder MODEL=gector GECTOR_MODEL=models/gector-roberta
eval-magicoder:
	@echo "Evaluating Magicoder test split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/magicoder/test.jsonl \
		--gector-model $(GECTOR_MODEL)/best \
		--workers $(WORKES_EVAL)

# Run evaluation on CodeAlpaca dataset (all splits).
# Usage: make eval-codealpaca MODEL=spellcheck
# Usage: make eval-codealpaca MODEL=gector GECTOR_MODEL=models/gector-roberta
eval-codealpaca:
	@echo "Evaluating CodeAlpaca test split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/codealpaca/test.jsonl \
		--gector-model $(GECTOR_MODEL)/best \
		--workers $(WORKES_EVAL)

eval-github:
	@echo "Evaluating github_python test split..."
	uv run python -m src.eval \
		--model $(MODEL) \
		--dataset data/github_python/test.jsonl \
		--gector-model $(GECTOR_MODEL)/best \
		--workers $(WORKES_EVAL)


# Run evaluation on ALL datasets (demo + mbpp + magicoder + codealpaca), all splits.
# Usage: make eval-all MODEL=spellcheck
# Usage: make eval-all MODEL=gector GECTOR_MODEL=models/gector-roberta
eval-all: eval-demo eval-mbpp eval-magicoder eval-codealpaca
# Run evaluation with an LLM model via llama-swap (serial — LLM requests are sequential anyway).
# Usage: make eval-llm PRESET=gemma4-26b
export PRESET=gemma4-e2b
eval-llm:
	uv run python -m src.eval \
		--model llm_api \
		--preset $(PRESET) \
		--dataset data/demo.jsonl

# Run all benchmarks, caching results to benchmarks/.
# Usage: make benchmark                    # run missing
#        make benchmark FORCE=1            # re-run all
#        make benchmark SAVE_SAMPLES=1     # include per-sample results
benchmark:
	uv run python -m src.benchmark \
		$(if $(FORCE),--force,) \
		$(if $(SAVE_SAMPLES),--save-samples,)

# Visualise cached benchmark results as charts.
benchmark-viz:
	uv run python -m src.benchmark_viz
# Start the dataset viewer (opens browser).
viewer:
	uv run python -m src.viewer

# Run ruff linter only (no formatting).
ruff:
	uv run ruff check src/ tests/

# Auto-fix safe ruff issues.
ruff-fix:
	uv run ruff check --fix src/ tests/

# Full lint: ruff check + py_compile.
lint: ruff
	uv run python -m py_compile src/*.py

# Test coverage — terminal report with missing lines (human-readable).
coverage:
	uv run python -m pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=json:coverage.json \
		-q
	uv run python -c 'import json; r=json.load(open("coverage.json")); print(f"\nTotal: {r["totals"]["percent_covered"]:.1f}% ({r["totals"]["covered_lines"]}/{r["totals"]["num_statements"]} lines)")'

# Test coverage — JSON report only (machine-readable for AI consumption).
coverage-json:
	uv run python -m pytest tests/ \
		--cov=src \
		--cov-report=json:coverage.json \
		-q
	uv run python -c 'import json; r=json.load(open("coverage.json")); print(json.dumps(r, indent=2))'

# Test coverage — interactive HTML report (opens browser).
coverage-html:
	uv run python -m pytest tests/ \
		--cov=src \
		--cov-report=html:htmlcov \
		-q
	@echo "Opening htmlcov/index.html ..."
	xdg-open htmlcov/index.html 2>/dev/null || open htmlcov/index.html 2>/dev/null || true

# Remove generated data files.
clean:
	rm -rf data/*.jsonl
	rm -rf data/mbpp/
	rm -rf data/magicoder/
	rm -rf data/codealpaca/
	rm -rf data/github_python/
	rm -rf htmlcov/ coverage.json .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete

# ------------------------------------------------------------------ #
# GECToR targets
# ------------------------------------------------------------------ #


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
# Train  = all three *train* splits merged (no demo, no test).
# Val    = all three *validation* splits merged.
# Test splits are reserved for post-training evaluation (eval-gector-all).
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
		        data/magicoder/validation.jsonl \
		        data/codealpaca/validation.jsonl \
		--out   $(GECTOR_MODEL)-all \
		--encoder $(GECTOR_ENCODER) \
		--epochs $(GECTOR_EPOCHS) \
		--batch-size $(GECTOR_BATCH) \
		--lr $(GECTOR_LR)

train-lora:
	uv run python scripts/train_lora.py

train-lora-full:
	uv run python scripts/train_lora.py --max-train-samples 20000

