.PHONY: test test-verbose test-match build-demo build-mbpp build-all eval lint clean

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
		--seed 42

# Run evaluation with the spellchecker baseline on the demo dataset.
eval:
	uv run python -m src.eval \
		--model spellcheck \
		--dataset data/demo.jsonl

# Build all datasets.
build-all:
	bash build_all_datasets.sh

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
