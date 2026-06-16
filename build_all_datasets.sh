#!/bin/bash

set -e

echo "=========================================="
echo "Building Code Grammar Error Datasets"
echo "=========================================="
echo ""

VARIANTS_PER_SNIPPET=5
MAX_EDITS=3
P_EDIT=0.8
SEED=42

echo "Building demo dataset..."
uv run src/build_dataset.py \
    --source demo \
    --out data/demo.jsonl \
    --variants-per-snippet $VARIANTS_PER_SNIPPET \
    --max-edits $MAX_EDITS \
    --p-edit $P_EDIT \
    --seed $SEED
echo "✓ Demo dataset completed"
echo ""

echo "Building MBPP dataset..."
uv run src/build_dataset.py \
    --source mbpp \
    --out data/mbpp/ \
    --variants-per-snippet $VARIANTS_PER_SNIPPET \
    --max-edits $MAX_EDITS \
    --p-edit $P_EDIT \
    --seed $SEED
echo "✓ MBPP dataset completed"
echo ""

echo "Building Magicoder dataset..."
uv run src/build_dataset.py \
    --source magicoder \
    --out data/magicoder/ \
    --variants-per-snippet $VARIANTS_PER_SNIPPET \
    --max-edits $MAX_EDITS \
    --p-edit $P_EDIT \
    --seed $SEED
echo "✓ Magicoder dataset completed"
echo ""

echo "Building CodeAlpaca dataset..."
uv run src/build_dataset.py \
    --source codealpaca \
    --out data/codealpaca/ \
    --variants-per-snippet $VARIANTS_PER_SNIPPET \
    --max-edits $MAX_EDITS \
    --p-edit $P_EDIT \
    --seed $SEED
echo "✓ CodeAlpaca dataset completed"
echo ""

echo "=========================================="
echo "All datasets built successfully!"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  - data/demo.jsonl"
echo "  - data/mbpp/"
echo "  - data/magicoder/"
echo "  - data/codealpaca/"
