#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Source and target tokenizers
export SOURCE_TOKENIZER="EleutherAI/pythia-1b"
export TARGET_TOKENIZER="Qwen/Qwen2-7B"

# Output directory for baseline alignment matrices
export OUTPUT_DIR="${MAIN_DIR}/data/pythia2qwen2-7b"

# Random seed
export SEED=42

echo "=========================================="
echo "Generating Random Baseline Alignments"
echo "=========================================="
echo "Source tokenizer: ${SOURCE_TOKENIZER}"
echo "Target tokenizer: ${TARGET_TOKENIZER}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Seed: ${SEED}"
echo "=========================================="

python src/generate_random_baselines.py \
    -s ${SOURCE_TOKENIZER} \
    -t ${TARGET_TOKENIZER} \
    -o ${OUTPUT_DIR} \
    --seed ${SEED} \
    --baseline-type both

echo ""
echo "Baseline alignment matrices generated!"
echo "  - Random permutation: ${OUTPUT_DIR}/align_matrix_random_permutation.json"
echo "  - Random initialization: ${OUTPUT_DIR}/align_matrix_random_initialization.json"

