#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Default paths
export OUTPUT_PATH="${MAIN_DIR}/data/pretrain-corpus/glove-corpus-1B.jsonl"
export TOTAL_TOKENS=1000000000
export CACHE_DIR="${MAIN_DIR}/data/cache"

# Dataset ratios (as specified in TokAlign paper)
export CULTURAX_RATIO=0.4
export STACK_RATIO=0.3
export PROOF_PILE_RATIO=0.3

# Tokenizer for counting tokens
export TOKENIZER_NAME="EleutherAI/pythia-1b"

# Dataset names (can be overridden)
export CULTURAX_DATASET="${CULTURAX_DATASET:-uonlp/CulturaX}"
export STACK_DATASET="${STACK_DATASET:-bigcode/the-stack}"
export PROOF_PILE_DATASET="${PROOF_PILE_DATASET:-lehduong/proof-pile-2}"

# Random seed
export SEED=42

# Create output directory
mkdir -p "$(dirname ${OUTPUT_PATH})"

echo "=========================================="
echo "Preparing GloVe Training Corpus"
echo "=========================================="
echo "Output path: ${OUTPUT_PATH}"
echo "Total tokens: ${TOTAL_TOKENS}"
echo "CulturaX ratio: ${CULTURAX_RATIO}"
echo "The Stack ratio: ${STACK_RATIO}"
echo "Proof-Pile-2 ratio: ${PROOF_PILE_RATIO}"
echo "Tokenizer: ${TOKENIZER_NAME}"
echo "=========================================="

python src/prepare_glove_corpus.py \
    --output-path ${OUTPUT_PATH} \
    --total-tokens ${TOTAL_TOKENS} \
    --culturax-ratio ${CULTURAX_RATIO} \
    --stack-ratio ${STACK_RATIO} \
    --proof-pile-ratio ${PROOF_PILE_RATIO} \
    --cache-dir ${CACHE_DIR} \
    --tokenizer-name ${TOKENIZER_NAME} \
    --seed ${SEED} \
    --culturax-dataset ${CULTURAX_DATASET} \
    --stack-dataset ${STACK_DATASET} \
    --proof-pile-dataset ${PROOF_PILE_DATASET}

echo ""
echo "GloVe corpus preparation complete!"
echo "Output saved to: ${OUTPUT_PATH}"

