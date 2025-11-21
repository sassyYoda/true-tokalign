#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Evaluation data path
export MATRIX_EVAL_DATA_PATH="${MAIN_DIR}/data/pretrain-dataset/pythia-2-qwen2-7b-glove-eval-mix"

# Alignment matrix directory
export ALIGN_DIR="${MAIN_DIR}/data/pythia2qwen2-7b"

# TokAlign alignment matrix (your method)
export TOKALIGN_MATRIX="${ALIGN_DIR}/align_matrix.json"

# Baseline alignment matrices
export RANDOM_PERM_MATRIX="${ALIGN_DIR}/align_matrix_random_permutation.json"
export RANDOM_INIT_MATRIX="${ALIGN_DIR}/align_matrix_random_initialization.json"

# Evaluation settings
export EVAL_METHOD=bleu
export BLEU_WEIGHT="1,0,0,0"
export TOKENIZER_PATH="EleutherAI/pythia-1b"
export BERT_SCORE_MODEL="all-mpnet-base-v2"

echo "=========================================="
echo "Evaluating Alignment Matrices"
echo "=========================================="
echo "Evaluation data: ${MATRIX_EVAL_DATA_PATH}"
echo "Evaluation method: ${EVAL_METHOD}"
echo "BLEU weights: ${BLEU_WEIGHT}"
echo "=========================================="
echo ""

# Function to evaluate a single alignment matrix
eval_matrix() {
    local matrix_name=$1
    local matrix_path=$2
    
    if [ ! -f "$matrix_path" ]; then
        echo "⚠ Warning: Matrix file not found: $matrix_path"
        echo "  Skipping evaluation for $matrix_name"
        echo ""
        return
    fi
    
    echo "------------------------------------------"
    echo "Evaluating: $matrix_name"
    echo "Matrix path: $matrix_path"
    echo "------------------------------------------"
    
    python src/eval_matrix.py \
        -e ${EVAL_METHOD} \
        -m ${matrix_path} \
        -f ${MATRIX_EVAL_DATA_PATH} \
        -t ${TOKENIZER_PATH} \
        -b ${BERT_SCORE_MODEL} \
        -w ${BLEU_WEIGHT}
    
    echo ""
}

# Evaluate TokAlign (your method)
if [ -f "$TOKALIGN_MATRIX" ]; then
    eval_matrix "TokAlign" "$TOKALIGN_MATRIX"
else
    echo "⚠ Warning: TokAlign matrix not found: $TOKALIGN_MATRIX"
    echo ""
fi

# Evaluate Random Permutation baseline
eval_matrix "Random Permutation Baseline" "$RANDOM_PERM_MATRIX"

# Evaluate Random Initialization baseline
eval_matrix "Random Initialization Baseline" "$RANDOM_INIT_MATRIX"

echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Summary of alignment matrices evaluated:"
echo "  1. TokAlign (your method)"
echo "  2. Random Permutation Baseline"
echo "  3. Random Initialization Baseline"
echo ""
echo "Compare the BLEU scores above to see the improvement"
echo "of TokAlign over random baselines."

