#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# The path of token alignment matrix
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2qwen2-7b/align_matrix.json"
# export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2qwen2-7b/align_matrix_demo.json"

export MATRIX_EVAL_DATA_PATH="${MAIN_DIR}/data/pretrain-dataset/pythia-2-qwen2-7b-glove-eval-mix"

# Evaluation settings
export EVAL_METHOD=both  # Run both BLEU and BERTScore
export BLEU_WEIGHT="1,0,0,0"
export BERT_SCORE_MODEL="microsoft/deberta-base-mnli"  # BERTScore model (lighter than deberta-xlarge)
export SOURCE_TOKENIZER_PATH="EleutherAI/pythia-1b"
export OUTPUT_DIR="${MAIN_DIR}/data/evaluation_results"
export BERTSCORE_BATCH_SIZE=8  # Reduce if you get CUDA OOM errors (try 4 or 2 if needed)

# Run both BLEU and BERTScore evaluations
python src/eval_matrix.py \
    -e ${EVAL_METHOD} \
    -m ${TGT_ID_2_SRC_ID_RES_PATH} \
    -f ${MATRIX_EVAL_DATA_PATH} \
    -t ${SOURCE_TOKENIZER_PATH} \
    -b ${BERT_SCORE_MODEL} \
    -w ${BLEU_WEIGHT} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BERTSCORE_BATCH_SIZE} \
    --device cuda

# Generate plots from evaluation results (only if results file exists)
if [ -f "${OUTPUT_DIR}/evaluation_results.json" ]; then
    echo ""
    echo "Generating evaluation plots..."
    python src/plot_evaluation_results.py \
        --results-file ${OUTPUT_DIR}/evaluation_results.json \
        --output-dir ${MAIN_DIR}/figure
else
    echo ""
    echo "WARNING: Evaluation results file not found. Skipping plot generation."
    echo "Check the error messages above for details."
fi