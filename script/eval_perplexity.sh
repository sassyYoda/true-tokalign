#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Model configuration (adjust based on your training setup)
export MODEL="1b"
export TGT="qwen2-7b"
export SEED=0
export NUM_STEPS=2500

# Path to the trained model checkpoint (STAGE-2 final checkpoint)
export MODEL_PATH="${MAIN_DIR}/log/${MODEL}/${SEED}_${TGT}_S2/checkpoint-${NUM_STEPS}"

# Evaluation settings
export CORPUS_NAME="mc4"  # Options: mc4, wikipedia, oscar
export SPLIT="validation"  # Options: train, validation, test
export MAX_SAMPLES=""  # Leave empty for all samples, or set a number like 10000
export OUTPUT_DIR="${MAIN_DIR}/log/perplexity_eval/${MODEL}_${SEED}_${TGT}_${CORPUS_NAME}"
export CACHE_DIR="${MAIN_DIR}/data/cache"  # Optional: specify cache directory for HuggingFace datasets
export DEVICE="cuda"  # Options: cuda or cpu

# Batch sizes (optimized for H100 GPU)
export BATCH_SIZE=32  # Batch size for perplexity computation
export MAX_LENGTH=512  # Maximum sequence length

# Check if model checkpoint exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model checkpoint not found at ${MODEL_PATH}"
    echo "Please check that the model has been trained successfully."
    exit 1
fi

echo "=========================================="
echo "Perplexity Evaluation (Spanish Monolingual)"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Corpus: ${CORPUS_NAME} (Spanish)"
echo "Split: ${SPLIT}"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max length: ${MAX_LENGTH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Build command
CMD="python eval/eval_perplexity.py \
    --model_path ${MODEL_PATH} \
    --corpus_name ${CORPUS_NAME} \
    --split ${SPLIT} \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE} \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH}"

if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

if [ -n "${CACHE_DIR}" ]; then
    CMD="${CMD} --cache_dir ${CACHE_DIR}"
fi

# Run evaluation
echo "Running perplexity evaluation..."
echo "Command: ${CMD}"
echo ""

eval ${CMD}

echo ""
echo "Evaluation complete!"

