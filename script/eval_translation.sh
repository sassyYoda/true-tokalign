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
export DATASET_SPLIT="test"  # Options: train, validation, test
export MAX_SAMPLES=""  # Leave empty for all samples, or set a number like 1000
export OUTPUT_DIR="${MAIN_DIR}/log/translation_eval/${MODEL}_${SEED}_${TGT}"
export CACHE_DIR="${MAIN_DIR}/data/cache"  # Optional: specify cache directory for HuggingFace datasets
export DEVICE="cuda"  # Options: cuda or cpu
export DIRECTIONS="both"  # Options: both, es-en, en-es

# Check if model checkpoint exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model checkpoint not found at ${MODEL_PATH}"
    echo "Please check that the model has been trained successfully."
    exit 1
fi

echo "=========================================="
echo "Translation Evaluation"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Dataset: OPUS Global Voices (Spanish-English)"
echo "Split: ${DATASET_SPLIT}"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Directions: ${DIRECTIONS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Build command
CMD="python src/eval_translation.py \
    --model_path ${MODEL_PATH} \
    --dataset_split ${DATASET_SPLIT} \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE} \
    --directions ${DIRECTIONS}"

if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

if [ -n "${CACHE_DIR}" ]; then
    CMD="${CMD} --cache_dir ${CACHE_DIR}"
fi

# Run evaluation
echo "Running evaluation..."
echo "Command: ${CMD}"
echo ""

eval ${CMD}

echo ""
echo "Evaluation complete!"

