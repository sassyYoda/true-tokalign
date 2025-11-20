#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Default paths
export OUTPUT_PATH="${MAIN_DIR}/data/pretrain-corpus/pile-corpus.jsonl"
# Using Common Pile (comma_v0.1_training_dataset) as alternative to The Pile
# The original EleutherAI/pile is no longer available from the-eye.eu
export DATASET_NAME="common-pile/comma_v0.1_training_dataset"
export SPLIT="train"
export CACHE_DIR="${MAIN_DIR}/data/cache"

# Optional: limit number of samples (set to empty string for all)
export MAX_SAMPLES=""

# Use streaming mode for large datasets (set to empty string to disable)
# Streaming is enabled by default to avoid downloading the entire Pile dataset
export USE_STREAMING="true"

# Create output directory
mkdir -p "$(dirname ${OUTPUT_PATH})"

echo "=========================================="
echo "Preparing Pile Corpus (Common Pile)"
echo "=========================================="
echo "Dataset: ${DATASET_NAME}"
echo "Split: ${SPLIT}"
echo "Output path: ${OUTPUT_PATH}"
if [ -n "${MAX_SAMPLES}" ]; then
    echo "Max samples: ${MAX_SAMPLES}"
fi
if [ -n "${USE_STREAMING}" ]; then
    echo "Streaming mode: enabled"
fi
echo "=========================================="

# Build command
CMD="python src/prepare_pile_corpus.py \
    --output-path ${OUTPUT_PATH} \
    --dataset-name ${DATASET_NAME} \
    --split ${SPLIT} \
    --cache-dir ${CACHE_DIR}"

if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
fi

if [ -n "${USE_STREAMING}" ]; then
    CMD="${CMD} --streaming"
fi

# Execute command
eval ${CMD}

echo ""
echo "Pile corpus preparation complete!"
echo "Output saved to: ${OUTPUT_PATH}"

