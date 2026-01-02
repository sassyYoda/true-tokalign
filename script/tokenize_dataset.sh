#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}
export CACHE_DIR="${MAIN_DIR}/data/cache"

export MODEL_PATH="./data/pythia2qwen2-7b/TokAlign-Init-1B"
export TOKENIZER_PATH="./data/pythia2qwen2-7b/TokAlign-Init-1B"

# Using The Pile corpus for fine-tuning (as specified in TokAlign paper)
export TRAIN_FILE="./data/pretrain-corpus/pile-corpus.jsonl"

export DATASET_PATH="./data/pretrain-dataset/pile00-qwen2-7b-tokenized"
# export DATASET_PATH="./data/pretrain-dataset/pile00-sample-qwen2-7b-tokenized"

export NUM_WORKERS=20  # Optimized for single GPU setup (26 CPU cores available)
export BLOCK_SIZE=2048

# Create log directory if it doesn't exist
mkdir -p ./log

# Create dataset output directory if it doesn't exist
mkdir -p "$(dirname ${DATASET_PATH})"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=========================================="
echo "Dataset Tokenization Script"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Tokenizer path: ${TOKENIZER_PATH}"
echo "Train file: ${TRAIN_FILE}"
echo "Output dataset path: ${DATASET_PATH}"
echo "Block size: ${BLOCK_SIZE}"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="
echo ""

python -u src/process_dataset.py \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name ${TOKENIZER_PATH} \
  --train_file ${TRAIN_FILE} \
  --cache_dir ${CACHE_DIR} \
  --dataset_path_in_disk ${DATASET_PATH} \
  --preprocessing_num_workers ${NUM_WORKERS} \
  --block_size ${BLOCK_SIZE} \
  --do_train \
  --output_dir ./log 2>&1 | tee ./log/process_dataset.log

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Dataset preparation completed successfully!"
    echo "Dataset saved to: ${DATASET_PATH}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: Dataset preparation failed!"
    echo "Check the log file: ./log/process_dataset.log"
    echo "=========================================="
    exit 1
fi