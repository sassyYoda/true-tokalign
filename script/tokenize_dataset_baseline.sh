#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}
export CACHE_DIR="${MAIN_DIR}/data/cache"

# Baseline uses original Pythia-1B tokenizer
export MODLE_PATH="EleutherAI/pythia-1b"
export TOKENIZER_PATH="EleutherAI/pythia-1b"

# Using The Pile corpus for fine-tuning (as specified in TokAlign paper)
export TRAIN_FILE="./data/pretrain-corpus/pile-corpus.jsonl"

# Tokenize with Pythia tokenizer for baseline
export DATASET_PATH="./data/pretrain-dataset/pile00-pythia-tokenized"
# export DATASET_PATH="./data/pretrain-dataset/pile00-sample-pythia-tokenized"

export NUM_WORKERS=60
export BLOCK_SIZE=2048

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python -u src/process_dataset.py \
  --model_name_or_path ${MODLE_PATH} \
  --tokenizer_name ${TOKENIZER_PATH} \
  --train_file ${TRAIN_FILE} \
  --cache_dir ${CACHE_DIR} \
  --dataset_path_in_disk ${DATASET_PATH} \
  --preprocessing_num_workers ${NUM_WORKERS} \
  --block_size ${BLOCK_SIZE} \
  --output_dir ./log 2>&1 | tee ./log/process_dataset_baseline.log
