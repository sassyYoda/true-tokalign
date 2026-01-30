#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2qwen2-7b/align_matrix.json"
# export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2qwen2-7b/align_matrix_demo.json"

export MODLE_PATH1="EleutherAI/pythia-1b"

export TOKENIZER_PATH2="Qwen/Qwen2-7B"

export OUTPUT_PATH="${MAIN_DIR}/data/pythia2qwen2-7b/TokAlign-Init-1B"

python original_tokalign/convert.py \
    -m ${TGT_ID_2_SRC_ID_RES_PATH} \
    -s ${MODLE_PATH1} \
    -t ${TOKENIZER_PATH2} \
    -o ${OUTPUT_PATH}
