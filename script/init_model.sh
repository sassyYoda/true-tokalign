#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2biogpt/align_matrix.json"
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2biogpt/align_matrix_demo.json"

export MODLE_PATH1="EleutherAI/pythia-1b"

export TOKENIZER_PATH2="microsoft/biogpt"

export OUTPUT_PATH="${MAIN_DIR}/data/pythia2biogpt/TokAlign-Init-1B"

python src/convert.py \
    -m ${TGT_ID_2_SRC_ID_RES_PATH} \
    -s ${MODLE_PATH1} \
    -t ${TOKENIZER_PATH2} \
    -o ${OUTPUT_PATH}
