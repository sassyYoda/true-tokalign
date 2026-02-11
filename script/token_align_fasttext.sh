#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# FastText repository should be cloned as a sibling directory
export FASTTEXT_DIR="${FASTTEXT_DIR:-$(cd "${MAIN_DIR}/.." && pwd)/fastText}"

export MODLE_PATH1="EleutherAI/pythia-1b"
export TOKENIZER_PATH1="EleutherAI/pythia-1b"
export FASTTEXT_TRAIN_PATH1="${MAIN_DIR}/data/pretrain-dataset/mix-pythia-glove"
export FASTTEXT_VECTOR_PATH1="${MAIN_DIR}/data/vec-mix-pythia_fasttext.txt"

export MODLE_PATH2="Qwen/Qwen2-7B"
export TOKENIZER_PATH2="Qwen/Qwen2-7B"
export FASTTEXT_TRAIN_PATH2="${MAIN_DIR}/data/pretrain-dataset/mix-qwen2-7b-glove"
export FASTTEXT_VECTOR_PATH2="${MAIN_DIR}/data/vec-mix-qwen2-7b_fasttext.txt"

export TGT_ID_2_SRC_ID_GOLD_PATH="${MAIN_DIR}/data/Vocab_count/qwen2-7b2pythia.json"
# The output path of token alignment matrix (with _fasttext suffix)
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2qwen2-7b/align_matrix_fasttext.json"


# Stage-1: train FastText vectors
cd ${FASTTEXT_DIR}
FASTTEXT_VECTOR_NAME1=$(basename ${FASTTEXT_VECTOR_PATH1})
FASTTEXT_VECTOR_NAME1="${FASTTEXT_VECTOR_NAME1%.*}"
printf "\n### Train FastText vector ${FASTTEXT_VECTOR_NAME1} with ${FASTTEXT_TRAIN_PATH1}  ###\n\n"
bash ${MAIN_DIR}/script/train_fasttext.sh ${FASTTEXT_TRAIN_PATH1} ${FASTTEXT_VECTOR_NAME1}
mv ${FASTTEXT_VECTOR_NAME1}.vec ${FASTTEXT_VECTOR_PATH1}

FASTTEXT_VECTOR_NAME2=$(basename ${FASTTEXT_VECTOR_PATH2})
FASTTEXT_VECTOR_NAME2="${FASTTEXT_VECTOR_NAME2%.*}"
printf "\n### Train FastText vector ${FASTTEXT_VECTOR_NAME2} with ${FASTTEXT_TRAIN_PATH2}  ###\n\n"
bash ${MAIN_DIR}/script/train_fasttext.sh ${FASTTEXT_TRAIN_PATH2} ${FASTTEXT_VECTOR_NAME2}
mv ${FASTTEXT_VECTOR_NAME2}.vec ${FASTTEXT_VECTOR_PATH2}


# Stage-2: token ID align
cd ${MAIN_DIR}

export VOCAB_SIZE1=$(python original_tokalign/count_vocab.py -m ${MODLE_PATH1})
export VOCAB_SIZE2=$(python original_tokalign/count_vocab.py -m ${MODLE_PATH2})

python original_tokalign/count_dict.py \
    -s ${TOKENIZER_PATH1} \
    -t ${TOKENIZER_PATH2} \
    -o ${TGT_ID_2_SRC_ID_GOLD_PATH}

python original_tokalign/cal_trans_matrix_fasttext.py \
    -s ${FASTTEXT_VECTOR_PATH1} \
    -s1 ${VOCAB_SIZE1} \
    -t ${FASTTEXT_VECTOR_PATH2} \
    -s2 ${VOCAB_SIZE2} \
    -r -n 300 \
    -g ${TGT_ID_2_SRC_ID_GOLD_PATH} \
    -o ${TGT_ID_2_SRC_ID_RES_PATH}
