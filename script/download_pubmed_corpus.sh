#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Output path for the formatted corpus
export OUTPUT_PATH="${MAIN_DIR}/data/pretrain-corpus/pubmed-abstract.json"

# Optional: limit the number of samples for testing (set to None or remove for full dataset)
# export MAX_SAMPLES=10000

# Optional: specify a specific dataset revision/date tag (leave empty for latest)
# export REVISION="2025-03-28"

echo "Downloading and preparing PubMed abstract dataset..."
echo "Output path: ${OUTPUT_PATH}"

python script/download_pubmed_corpus.py \
    -o ${OUTPUT_PATH} \
    ${MAX_SAMPLES:+-m ${MAX_SAMPLES}} \
    ${REVISION:+-r ${REVISION}}

echo "Done! Dataset prepared at: ${OUTPUT_PATH}"

