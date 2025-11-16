#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

echo "Baseline model (Original Pythia-1B) requires no initialization."
echo "Using EleutherAI/pythia-1b directly from HuggingFace."
echo "Model will be loaded automatically during training."
