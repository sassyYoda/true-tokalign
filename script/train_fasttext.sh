#!/bin/bash
set -e
# args:
# corpus save_file
#
# Trains a FastText model using the same corpus format as GloVe
# FastText repository should be cloned as a sibling directory and built
# 
# Installation options:
#   1. Build binary: cd fastText && mkdir build && cd build && cmake .. && make
#   2. Install via pip: cd fastText && pip install .

CORPUS=$1
SAVE_FILE=$2
VECTOR_SIZE=300
MIN_COUNT=5
NUM_THREADS=64
WINDOW_SIZE=15
EPOCH=15

# Try to find FastText binary (built version)
# Check in build directory first, then current directory
if [ -f "./build/fasttext" ]; then
    FASTTEXT_BIN="./build/fasttext"
elif [ -f "./fasttext" ]; then
    FASTTEXT_BIN="./fasttext"
else
    # If binary not found, try using Python API (if pip installed)
    echo "FastText binary not found, trying Python API..."
    python3 << EOF
import fasttext
import sys

model = fasttext.train_unsupervised(
    '$CORPUS',
    model='skipgram',
    dim=$VECTOR_SIZE,
    minCount=$MIN_COUNT,
    thread=$NUM_THREADS,
    ws=$WINDOW_SIZE,
    epoch=$EPOCH
)

# Save model (will create .bin and .vec files)
model.save_model('${SAVE_FILE}.bin')

print("FastText training completed via Python API!")
print("  Model saved to: ${SAVE_FILE}.bin")
print("  Vectors saved to: ${SAVE_FILE}.vec")
EOF
    exit 0
fi

# FastText parameters
# -input: input file (same format as GloVe - space-separated token IDs)
# -output: output model prefix
# -dim: vector dimension
# -minCount: minimum word count
# -thread: number of threads
# -ws: window size
# -epoch: number of epochs
# -skipgram: use skipgram model (similar to GloVe's approach)

echo
echo "Training FastText model..."
echo "  Corpus: $CORPUS"
echo "  Output: $SAVE_FILE"
echo "  Vector size: $VECTOR_SIZE"
echo "  Min count: $MIN_COUNT"
echo "  Window size: $WINDOW_SIZE"
echo "  Epochs: $EPOCH"
echo "  Threads: $NUM_THREADS"
echo "  Using binary: $FASTTEXT_BIN"
echo

# FastText command - skipgram model (similar to GloVe)
# Note: FastText outputs both .bin (binary) and .vec (text) files
# We'll use the .vec file which has the same format as GloVe (with header)
$FASTTEXT_BIN skipgram \
    -input "$CORPUS" \
    -output "$SAVE_FILE" \
    -dim $VECTOR_SIZE \
    -minCount $MIN_COUNT \
    -thread $NUM_THREADS \
    -ws $WINDOW_SIZE \
    -epoch $EPOCH

echo
echo "FastText training completed!"
echo "  Model saved to: ${SAVE_FILE}.bin"
echo "  Vectors saved to: ${SAVE_FILE}.vec"
echo
