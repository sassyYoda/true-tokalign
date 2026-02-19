#!/bin/bash
set -e
# args:
# corpus save_file
#
# Trains a FastText model using the same corpus format as GloVe
# FastText repository should be cloned as a sibling directory and built
# 
# Build binary: cd fastText && mkdir build && cd build && cmake .. && make

CORPUS=$1
SAVE_FILE=$2
VECTOR_SIZE=300
MIN_COUNT=5
NUM_THREADS=64
WINDOW_SIZE=15
EPOCH=15

# Find FastText binary (build directory first, then current directory)
if [ -f "./build/fasttext" ]; then
    FASTTEXT_BIN="./build/fasttext"
elif [ -f "./fasttext" ]; then
    FASTTEXT_BIN="./fasttext"
else
    echo "Error: FastText binary not found."
    echo "Build FastText: cd fastText && mkdir -p build && cd build && cmake .. && make"
    exit 1
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
# -maxn 0: CRITICAL - Disable subword/char n-grams. Our "words" are token IDs (e.g. "1234").
#   With subwords enabled, FastText breaks "1234" into char n-grams, producing meaningless
#   embeddings that hurt alignment. Word-level only (like GloVe) is correct.
$FASTTEXT_BIN skipgram \
    -input "$CORPUS" \
    -output "$SAVE_FILE" \
    -dim $VECTOR_SIZE \
    -minCount $MIN_COUNT \
    -thread $NUM_THREADS \
    -ws $WINDOW_SIZE \
    -epoch $EPOCH \
    -maxn 0

echo
echo "FastText training completed!"
echo "  Model saved to: ${SAVE_FILE}.bin"
echo "  Vectors saved to: ${SAVE_FILE}.vec"
echo
