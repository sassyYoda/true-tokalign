#!/bin/sh

# Chunked tokenization script for large JSONL files
# Splits large files into chunks, processes each (optionally in parallel), then concatenates

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}
export CACHE_DIR="${MAIN_DIR}/data/cache"

export MODLE_PATH="./data/pythia2qwen2-7b/TokAlign-Init-1B"
export TOKENIZER_PATH="./data/pythia2qwen2-7b/TokAlign-Init-1B"

# Input file
export TRAIN_FILE="./data/pretrain-corpus/pile-corpus.jsonl"

# Output path
export DATASET_PATH="./data/pretrain-dataset/pile00-qwen2-7b-tokenized"

# Chunking settings
export CHUNK_SIZE=2000000  # Lines per chunk (~2M examples per chunk)
export NUM_WORKERS=20  # Optimized for single GPU setup (26 CPU cores available)
export BLOCK_SIZE=2048

# Parallel processing settings
# With 26 CPU cores and 20 workers per chunk, we can process 1 chunk at a time efficiently
# Each chunk uses ~20 workers, leaving 6 cores for system/overhead
# Can do 1 chunk safely, or 2 chunks if you want to push it (but may be slower due to context switching)
export PARALLEL_CHUNKS=1  # Number of chunks to process in parallel (set to 1 for sequential, 2 max for 26 cores)

# Temporary directory for chunks
export CHUNK_DIR="./data/pretrain-corpus/chunks"
export CHUNK_OUTPUT_DIR="./data/pretrain-dataset/chunks-tokenized"

# Create directories
mkdir -p ./log
mkdir -p ${CHUNK_DIR}
mkdir -p ${CHUNK_OUTPUT_DIR}

echo "=========================================="
echo "Chunked Tokenization Script (Single GPU, 26 CPU Cores)"
echo "=========================================="
echo "Input file: ${TRAIN_FILE}"
echo "Output path: ${DATASET_PATH}"
echo "Chunk size: ${CHUNK_SIZE} lines"
echo "Workers per chunk: ${NUM_WORKERS} (26 CPU cores available)"
echo "Parallel chunks: ${PARALLEL_CHUNKS}"
echo "=========================================="
echo ""

# Step 1: Split the file into chunks
echo "Step 1: Splitting file into chunks..."
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Error: Input file not found: ${TRAIN_FILE}"
    exit 1
fi

FILE_SIZE=$(wc -l < "${TRAIN_FILE}")
NUM_CHUNKS=$(( (FILE_SIZE + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "  Total lines: ${FILE_SIZE}"
echo "  Number of chunks: ${NUM_CHUNKS}"
echo ""

# Check if chunks already exist (with or without .jsonl extension)
CHUNK_COUNT_JSONL=$(ls -1 ${CHUNK_DIR}/pile-corpus-chunk-*.jsonl 2>/dev/null | wc -l)
CHUNK_COUNT_NO_EXT=$(ls -1 ${CHUNK_DIR}/pile-corpus-chunk-* 2>/dev/null | grep -v "\.jsonl$" | wc -l)

# If we have chunks without .jsonl extension, rename them
if [ ${CHUNK_COUNT_NO_EXT} -gt 0 ]; then
    echo "  Found chunks without .jsonl extension, renaming..."
    for CHUNK_FILE in ${CHUNK_DIR}/pile-corpus-chunk-*; do
        # Skip if already has .jsonl extension or is a directory
        if [[ -f "$CHUNK_FILE" && ! "$CHUNK_FILE" =~ \.jsonl$ ]]; then
            mv "$CHUNK_FILE" "${CHUNK_FILE}.jsonl"
        fi
    done
    echo "  ✓ Renamed chunks to include .jsonl extension"
fi

# Check again after renaming
CHUNK_COUNT=$(ls -1 ${CHUNK_DIR}/pile-corpus-chunk-*.jsonl 2>/dev/null | wc -l)
if [ ${CHUNK_COUNT} -eq ${NUM_CHUNKS} ]; then
    echo "  Chunks already exist (${CHUNK_COUNT} chunks), skipping split..."
else
    echo "  Splitting file..."
    split -l ${CHUNK_SIZE} --numeric-suffixes=1 --suffix-length=3 \
        "${TRAIN_FILE}" "${CHUNK_DIR}/pile-corpus-chunk-" || {
        echo "Error: Failed to split file"
        exit 1
    }
    # Rename chunks to add .jsonl extension
    echo "  Adding .jsonl extension to chunks..."
    for CHUNK_FILE in ${CHUNK_DIR}/pile-corpus-chunk-*; do
        # Skip if already has .jsonl extension or is a directory
        if [[ -f "$CHUNK_FILE" && ! "$CHUNK_FILE" =~ \.jsonl$ ]]; then
            mv "$CHUNK_FILE" "${CHUNK_FILE}.jsonl"
        fi
    done
    echo "  ✓ File split into chunks with .jsonl extension"
fi
echo ""

# Step 2: Process each chunk (sequential or parallel)
echo "Step 2: Processing chunks..."
CHUNK_NUM=1
CHUNK_FILES=$(ls -1 ${CHUNK_DIR}/pile-corpus-chunk-*.jsonl | sort)

if [ "${PARALLEL_CHUNKS}" -eq 1 ]; then
    # Sequential processing
    echo "  Processing sequentially..."
    for CHUNK_FILE in ${CHUNK_FILES}; do
        CHUNK_NAME=$(basename "$CHUNK_FILE")
        CHUNK_OUTPUT="${CHUNK_OUTPUT_DIR}/${CHUNK_NAME}-tokenized"
        
        echo "  Processing chunk ${CHUNK_NUM}/${NUM_CHUNKS}: ${CHUNK_NAME}"
        
        if HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u original_tokalign/process_dataset.py \
            --model_name_or_path ${MODLE_PATH} \
            --tokenizer_name ${TOKENIZER_PATH} \
            --train_file ${CHUNK_FILE} \
            --cache_dir ${CACHE_DIR} \
            --dataset_path_in_disk ${CHUNK_OUTPUT} \
            --preprocessing_num_workers ${NUM_WORKERS} \
            --block_size ${BLOCK_SIZE} \
            --only_tokenize \
            --output_dir ./log 2>&1 | tee ./log/process_chunk_${CHUNK_NUM}.log; then
            echo "  ✓ Chunk ${CHUNK_NUM} completed"
        else
            echo "  ✗ Error processing chunk ${CHUNK_NUM}"
            exit 1
        fi
        CHUNK_NUM=$((CHUNK_NUM + 1))
    done
else
    # Parallel processing using background jobs
    echo "  Processing ${PARALLEL_CHUNKS} chunks in parallel..."
    
    # Array to store background job PIDs
    PIDS=()
    CHUNK_INDEX=0
    
    for CHUNK_FILE in ${CHUNK_FILES}; do
        CHUNK_INDEX=$((CHUNK_INDEX + 1))
        CHUNK_NAME=$(basename "$CHUNK_FILE")
        CHUNK_OUTPUT="${CHUNK_OUTPUT_DIR}/${CHUNK_NAME}-tokenized"
        
        # Wait if we've reached the parallel limit
        while [ ${#PIDS[@]} -ge ${PARALLEL_CHUNKS} ]; do
            for PID in "${PIDS[@]}"; do
                if ! kill -0 $PID 2>/dev/null; then
                    # Process finished, remove from array
                    PIDS=("${PIDS[@]/$PID}")
                    wait $PID  # Get exit status
                fi
            done
            sleep 2
        done
        
        echo "  [Chunk ${CHUNK_INDEX}/${NUM_CHUNKS}] Starting: ${CHUNK_NAME}"
        
        # Start chunk processing in background
        (
            HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u original_tokalign/process_dataset.py \
                --model_name_or_path ${MODLE_PATH} \
                --tokenizer_name ${TOKENIZER_PATH} \
                --train_file ${CHUNK_FILE} \
                --cache_dir ${CACHE_DIR} \
                --dataset_path_in_disk ${CHUNK_OUTPUT} \
                --preprocessing_num_workers ${NUM_WORKERS} \
                --block_size ${BLOCK_SIZE} \
                --only_tokenize \
                --output_dir ./log > ./log/process_chunk_${CHUNK_INDEX}.log 2>&1
            
            if [ $? -eq 0 ]; then
                echo "  [Chunk ${CHUNK_INDEX}] ✓ Completed: ${CHUNK_NAME}"
            else
                echo "  [Chunk ${CHUNK_INDEX}] ✗ Error: ${CHUNK_NAME}"
                exit 1
            fi
        ) &
        
        PID=$!
        PIDS+=($PID)
    done
    
    # Wait for all remaining background jobs
    echo "  Waiting for all chunks to complete..."
    FAILED=0
    for PID in "${PIDS[@]}"; do
        wait $PID || FAILED=1
    done
    
    if [ $FAILED -eq 1 ]; then
        echo "  ✗ One or more chunks failed. Check logs in ./log/process_chunk_*.log"
        exit 1
    fi
    
    echo ""
    echo "  ✓ All chunks processed"
fi

echo ""

# Step 3: Concatenate all chunks
echo "Step 3: Concatenating tokenized chunks..."
python -u datasets/concatenate_datasets.py \
    --input-dir ${CHUNK_OUTPUT_DIR} \
    --output-path ${DATASET_PATH} \
    2>&1 | tee ./log/concatenate_datasets.log

if [ $? -ne 0 ]; then
    echo "✗ Error concatenating datasets"
    exit 1
fi

echo ""
echo "✓ Datasets concatenated"
echo ""

# Step 4: Cleanup (optional)
echo "Step 4: Cleanup..."
read -p "Delete chunk files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf ${CHUNK_DIR}
    rm -rf ${CHUNK_OUTPUT_DIR}
    echo "✓ Chunk files deleted"
else
    echo "  Chunk files kept in:"
    echo "    - ${CHUNK_DIR}"
    echo "    - ${CHUNK_OUTPUT_DIR}"
fi

echo ""
echo "=========================================="
echo "Tokenization complete!"
echo "Output: ${DATASET_PATH}"
echo "=========================================="

