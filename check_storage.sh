#!/bin/bash

# Storage analysis script for TokAlign project

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

echo "=========================================="
echo "Storage Usage Analysis"
echo "=========================================="
echo ""

# Overall project size
echo "=== TOTAL PROJECT SIZE ==="
du -sh . 2>/dev/null | awk '{print "Total: " $1}'
echo ""

# Breakdown by main directories
echo "=== BREAKDOWN BY DIRECTORY ==="
echo "Data directory:"
if [ -d "./data" ]; then
    du -sh ./data/* 2>/dev/null | sort -h | head -20
    echo ""
    echo "  Detailed breakdown:"
    echo "  - Pretrain corpus:"
    du -sh ./data/pretrain-corpus/* 2>/dev/null | sort -h | head -10
    echo "  - Pretrain datasets:"
    du -sh ./data/pretrain-dataset/* 2>/dev/null | sort -h | head -10
    echo "  - Model checkpoints:"
    du -sh ./data/pythia2qwen2-7b/* 2>/dev/null | sort -h 2>/dev/null || echo "    (none)"
    echo "  - Cache:"
    du -sh ./data/cache 2>/dev/null || echo "    (none)"
else
    echo "  (data directory not found)"
fi
echo ""

echo "Log directory:"
if [ -d "./log" ]; then
    du -sh ./log/* 2>/dev/null | sort -h | head -20
    echo ""
    echo "  Training checkpoints:"
    find ./log -type d -name "checkpoint-*" -exec du -sh {} \; 2>/dev/null | sort -h
else
    echo "  (log directory not found)"
fi
echo ""

echo "=== LARGEST FILES ==="
find . -type f -size +1G -exec du -h {} \; 2>/dev/null | sort -rh | head -10
echo ""

echo "=== SUMMARY ==="
echo "Chunk files:"
du -sh ./data/pretrain-corpus/chunks 2>/dev/null || echo "  (none)"
echo "Tokenized chunks:"
du -sh ./data/pretrain-dataset/chunks-tokenized 2>/dev/null || echo "  (none)"
echo "Final tokenized dataset:"
du -sh ./data/pretrain-dataset/pile00-qwen2-7b-tokenized 2>/dev/null || echo "  (none)"
echo "Final models:"
du -sh ./log/1b/0_qwen2-7b_S*/checkpoint-2500 2>/dev/null | head -2 || echo "  (none)"
echo ""

echo "=========================================="
echo "Analysis Complete"
echo "=========================================="

