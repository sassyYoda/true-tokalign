#!/usr/bin/env python
# coding=utf-8
"""
Generate random baseline alignment matrices for comparison.
Creates:
1. Random permutation: randomly maps each target token to a random source token
2. Random initialization: creates a random mapping (similar to permutation but can be used differently)
"""

import json
import argparse
import random
from transformers import AutoTokenizer

def generate_random_permutation_alignment(
    source_tokenizer_path,
    target_tokenizer_path,
    output_path,
    seed=42
):
    """
    Generate a random permutation alignment matrix.
    Each target token ID is randomly mapped to a source token ID.
    """
    print(f"Generating random permutation alignment...")
    print(f"  Source tokenizer: {source_tokenizer_path}")
    print(f"  Target tokenizer: {target_tokenizer_path}")
    print(f"  Seed: {seed}")
    
    random.seed(seed)
    
    # Load tokenizers to get vocab sizes
    src_tok = AutoTokenizer.from_pretrained(source_tokenizer_path, trust_remote_code=True)
    tgt_tok = AutoTokenizer.from_pretrained(target_tokenizer_path, trust_remote_code=True)
    
    src_vocab_size = len(src_tok)
    tgt_vocab_size = len(tgt_tok)
    
    print(f"  Source vocab size: {src_vocab_size}")
    print(f"  Target vocab size: {tgt_vocab_size}")
    
    # Create random mapping: target_id -> random_source_id
    alignment = {}
    for tgt_id in range(tgt_vocab_size):
        src_id = random.randint(0, src_vocab_size - 1)
        alignment[str(tgt_id)] = src_id
    
    # Save alignment matrix
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alignment, f, indent=2)
    
    print(f"✓ Random permutation alignment saved to: {output_path}")
    print(f"  Total mappings: {len(alignment)}")
    
    return alignment

def generate_random_initialization_alignment(
    source_tokenizer_path,
    target_tokenizer_path,
    output_path,
    seed=42
):
    """
    Generate a random initialization alignment matrix.
    Similar to permutation but can be used for different initialization strategies.
    For now, same as permutation but kept separate for clarity.
    """
    print(f"Generating random initialization alignment...")
    print(f"  Source tokenizer: {source_tokenizer_path}")
    print(f"  Target tokenizer: {target_tokenizer_path}")
    print(f"  Seed: {seed}")
    
    # For initialization baseline, we can use the same random mapping
    # In practice, initialization might use different strategies, but for evaluation
    # purposes, random mapping is a reasonable baseline
    return generate_random_permutation_alignment(
        source_tokenizer_path,
        target_tokenizer_path,
        output_path,
        seed
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate random baseline alignment matrices")
    parser.add_argument(
        "-s", "--source-tokenizer-path",
        type=str,
        default="EleutherAI/pythia-1b",
        help="Source tokenizer path (default: EleutherAI/pythia-1b)"
    )
    parser.add_argument(
        "-t", "--target-tokenizer-path",
        type=str,
        default="Qwen/Qwen2-7B",
        help="Target tokenizer path (default: Qwen/Qwen2-7B)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./data/pythia2qwen2-7b",
        help="Output directory for alignment matrices (default: ./data/pythia2qwen2-7b)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--baseline-type",
        type=str,
        choices=["permutation", "initialization", "both"],
        default="both",
        help="Type of baseline to generate: permutation, initialization, or both (default: both)"
    )
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.baseline_type in ["permutation", "both"]:
        perm_output = os.path.join(args.output_dir, "align_matrix_random_permutation.json")
        generate_random_permutation_alignment(
            args.source_tokenizer_path,
            args.target_tokenizer_path,
            perm_output,
            seed=args.seed
        )
    
    if args.baseline_type in ["initialization", "both"]:
        init_output = os.path.join(args.output_dir, "align_matrix_random_initialization.json")
        generate_random_initialization_alignment(
            args.source_tokenizer_path,
            args.target_tokenizer_path,
            init_output,
            seed=args.seed + 1  # Use different seed for initialization
        )
    
    print("\n✓ Baseline alignment matrices generated successfully!")

