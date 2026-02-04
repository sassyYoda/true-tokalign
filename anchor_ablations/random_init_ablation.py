import numpy as np
import json
from tqdm import tqdm
import random
import argparse
import os

# Import functions from cal_trans_matrix.py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from original_tokalign.cal_trans_matrix import (
    load_glove_model,
    convert2matrix
)

def generate_random_alignment_matrix(
    embed1,
    embed2,
    t2l_supl,
    g_vocab_len1,
    g_vocab_len2,
    ids1,
    ids2,
    seed=0,
):
    """Generate alignment matrix with random mappings (except gold mappings)."""
    random.seed(seed)
    
    td = {}
    tids = [str(tid) for tid in range(g_vocab_len1)]
    gold_count = 0
    random_count = 0
    
    for tid in tqdm(tids, desc="Generating random alignment"):
        # gold label - keep one-to-one mapping
        if tid in t2l_supl:
            td[tid] = t2l_supl[tid]
            gold_count += 1
            continue

        # missing token id: random pick
        if tid not in ids1:
            td[tid] = random.randint(0, g_vocab_len2-1)
            random_count += 1
            continue

        # Random mapping for tokens in vocab
        td[tid] = random.randint(0, g_vocab_len2-1)
        random_count += 1

    total = len(tids)
    print(f"\nAlignment statistics:")
    print(f"  Total tokens: {total}")
    print(f"  Gold mappings: {gold_count} ({100*gold_count/total:.1f}%)")
    print(f"  Random assignments: {random_count} ({100*random_count/total:.1f}%)")
    print(f"  Note: All non-gold tokens are randomly mapped!")
    
    return td

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-glove-vector-path", type=str, default="./data/glove_vec.pythia.txt")
    parser.add_argument("-s1", "--source-vocab-size", type=int, default=50304)
    parser.add_argument("-t", "--target-glove-vector-path", type=str, default="./data/glove_vec.qwen2-7b.txt")
    parser.add_argument("-s2", "--target-vocab-size", type=int, default=151646)
    parser.add_argument("-g", "--gold-target-to-source-path", type=str, default="./data/Vocab_count/qwen2-7b2pythia.json")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for random mapping")
    parser.add_argument("-o", "--output-dir", type=str, default="./random_init_evaluations")
    parser.add_argument("--output-name", type=str, default="alignment_matrix_random_init.json",
                       help="Output filename (default: alignment_matrix_random_init.json)")

    args = parser.parse_args()

    # new tokenizer glove path
    g_p1 = args.target_glove_vector_path
    g_vocab_len1 = args.target_vocab_size

    # old tokenizer glove path
    g_p2 = args.source_glove_vector_path
    g_vocab_len2 = args.source_vocab_size

    # gold src to tgt id transition json dict
    t2l_supl_path = args.gold_target_to_source_path

    with open(t2l_supl_path, "r") as f:
        t2l_supl = json.load(f)

    print("Loading GloVe models...")
    embed1 = load_glove_model(g_p1)
    embed2 = load_glove_model(g_p2)

    # Get vocab IDs from GloVe models (same as anchor_num_ablations.py)
    ids1, _ = convert2matrix(embed1)
    ids2, _ = convert2matrix(embed2)

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating random initialization alignment matrix")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Generate random alignment matrix
    td = generate_random_alignment_matrix(
        embed1=embed1,
        embed2=embed2,
        t2l_supl=t2l_supl,
        g_vocab_len1=g_vocab_len1,
        g_vocab_len2=g_vocab_len2,
        ids1=ids1,
        ids2=ids2,
        seed=args.seed
    )

    # Save alignment matrix
    output_path = os.path.join(output_dir, args.output_name)
    with open(output_path, "w") as f:
        json.dump(td, f, indent="\t")
    
    print(f"\nSaved alignment matrix to {output_path}")
    print(f"\n{'='*60}")
    print(f"Random initialization alignment matrix generated")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
