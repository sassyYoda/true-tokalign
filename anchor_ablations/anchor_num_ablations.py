import numpy as np
import json
from tqdm import tqdm
import random
import argparse
import os

# Import functions from cal_trans_matrix.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from original_tokalign.cal_trans_matrix import (
    load_glove_model,
    convert2matrix,
    top_k_indices,
    convert2relative_rep
)

def generate_alignment_matrix(
    embed1,
    embed2,
    t2l_supl,
    g_vocab_len1,
    g_vocab_len2,
    ids1,
    rep1,
    ids2,
    rep2,
):
    """Generate alignment matrix from relative representations."""
    sim = np.matmul(rep1, rep2.T)
    
    # Debug: Verify similarity matrix dimensions
    print(f"\nSimilarity matrix shape: {sim.shape}")
    print(f"Expected: ({len(ids1)}, {len(ids2)})")
    
    # Debug: Sample some similarity values to verify they're reasonable
    if sim.size > 0:
        print(f"Similarity matrix stats:")
        print(f"  Min: {sim.min():.4f}, Max: {sim.max():.4f}, Mean: {sim.mean():.4f}")
        print(f"  Sample values (first 5x5):\n{sim[:5, :5]}")
    
    td = {}
    tids = [str(tid) for tid in range(g_vocab_len1)]
    gold_count = 0
    random_count = 0
    similarity_count = 0
    
    for tid in tqdm(tids, desc="Get the max prob target idx"):
        # gold label
        if tid in t2l_supl:
            td[tid] = t2l_supl[tid]
            gold_count += 1
            continue

        # missing token id: random pick
        if tid not in ids1:
            td[tid] = random.randint(0, g_vocab_len2-1)
            random_count += 1
            continue

        # Use similarity matrix
        id1_idx = ids1.index(tid)
        lix = np.argmax(sim[id1_idx])
        lid = ids2[lix]

        # back to the second top id
        if lid == 'unk' or lid == '<unk>':
            lix = set(top_k_indices(sim[id1_idx], 2)) - set(top_k_indices(sim[id1_idx], 1))
            lid = ids2[lix.pop()]

        td[tid] = int(lid)
        similarity_count += 1

    total = len(tids)
    print(f"\nAlignment statistics:")
    print(f"  Total tokens: {total}")
    print(f"  Gold mappings: {gold_count} ({100*gold_count/total:.1f}%)")
    print(f"  Random assignments: {random_count} ({100*random_count/total:.1f}%)")
    print(f"  Similarity-based: {similarity_count} ({100*similarity_count/total:.1f}%)")
    print(f"  Note: Only similarity-based tokens use the relative representation!")
    
    return td

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-glove-vector-path", type=str, default="./data/glove_vec.pythia.txt")
    parser.add_argument("-s1", "--source-vocab-size", type=int, default=50304)
    parser.add_argument("-t", "--target-glove-vector-path", type=str, default="./data/glove_vec.qwen2-7b.txt")
    parser.add_argument("-s2", "--target-vocab-size", type=int, default=151646)
    parser.add_argument("-g", "--gold-target-to-source-path", type=str, default="./data/Vocab_count/qwen2-7b2pythia.json")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for anchor selection")
    parser.add_argument("--anchor-numbers", type=int, nargs='+', default=[100, 300, 500, 1000],
                       help="List of anchor numbers to test (default: 100 300 500 1000)")
    parser.add_argument("-o", "--output-dir", type=str, default="./anchor_num_evaluations")

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

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Generate alignment matrices for different numbers of anchors
    for num_anchors in args.anchor_numbers:
        print(f"\n{'='*60}")
        print(f"Generating alignment matrix with {num_anchors} anchors")
        print(f"{'='*60}\n")

        # Generate relative representations with this number of anchors
        ids1, rep1, ids2, rep2 = convert2relative_rep(
            embed1=embed1,
            embed2=embed2,
            gold=t2l_supl,
            num_pivot=num_anchors,
            seed=args.seed
        )
        
        # Debug: Verify dimensions
        print(f"Relative representation dimensions:")
        print(f"  rep1 shape: {rep1.shape} (should be (vocab_size, {num_anchors}))")
        print(f"  rep2 shape: {rep2.shape} (should be (vocab_size, {num_anchors}))")
        print(f"  ids1 length: {len(ids1)}")
        print(f"  ids2 length: {len(ids2)}")

        # Generate alignment matrix
        td = generate_alignment_matrix(
            embed1=embed1,
            embed2=embed2,
            t2l_supl=t2l_supl,
            g_vocab_len1=g_vocab_len1,
            g_vocab_len2=g_vocab_len2,
            ids1=ids1,
            rep1=rep1,
            ids2=ids2,
            rep2=rep2
        )

        # Save alignment matrix
        output_path = os.path.join(output_dir, f"alignment_matrix_anchors_{num_anchors}.json")
        with open(output_path, "w") as f:
            json.dump(td, f, indent="\t")
        
        print(f"Saved alignment matrix to {output_path}")

    print(f"\n{'='*60}")
    print(f"Completed generating {len(args.anchor_numbers)} alignment matrices")
    print(f"Anchor numbers tested: {args.anchor_numbers}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
