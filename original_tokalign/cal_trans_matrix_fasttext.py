"""
FastText-specific version of cal_trans_matrix.py
Reuses all functions from cal_trans_matrix.py except for loading which uses FastText format.
"""

import numpy as np
import json
from tqdm import tqdm
import random
import argparse
import os

# Import shared functions from cal_trans_matrix.py
from cal_trans_matrix import (
    normalize,
    convert2matrix,
    top_k_indices,
    get_pivot_matrix,
    convert2rel_rep,
    convert2relative_rep,
    load_fasttext_model  # FastText-specific loader
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-glove-vector-path", type=str, default="./data/glove_vec.pythia.txt",
                       help="Source FastText vector path (using same arg name for compatibility)")
    parser.add_argument("-s1", "--source-vocab-size", type=int, default=50304)
    parser.add_argument("-t", "--target-glove-vector-path", type=str, default="./data/glove_vec.qwen2-7b.txt",
                       help="Target FastText vector path (using same arg name for compatibility)")
    parser.add_argument("-s2", "--target-vocab-size", type=int, default=151646)
    parser.add_argument("-g", "--gold-target-to-source-path", type=str, default="./data/Vocab_count/qwen2-7b2pythia.json")
    parser.add_argument("-r", "--relative-representation", action='store_true')
    parser.add_argument("-v", "--vanilla-representation", action='store_true')
    parser.add_argument("-n", "--pivotal-token-number", type=int, default=300)
    parser.add_argument("-o", "--output-path", type=str, default="./data/pythia2qwen2-7b/glove_fasttext.json")

    args = parser.parse_args()

    # new tokenizer FastText path
    g_p1 = args.target_glove_vector_path
    g_vocab_len1 = args.target_vocab_size

    # old tokenizer FastText path
    g_p2 = args.source_glove_vector_path
    g_vocab_len2 = args.source_vocab_size

    tgt_path = args.output_path

    # gold src to tgt id transition json dict
    t2l_supl_path = args.gold_target_to_source_path

    with open(t2l_supl_path, "r") as f:
        t2l_supl = json.load(f)

    # Load FastText models (instead of GloVe)
    print("Loading FastText models...")
    embed1 = load_fasttext_model(g_p1)
    embed2 = load_fasttext_model(g_p2)

    if args.relative_representation:
        ids1, rep1, ids2, rep2 = convert2relative_rep(
            embed1=embed1,
            embed2=embed2,
            gold=t2l_supl,
            num_pivot=args.pivotal_token_number
        )
    elif args.vanilla_representation:
        # Calculate the transition matrix
        ids1, rep1 = convert2matrix(embed1)
        ids2, rep2 = convert2matrix(embed2)
    else:
        raise Exception(f"Only relative and vanilla representation are implemented.")

    sim = np.matmul(rep1, rep2.T)

    td = {}
    tids = [str(tid) for tid in range(g_vocab_len1)]
    supl_id = 0
    for tid in tqdm(tids, desc="Get the max prob target idx"):
        # gold label
        if tid in t2l_supl:
            td[tid] = t2l_supl[tid]
            supl_id += 1
            continue

        # missing token id: random pick
        if tid not in ids1:
            td[tid] = random.randint(0, g_vocab_len2-1)
            supl_id += 1
            continue

        id1_idx = ids1.index(tid)
        lix = np.argmax(sim[id1_idx])
        lid = ids2[lix]

        # assert(lid != 'unk'), tid
        # back to the second top id
        if lid == 'unk' or lid == '<unk>':
            lix = set(top_k_indices(sim[id1_idx], 2)) - set(top_k_indices(sim[id1_idx], 1))
            lid = ids2[lix.pop()]

        td[tid] = int(lid)

    print(f"{supl_id} ids are suppled with gold transition dictionary.")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(tgt_path)
    if output_dir:  # Only create directory if path contains a directory component
        os.makedirs(output_dir, exist_ok=True)

    with open(tgt_path, "w") as f:
        json.dump(td, f, indent="\t")
