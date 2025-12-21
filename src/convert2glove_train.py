import datasets
import argparse
from tqdm import tqdm

def convert2train(
    src_path = "llama-3-tok-20GB_tk",
    tgt_path = "llama-3-tok-20GB_tk-train-GloVe",
    key = "train",
    min_line_len = 15,
    max_line = 1000000
):
    print(f"Loading dataset from {src_path}...")
    d = datasets.load_from_disk(src_path)[key]
    total_examples = min(max_line, len(d))
    print(f"Processing {total_examples:,} examples (min length: {min_line_len})...")

    written = 0
    with open(tgt_path, "w") as f:
        for i in tqdm(range(total_examples), desc="Extracting token IDs"):
            data = d[i]["input_ids"]
            if len(data) < min_line_len:
                continue
            line = " ".join([f"{cd}" for cd in data])
            f.write(line+"\n")
            written += 1
    
    print(f"Written {written:,} lines to {tgt_path}")

def convert2eval(
    src_tok_path = "llama-3-tok-20GB_tk",
    tgt_tok_path = "gemma-tok-20GB_tk",
    file_path = "llama3-2-gemma-MX1k-eval",
    key = "validation",
    min_line_len = 15,
    max_line = 1000,
):
    print(f"Loading source dataset from {src_tok_path}...")
    d1 = datasets.load_from_disk(src_tok_path)
    print(f"Loading target dataset from {tgt_tok_path}...")
    d2 = datasets.load_from_disk(tgt_tok_path)
    
    # Check if the key exists in both datasets
    if key not in d1:
        raise KeyError(f"Key '{key}' not found in source dataset. Available keys: {list(d1.keys())}")
    if key not in d2:
        raise KeyError(f"Key '{key}' not found in target dataset. Available keys: {list(d2.keys())}")
    
    len1 = len(d1[key])
    len2 = len(d2[key])
    
    if len1 != len2:
        print(f"\n{'='*70}")
        print(f"WARNING: Dataset length mismatch detected!")
        print(f"{'='*70}")
        print(f"Source dataset ({src_tok_path}):")
        print(f"  - '{key}' split: {len1:,} examples")
        if len(d1[key]) > 0:
            sample1 = d1[key][0]
            if "input_ids" in sample1:
                print(f"  - Sample token length: {len(sample1['input_ids'])}")
        print(f"\nTarget dataset ({tgt_tok_path}):")
        print(f"  - '{key}' split: {len2:,} examples")
        if len(d2[key]) > 0:
            sample2 = d2[key][0]
            if "input_ids" in sample2:
                print(f"  - Sample token length: {len(sample2['input_ids'])}")
        print(f"\n{'='*70}")
        print("POSSIBLE CAUSES:")
        print("1. Datasets were created with old code that concatenated batches")
        print("2. Different filtering removed different examples (empty/too long)")
        print("3. Datasets were created from different source corpora")
        print("\nWith the updated process_dataset.py, each example should correspond")
        print("to the same original text (variable-length chunks, one per example).")
        print(f"{'='*70}\n")
        print(f"Will process only the first {min(len1, len2):,} examples.")
        if len1 == len2:
            print("✓ Datasets have matching lengths - alignment should be preserved!")
        else:
            print("⚠ Length mismatch - ensure datasets were created with updated code.")
    
    total_examples = min(max_line, len1, len2)
    print(f"Processing {total_examples:,} aligned examples (min length: {min_line_len})...")

    written = 0
    with open(file_path, "w") as f:
        for i in tqdm(range(total_examples), desc="Extracting aligned token IDs"):
            data1 = d1[key][i]["input_ids"]
            data2 = d2[key][i]["input_ids"]
            if len(data1) < min_line_len or len(data2) < min_line_len:
                continue

            line1 = " ".join([f"{cd}" for cd in data1])
            line2 = " ".join([f"{cd}" for cd in data2])

            f.write(f"{line1}\t{line2}\n")
            written += 1
    
    print(f"Written {written:,} aligned lines to {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", type=str, default="./data/pretrain-dataset/multilingual-pythia-tok_tk")
    parser.add_argument("-t", "--target-path", type=str, default="./data/pretrain-dataset/multilingual-qwen2-7b-tok_tk")
    parser.add_argument("-k", "--key", type=str, default="train")
    parser.add_argument("-m", "--min-line-length", type=int, default=15)
    parser.add_argument("-l", "--max-line", type=int, default=10000000)
    parser.add_argument("-o", "--output-path", type=str, default="./data/pretrain-dataset/pythia-tok_train-GloVe")

    args = parser.parse_args()

    if args.key == "train":
        convert2train(
            src_path = args.source_path,
            tgt_path = args.output_path,
            key = "train",
            min_line_len = args.min_line_length,
            max_line = args.max_line
        )
    elif args.key == "valid" or args.key == "validation":
        convert2eval(
            src_tok_path = args.source_path,
            tgt_tok_path = args.target_path,
            file_path = args.output_path,
            key = args.key,
            min_line_len = args.min_line_length,
            max_line = args.max_line,
        )
    else:
        raise Exception(f"Method of {args.key} is not implemented.")

    # convert2train(
    #     src_path = "./data/pretrain-dataset/multilingual-gemma-tok_tk",
    #     tgt_path = "./data/pretrain-dataset/gemma-tok_train-GloVe",
    #     key = "train",
    #     min_line_len = 15,
    #     max_line = 10000000
    # )

    # convert2train(
    #     src_path = "./data/pretrain-dataset/multilingual-pythia-tok_tk",
    #     tgt_path = "./data/pretrain-dataset/pythia-tok_train-GloVe",
    #     key = "train",
    #     min_line_len = 15,
    #     max_line = 10000000
    # )
