#!/usr/bin/env python
# coding=utf-8
"""
Concatenate multiple tokenized datasets into a single dataset.
Used for combining chunked tokenization results.
"""

import argparse
import os
from datasets import load_from_disk, concatenate_datasets

def main():
    parser = argparse.ArgumentParser(description="Concatenate tokenized datasets")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing tokenized dataset chunks"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for concatenated dataset"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Concatenating Tokenized Datasets")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output path: {args.output_path}")
    print("="*60)
    print("")
    
    # Find all chunk directories
    chunk_dirs = []
    for item in sorted(os.listdir(args.input_dir)):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a tokenized chunk directory
            # It might be named like "pile-corpus-chunk-001-tokenized" or just be a directory
            if "-tokenized" in item or os.path.exists(os.path.join(item_path, "dataset_info.json")):
                chunk_dirs.append(item_path)
    
    if not chunk_dirs:
        raise ValueError(f"No tokenized dataset chunks found in {args.input_dir}")
    
    print(f"Found {len(chunk_dirs)} dataset chunks")
    print("")
    
    # Load each chunk
    datasets = []
    for i, chunk_dir in enumerate(chunk_dirs, 1):
        print(f"Loading chunk {i}/{len(chunk_dirs)}: {os.path.basename(chunk_dir)}")
        try:
            ds = load_from_disk(chunk_dir)
            # Handle both dict and Dataset formats
            if isinstance(ds, dict):
                if "train" in ds:
                    datasets.append(ds["train"])
                else:
                    datasets.append(ds[list(ds.keys())[0]])
            else:
                datasets.append(ds)
            print(f"  ✓ Loaded {len(datasets[-1]):,} examples")
        except Exception as e:
            print(f"  ✗ Error loading {chunk_dir}: {e}")
            raise
    
    print("")
    print("Concatenating datasets...")
    
    # Concatenate all datasets
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = concatenate_datasets(datasets)
    
    print(f"✓ Concatenated into {len(combined_dataset):,} total examples")
    print("")
    
    # Create train/validation split (5% validation, 95% train)
    print("Creating train/validation split (5% validation)...")
    split_dataset = combined_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"]
    
    print(f"  Train examples: {len(train_dataset):,}")
    print(f"  Validation examples: {len(validation_dataset):,}")
    print("")
    
    # Create output directory structure with both train and validation splits
    from datasets import DatasetDict
    output_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })
    
    print(f"Saving to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    output_dict.save_to_disk(args.output_path)
    
    print("")
    print("="*60)
    print("Concatenation Complete!")
    print(f"Total examples: {len(combined_dataset):,}")
    print(f"Output: {args.output_path}")
    print("="*60)

if __name__ == "__main__":
    main()

