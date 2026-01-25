#!/usr/bin/env python
# coding=utf-8
"""
Prepare Pile corpus for post-alignment fine-tuning.
Based on TokAlign paper: uses vanilla pretraining corpus Pile from Pythia.
Uses Common Pile (comma_v0.1_training_dataset) as alternative to EleutherAI/pile
since the original is no longer available from the-eye.eu.
"""

import json
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Prepare The Pile corpus for fine-tuning")
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="common-pile/comma_v0.1_training_dataset",
        help="Pile dataset name on HuggingFace (default: common-pile/comma_v0.1_training_dataset, alternative to EleutherAI/pile)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for datasets"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None, process all)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Preparing Pile Corpus (Common Pile)")
    print("="*60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_path}")
    print(f"Streaming: {args.streaming}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("="*60)
    
    # Load The Pile dataset
    print(f"\nLoading dataset: {args.dataset_name}...")
    try:
        if args.streaming:
            dataset = load_dataset(
                args.dataset_name,
                split=args.split,
                cache_dir=args.cache_dir,
                streaming=True,
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                args.dataset_name,
                split=args.split,
                cache_dir=args.cache_dir,
                streaming=False,
                trust_remote_code=True
            )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative loading method...")
        try:
            dataset = load_dataset(
                args.dataset_name,
                cache_dir=args.cache_dir,
                streaming=args.streaming,
                trust_remote_code=True
            )
            if isinstance(dataset, dict):
                if args.split in dataset:
                    dataset = dataset[args.split]
                else:
                    dataset = dataset[list(dataset.keys())[0]]
        except Exception as e2:
            print(f"Failed to load dataset: {e2}")
            raise
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    
    # Process and write to JSONL
    print(f"\nWriting to {args.output_path}...")
    
    count = 0
    with open(args.output_path, "w", encoding="utf-8") as f:
        if args.streaming:
            # Streaming mode
            for example in tqdm(dataset, desc="Processing examples"):
                if args.max_samples and count >= args.max_samples:
                    break
                
                # Extract text and meta fields
                text = example.get("text", "")
                meta = example.get("meta", {})
                
                # If meta is None or empty, try to get pile_set_name
                if not meta:
                    meta = {}
                    if "pile_set_name" in example:
                        meta["pile_set_name"] = example["pile_set_name"]
                
                # Write JSONL entry
                json.dump({"text": text, "meta": meta}, f, ensure_ascii=False)
                f.write("\n")
                count += 1
        else:
            # Non-streaming mode
            total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
            
            for i in tqdm(range(total), desc="Processing examples"):
                example = dataset[i]
                
                # Extract text and meta fields
                text = example.get("text", "")
                meta = example.get("meta", {})
                
                # If meta is None or empty, try to get pile_set_name
                if not meta:
                    meta = {}
                    if "pile_set_name" in example:
                        meta["pile_set_name"] = example["pile_set_name"]
                
                # Write JSONL entry
                json.dump({"text": text, "meta": meta}, f, ensure_ascii=False)
                f.write("\n")
                count += 1
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Processed examples: {count:,}")
    print(f"  Output file: {args.output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

