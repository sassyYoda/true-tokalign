#!/usr/bin/env python3
"""
Download and prepare the PubMed abstract dataset for token alignment training.

This script downloads the pubmed-abstract dataset from HuggingFace and formats it
in the same JSONL format required by the token alignment pipeline.
Each line is a JSON object with a "text" field containing the abstract.
"""

import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm


def download_and_prepare_pubmed(
    output_path: str,
    max_samples: int = None,
    revision: str = None,
):
    """
    Download and prepare the PubMed abstract dataset.
    
    Args:
        output_path: Path to save the formatted JSONL file
        max_samples: Maximum number of samples to include (None for all)
        revision: Specific dataset revision/date tag (None for latest)
    """
    print("Loading PubMed abstract dataset from HuggingFace...")
    
    # Load the dataset
    if revision:
        print(f"Loading specific revision: {revision}")
        dataset = load_dataset("uiyunkim-hub/pubmed-abstract", revision=revision)
    else:
        print("Loading latest version...")
        dataset = load_dataset("uiyunkim-hub/pubmed-abstract")
    
    # Get the train split
    train_data = dataset["train"]
    
    print(f"Dataset loaded. Total samples: {len(train_data)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Process and save the dataset
    print(f"Formatting and saving to {output_path}...")
    
    num_samples = len(train_data)
    if max_samples:
        num_samples = min(num_samples, max_samples)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(train_data, total=num_samples, desc="Processing abstracts")):
            if max_samples and i >= max_samples:
                break
            
            # Extract abstract text
            abstract = item.get("abstract", "")
            
            # Skip empty abstracts
            if not abstract or not abstract.strip():
                continue
            
            # Format as JSONL: each line is a JSON object with "text" field
            json_obj = {"text": abstract}
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Successfully saved {num_samples} abstracts to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare PubMed abstract dataset for token alignment"
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        default="./data/pretrain-corpus/pubmed-abstract.json",
        help="Output path for the formatted JSONL file"
    )
    parser.add_argument(
        "-m", "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to include (None for all)"
    )
    parser.add_argument(
        "-r", "--revision",
        type=str,
        default=None,
        help="Specific dataset revision/date tag (e.g., '2025-03-28'). None for latest"
    )
    
    args = parser.parse_args()
    
    download_and_prepare_pubmed(
        output_path=args.output_path,
        max_samples=args.max_samples,
        revision=args.revision,
    )

