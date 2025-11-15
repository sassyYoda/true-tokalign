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
from transformers import AutoTokenizer


def download_and_prepare_pubmed(
    output_path: str,
    max_samples: int = None,
    max_tokens: int = None,
    tokenizer_path: str = "EleutherAI/pythia-1b",
    revision: str = None,
):
    """
    Download and prepare the PubMed abstract dataset.
    
    Args:
        output_path: Path to save the formatted JSONL file
        max_samples: Maximum number of samples to include (None for all)
        max_tokens: Maximum number of tokens to include (None for all)
        tokenizer_path: Path to tokenizer for counting tokens
        revision: Specific dataset revision/date tag (None for latest)
    """
    print("Loading PubMed abstract dataset from HuggingFace...")
    
    # Load tokenizer if we need to count tokens
    tokenizer = None
    if max_tokens:
        print(f"Loading tokenizer from {tokenizer_path} for token counting...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"Target: {max_tokens:,} tokens")
    
    # Load the dataset
    if revision:
        print(f"Loading specific revision: {revision}")
        dataset = load_dataset("uiyunkim-hub/pubmed-abstract", revision=revision)
    else:
        print("Loading latest version...")
        dataset = load_dataset("uiyunkim-hub/pubmed-abstract")
    
    # Get the train split
    train_data = dataset["train"]
    
    print(f"Dataset loaded. Total samples: {len(train_data):,}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Process and save the dataset
    print(f"Formatting and saving to {output_path}...")
    
    total_tokens = 0
    samples_written = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(train_data, desc="Processing abstracts"):
            # Check max_samples limit
            if max_samples and samples_written >= max_samples:
                break
            
            # Extract abstract text
            abstract = item.get("abstract", "")
            
            # Skip empty abstracts
            if not abstract or not abstract.strip():
                continue
            
            # Count tokens if needed
            if max_tokens:
                # Tokenize to count tokens
                tokens = tokenizer.encode(abstract, add_special_tokens=False)
                token_count = len(tokens)
                
                # Check if adding this would exceed the limit
                if total_tokens + token_count > max_tokens:
                    print(f"\nReached token limit: {total_tokens:,} / {max_tokens:,} tokens")
                    break
                
                total_tokens += token_count
            
            # Format as JSONL: each line is a JSON object with "text" field
            json_obj = {"text": abstract}
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            samples_written += 1
    
    print(f"\n✓ Successfully saved {samples_written:,} abstracts to {output_path}")
    if max_tokens:
        print(f"  Total tokens: {total_tokens:,} / {max_tokens:,}")
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
        "-t", "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to include (None for all). Uses Pythia tokenizer for counting."
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="EleutherAI/pythia-1b",
        help="Tokenizer path for counting tokens (default: EleutherAI/pythia-1b)"
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
        max_tokens=args.max_tokens,
        tokenizer_path=args.tokenizer_path,
        revision=args.revision,
    )

