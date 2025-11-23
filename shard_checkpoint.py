#!/usr/bin/env python
# coding=utf-8
"""
Shard a PyTorch model checkpoint into smaller files.
Used for models >10GB to avoid loading issues.
Uses HuggingFace's built-in sharding functionality.
"""

import argparse
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

def shard_checkpoint(output_dir, max_shard_size_gb=10):
    """
    Shard a pytorch_model.bin file into smaller shards using HuggingFace's method.
    
    Args:
        output_dir: Directory containing the checkpoint
        max_shard_size_gb: Maximum size per shard in GB (default: 10GB)
    """
    output_dir = Path(output_dir)
    model_file = output_dir / "pytorch_model.bin"
    
    # Check if already sharded
    index_file = output_dir / "pytorch_model.bin.index.json"
    if index_file.exists():
        print(f"Model is already sharded (index file exists). Skipping.")
        return
    
    if not model_file.exists():
        print(f"No pytorch_model.bin found in {output_dir}. Model may already be sharded or saved differently.")
        return
    
    # Calculate file size
    file_size_gb = model_file.stat().st_size / (1024 ** 3)
    print(f"Model file size: {file_size_gb:.2f} GB")
    
    if file_size_gb <= max_shard_size_gb:
        print(f"Model is smaller than {max_shard_size_gb}GB, no sharding needed.")
        return
    
    print(f"Sharding model (>{max_shard_size_gb}GB)...")
    
    try:
        # Load model and resave with sharding
        # This uses HuggingFace's built-in sharding
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(output_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        print("Saving with sharding...")
        # Save with max_shard_size parameter (in bytes)
        max_shard_size = int(max_shard_size_gb * 1024 ** 3)
        model.save_pretrained(
            str(output_dir),
            max_shard_size=max_shard_size,
            safe_serialization=False  # Use .bin files instead of .safetensors
        )
        
        # Remove original file if it still exists
        if model_file.exists():
            print(f"Removing original {model_file.name}...")
            model_file.unlink()
        
        print("✓ Checkpoint sharded successfully")
        
    except Exception as e:
        print(f"Error during sharding: {e}")
        print("Model checkpoint may already be in the correct format.")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Shard a PyTorch model checkpoint")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing the checkpoint"
    )
    parser.add_argument(
        "--max_shard_size_gb",
        type=float,
        default=10.0,
        help="Maximum size per shard in GB (default: 10GB)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Sharding Model Checkpoint")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Max shard size: {args.max_shard_size_gb} GB")
    print("="*60)
    print("")
    
    shard_checkpoint(args.output_dir, args.max_shard_size_gb)
    
    print("")
    print("="*60)
    print("Sharding Complete!")
    print("="*60)

if __name__ == "__main__":
    main()

