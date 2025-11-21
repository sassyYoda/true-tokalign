#!/usr/bin/env python
# coding=utf-8
"""
Prepare GloVe training corpus from multiple sources.
Based on TokAlign paper: mix of CulturaX (40%), The Stack (30%), Proof-Pile-2 (30%)
Total: 1B tokens
"""

import json
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import random

def count_tokens(text, tokenizer=None):
    """
    Estimate token count using character-based approximation.
    Uses ~3.5 characters per token as a reasonable average for multilingual text.
    Much faster than actual tokenization for corpus preparation.
    """
    if tokenizer is None:
        # Character-based estimation (fast)
        return max(1, len(text) // 3.5)
    else:
        # Actual tokenization (slower, but accurate)
        return len(tokenizer.encode(text, add_special_tokens=False))

def estimate_tokens_from_chars(text):
    """
    Fast character-based token estimation.
    Uses ~3.5 chars/token average for multilingual text.
    """
    return max(1, int(len(text) / 3.5))

def sample_from_dataset(dataset, target_tokens, tokenizer, dataset_name, seed=42, is_streaming=False):
    """
    Sample from a dataset until we reach approximately target_tokens.
    Works with both streaming and non-streaming datasets.
    Returns list of texts.
    """
    random.seed(seed)
    texts = []
    total_tokens = 0
    
    if is_streaming:
        # For streaming datasets, iterate directly
        dataset_size = None
        dataset_iter = iter(dataset)
    else:
        # For non-streaming datasets, use indices
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        dataset_iter = None
    
    print(f"\nSampling from {dataset_name}...")
    pbar = tqdm(total=target_tokens, desc=f"Collecting tokens from {dataset_name}")
    
    idx = 0
    while total_tokens < target_tokens:
        try:
            if is_streaming:
                example = next(dataset_iter)
            else:
                if idx >= dataset_size:
                    break
                example = dataset[indices[idx]]
            
            # Extract text field - The Stack uses "content", other datasets use "text"
            if isinstance(example, dict):
                # Try common field names in order of preference
                text = example.get("text") or example.get("content") or example.get("code") or ""
            else:
                text = str(example)
            
            if text and len(text.strip()) > 0:
                # Use fast character-based estimation instead of tokenization
                tokens = estimate_tokens_from_chars(text)
                if tokens > 0:
                    texts.append(text)
                    total_tokens += tokens
                    pbar.update(tokens)
            
            idx += 1
        except StopIteration:
            # End of streaming dataset
            break
    
    pbar.close()
    print(f"Collected {total_tokens:,} tokens from {dataset_name} ({len(texts):,} examples)")
    return texts, total_tokens

def sample_from_multiple_streaming_datasets(datasets, target_tokens, tokenizer, dataset_name, seed=42):
    """
    Sample tokens from multiple streaming datasets proportionally.
    Uses round-robin approach to ensure diversity across languages.
    """
    import itertools
    random.seed(seed)
    texts = []
    total_tokens = 0
    
    # Create iterators for all datasets
    dataset_iters = [iter(ds) for ds in datasets]
    active_iters = list(range(len(dataset_iters)))
    
    print(f"\nSampling from {len(datasets)} {dataset_name} languages (streaming mode)...")
    pbar = tqdm(total=target_tokens, desc=f"Collecting tokens from {dataset_name}")
    
    # Round-robin through active datasets to ensure diversity
    iter_idx = 0
    while total_tokens < target_tokens and active_iters:
        # Get next iterator index (round-robin)
        idx = active_iters[iter_idx % len(active_iters)]
        dataset_iter = dataset_iters[idx]
        
        try:
            example = next(dataset_iter)
            
            # Extract text field - The Stack uses "content", other datasets use "text"
            if isinstance(example, dict):
                # Try common field names in order of preference
                text = example.get("text") or example.get("content") or example.get("code") or ""
            else:
                text = str(example)
            
            if text and len(text.strip()) > 0:
                # Use fast character-based estimation instead of tokenization
                tokens = estimate_tokens_from_chars(text)
                if tokens > 0:
                    texts.append(text)
                    total_tokens += tokens
                    pbar.update(tokens)
            
            iter_idx += 1
        except StopIteration:
            # This dataset is exhausted, remove it from active list
            active_iters.remove(idx)
            if not active_iters:
                break
            # Adjust iter_idx if needed
            if iter_idx >= len(active_iters):
                iter_idx = 0
    
    pbar.close()
    print(f"Collected {total_tokens:,} tokens from {dataset_name} ({len(texts):,} examples)")
    return texts, total_tokens

def load_culturax(cache_dir=None, max_samples=None):
    """Load CulturaX dataset."""
    try:
        # Try common CulturaX dataset names
        dataset = load_dataset("uonlp/CulturaX", cache_dir=cache_dir, streaming=False)
        if "train" in dataset:
            return dataset["train"]
        return dataset[list(dataset.keys())[0]]
    except Exception as e:
        print(f"Warning: Could not load CulturaX from uonlp/CulturaX: {e}")
        print("Please ensure CulturaX is available or update the dataset name.")
        raise

def load_the_stack(cache_dir=None, max_samples=None):
    """Load The Stack dataset."""
    try:
        # The Stack is large, we'll use a subset
        dataset = load_dataset("bigcode/the-stack", data_dir="data/python", cache_dir=cache_dir, streaming=False)
        if "train" in dataset:
            return dataset["train"]
        return dataset[list(dataset.keys())[0]]
    except Exception as e:
        print(f"Warning: Could not load The Stack from bigcode/the-stack: {e}")
        print("Trying alternative: bigcode/starcoderdata")
        try:
            dataset = load_dataset("bigcode/starcoderdata", cache_dir=cache_dir, streaming=False)
            if "train" in dataset:
                return dataset["train"]
            return dataset[list(dataset.keys())[0]]
        except Exception as e2:
            print(f"Could not load alternative: {e2}")
            raise

def load_proof_pile_2(cache_dir=None, max_samples=None):
    """Load Proof-Pile-2 dataset."""
    try:
        dataset = load_dataset("EleutherAI/proof-pile-2", cache_dir=cache_dir, streaming=False)
        if "train" in dataset:
            return dataset["train"]
        return dataset[list(dataset.keys())[0]]
    except Exception as e:
        print(f"Warning: Could not load Proof-Pile-2: {e}")
        print("Please ensure Proof-Pile-2 is available or update the dataset name.")
        raise

def main():
    parser = argparse.ArgumentParser(description="Prepare GloVe training corpus")
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=1_000_000_000,
        help="Total number of tokens to collect (default: 1B)"
    )
    parser.add_argument(
        "--culturax-ratio",
        type=float,
        default=0.4,
        help="Ratio of CulturaX tokens (default: 0.4)"
    )
    parser.add_argument(
        "--stack-ratio",
        type=float,
        default=0.3,
        help="Ratio of The Stack tokens (default: 0.3)"
    )
    parser.add_argument(
        "--proof-pile-ratio",
        type=float,
        default=0.3,
        help="Ratio of Proof-Pile-2 tokens (default: 0.3)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for datasets"
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="EleutherAI/pythia-1b",
        help="Tokenizer to use for counting tokens (default: EleutherAI/pythia-1b)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for datasets (useful for large datasets)"
    )
    parser.add_argument(
        "--culturax-dataset",
        type=str,
        default="uonlp/CulturaX",
        help="CulturaX dataset name on HuggingFace"
    )
    parser.add_argument(
        "--stack-dataset",
        type=str,
        default="bigcode/the-stack",
        help="The Stack dataset name on HuggingFace"
    )
    parser.add_argument(
        "--proof-pile-dataset",
        type=str,
        default="lehduong/proof-pile-2",
        help="Proof-Pile-2 dataset name on HuggingFace (default: lehduong/proof-pile-2, Parquet format)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.culturax_ratio + args.stack_ratio + args.proof_pile_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total_ratio}, not 1.0. Normalizing...")
        args.culturax_ratio /= total_ratio
        args.stack_ratio /= total_ratio
        args.proof_pile_ratio /= total_ratio
    
    # Calculate target tokens for each dataset
    culturax_tokens = int(args.total_tokens * args.culturax_ratio)
    stack_tokens = int(args.total_tokens * args.stack_ratio)
    proof_pile_tokens = args.total_tokens - culturax_tokens - stack_tokens
    
    print(f"Target token distribution:")
    print(f"  CulturaX: {culturax_tokens:,} tokens ({args.culturax_ratio*100:.1f}%)")
    print(f"  The Stack: {stack_tokens:,} tokens ({args.stack_ratio*100:.1f}%)")
    print(f"  Proof-Pile-2: {proof_pile_tokens:,} tokens ({args.proof_pile_ratio*100:.1f}%)")
    print(f"  Total: {args.total_tokens:,} tokens")
    
    # Load tokenizer for counting
    print(f"\nLoading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    
    # Load and sample from each dataset
    all_texts = []
    
    # CulturaX (40%)
    try:
        print("\n" + "="*60)
        print("Loading CulturaX dataset...")
        
        # Load multiple languages for multilingual corpus
        # Using the exact 30 languages from TokAlign paper (Nguyen et al., 2023)
        # High-Langs (10): en, ru, es, de, fr, zh, it, pt, pl, ja
        # Medium-Langs (10): cs, vi, fa, hu, el, ro, sv, uk, fi, ko
        # Low-Langs (10): he, sr, ta, sq, az, kk, ur, ka, hy, is
        culturax_languages = [
            # High-Langs
            'en', 'ru', 'es', 'de', 'fr', 'zh', 'it', 'pt', 'pl', 'ja',
            # Medium-Langs
            'cs', 'vi', 'fa', 'hu', 'el', 'ro', 'sv', 'uk', 'fi', 'ko',
            # Low-Langs
            'he', 'sr', 'ta', 'sq', 'az', 'kk', 'ur', 'ka', 'hy', 'is'
        ]
        
        # Use streaming mode to sample on-the-fly (much more efficient)
        print(f"Loading {len(culturax_languages)} languages from CulturaX (streaming mode)...")
        culturax_streaming = True
        
        culturax_datasets = []
        for lang in tqdm(culturax_languages, desc="Initializing CulturaX language streams"):
            try:
                lang_dataset = load_dataset(
                    args.culturax_dataset,
                    lang,
                    cache_dir=args.cache_dir,
                    streaming=True,
                    split="train"
                )
                culturax_datasets.append(lang_dataset)
            except Exception as e:
                print(f"Warning: Could not load CulturaX language '{lang}': {e}")
                continue
        
        if not culturax_datasets:
            raise Exception("No CulturaX languages could be loaded")
        
        # Sample tokens from all languages using round-robin (streaming, on-the-fly)
        culturax_texts, actual_tokens = sample_from_multiple_streaming_datasets(
            culturax_datasets, culturax_tokens, tokenizer, "CulturaX", seed=args.seed
        )
        all_texts.extend(culturax_texts)
    except Exception as e:
        print(f"Error loading CulturaX: {e}")
        print("Skipping CulturaX. Please check dataset availability.")
    
    # The Stack (30%)
    try:
        print("\n" + "="*60)
        print("Loading The Stack dataset (streaming mode)...")
        # Note: data_dir parameter doesn't work well with streaming mode
        # Load full dataset - filtering can be done post-load if needed
        stack_dataset = load_dataset(
            args.stack_dataset, 
            cache_dir=args.cache_dir, 
            streaming=True,
            split="train",
            trust_remote_code=True
        )
        print("Note: Using full The Stack dataset (not filtered to Python-only)")
        
        # Debug: Check first example to see field structure
        print("Checking dataset structure...")
        try:
            first_example = next(iter(stack_dataset))
            print(f"Available fields in first example: {list(first_example.keys()) if isinstance(first_example, dict) else 'Not a dict'}")
            # The Stack typically uses "content" field, not "text"
            if isinstance(first_example, dict):
                text_field = first_example.get("content") or first_example.get("text") or first_example.get("code")
                if text_field:
                    print(f"Found text field with {len(str(text_field))} characters")
                else:
                    print(f"Warning: No obvious text field found. Example keys: {list(first_example.keys())}")
        except Exception as debug_e:
            print(f"Could not inspect first example: {debug_e}")
        
        stack_texts, actual_tokens = sample_from_dataset(
            stack_dataset, stack_tokens, tokenizer, "The Stack", seed=args.seed + 1, is_streaming=True
        )
        all_texts.extend(stack_texts)
    except Exception as e:
        print(f"Error loading The Stack: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping The Stack. Please check dataset availability.")
    
    # Proof-Pile-2 (30%)
    try:
        print("\n" + "="*60)
        print("Loading Proof-Pile-2 dataset...")
        print(f"Using dataset: {args.proof_pile_dataset} (Parquet format)")
        # Try loading individual configs/subsets and combining
        # Parquet format should avoid zstd decompression issues
        configs = ["arxiv", "open-web-math", "algebraic-stack"]
        proof_pile_datasets = []
        is_streaming_list = []
        
        for config in configs:
            try:
                print(f"Loading Proof-Pile-2 config: {config} (trying streaming first)...")
                # Try streaming first
                try:
                    ds = load_dataset(
                        args.proof_pile_dataset, 
                        config,
                        cache_dir=args.cache_dir, 
                        streaming=True,
                        split="train",
                        trust_remote_code=True
                    )
                    # Test if streaming works by trying to get first example
                    test_iter = iter(ds)
                    try:
                        first_example = next(test_iter)  # This will fail if zstd error occurs
                        # Verify we got actual data
                        if isinstance(first_example, dict):
                            text = first_example.get("text") or first_example.get("content") or first_example.get("code") or ""
                            if not text or len(str(text).strip()) == 0:
                                raise ValueError(f"First example from {config} has no text content")
                        proof_pile_datasets.append(ds)
                        is_streaming_list.append(True)
                        print(f"Successfully loaded {config} (streaming mode)")
                    except StopIteration:
                        print(f"Warning: Config {config} streaming dataset is empty. Skipping.")
                    except Exception as test_error:
                        raise  # Re-raise zstd and other errors
                except Exception as stream_error:
                    if "zstd" in str(stream_error).lower() or "decompress" in str(stream_error).lower():
                        print(f"Streaming failed for {config} (zstd error), trying non-streaming...")
                        try:
                            ds = load_dataset(
                                args.proof_pile_dataset, 
                                config,
                                cache_dir=args.cache_dir, 
                                streaming=False,
                                split="train",
                                trust_remote_code=True
                            )
                            # Verify dataset has data
                            if isinstance(ds, dict):
                                ds = ds.get("train", ds.get(list(ds.keys())[0]))
                            
                            # Check if dataset is empty
                            if len(ds) == 0:
                                print(f"Warning: Config {config} loaded but has 0 examples. Skipping.")
                            else:
                                proof_pile_datasets.append(ds)
                                is_streaming_list.append(False)
                                print(f"Successfully loaded {config} (non-streaming mode, {len(ds):,} examples)")
                        except Exception as non_stream_error:
                            print(f"Non-streaming also failed for {config}: {non_stream_error}")
                            import traceback
                            traceback.print_exc()
                            print(f"Skipping config {config}, will continue with other configs")
                    else:
                        print(f"Streaming failed for {config} with non-zstd error: {stream_error}")
                        print(f"Skipping config {config}, will continue with other configs")
            except Exception as config_error:
                print(f"Warning: Could not load config {config}: {config_error}")
                print(f"Skipping config {config}, will continue with other configs")
        
        if not proof_pile_datasets:
            raise Exception("Could not load any Proof-Pile-2 configs")
        
        # Sample proportionally from all configs
        tokens_per_config = proof_pile_tokens // len(proof_pile_datasets)
        all_proof_pile_texts = []
        total_proof_pile_tokens = 0
        
        for i, ds in enumerate(proof_pile_datasets):
            texts, tokens = sample_from_dataset(
                ds, tokens_per_config, tokenizer, f"Proof-Pile-2-{configs[i]}", 
                seed=args.seed + 2 + i, is_streaming=is_streaming_list[i]
            )
            all_proof_pile_texts.extend(texts)
            total_proof_pile_tokens += tokens
        
        all_texts.extend(all_proof_pile_texts)
        print(f"\nCollected {total_proof_pile_tokens:,} tokens from Proof-Pile-2 ({len(all_proof_pile_texts):,} examples)")
    except Exception as e:
        print(f"Error loading Proof-Pile-2: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping Proof-Pile-2. Please check dataset availability.")
    
    # Shuffle all texts
    print("\n" + "="*60)
    print("Shuffling combined corpus...")
    random.seed(args.seed)
    random.shuffle(all_texts)
    
    # Write to JSONL file
    print(f"\nWriting {len(all_texts):,} examples to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        for text in tqdm(all_texts, desc="Writing JSONL"):
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
    
    # Count total tokens in output (using character-based estimation for speed)
    print("\nEstimating total tokens in output file (character-based)...")
    total_output_tokens = 0
    with open(args.output_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Estimating tokens"):
            data = json.loads(line)
            total_output_tokens += estimate_tokens_from_chars(data["text"])
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total examples: {len(all_texts):,}")
    print(f"  Total tokens: {total_output_tokens:,}")
    print(f"  Output file: {args.output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

