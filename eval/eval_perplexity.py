"""
Perplexity evaluation script for Spanish monolingual corpus.
Evaluates the trained model's perplexity on Spanish text, treating it as a monolingual Spanish model.
"""

import argparse
import json
import os
from typing import List, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype: Optional[torch.dtype] = None
):
    """Load the trained model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print(f"Model loaded on {device}")
    
    return model, tokenizer


def load_spanish_corpus(
    corpus_name: str = "mc4",
    split: str = "validation",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    text_field: str = "text"
) -> List[str]:
    """Load a Spanish monolingual corpus from HuggingFace.
    
    Args:
        corpus_name: Name of the corpus. Options:
            - "mc4": Spanish subset of mC4 (Common Crawl)
            - "wikipedia": Spanish Wikipedia
            - "oscar": Spanish subset of OSCAR
        split: Dataset split to use
        max_samples: Maximum number of samples to load
        cache_dir: Cache directory for HuggingFace datasets
        text_field: Field name containing the text
    
    Returns:
        List of Spanish text strings
    """
    print(f"\nLoading Spanish corpus: {corpus_name}...")
    
    try:
        if corpus_name == "mc4":
            # Load Spanish subset of mC4
            print("Loading mC4 Spanish subset from HuggingFace...")
            dataset = load_dataset(
                "allenai/c4",
                "es",  # Spanish language code
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            text_field = "text"
            
        elif corpus_name == "wikipedia":
            # Load Spanish Wikipedia
            print("Loading Spanish Wikipedia from HuggingFace...")
            dataset = load_dataset(
                "wikipedia",
                "20220301.es",  # Spanish Wikipedia dump
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            text_field = "text"
            
        elif corpus_name == "oscar":
            # Load Spanish subset of OSCAR
            print("Loading OSCAR Spanish subset from HuggingFace...")
            dataset = load_dataset(
                "oscar",
                "unshuffled_deduplicated_es",  # Spanish subset
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            text_field = "text"
            
        else:
            raise ValueError(f"Unknown corpus name: {corpus_name}. Choose from: mc4, wikipedia, oscar")
        
        print(f"Loaded {len(dataset)} examples from {corpus_name} ({split} split)")
        
        # Extract texts
        texts = []
        for i in tqdm(range(len(dataset)), desc="Extracting texts"):
            example = dataset[i]
            text = example.get(text_field, "")
            if text and isinstance(text, str) and len(text.strip()) > 0:
                texts.append(text.strip())
            
            if max_samples and len(texts) >= max_samples:
                break
        
        print(f"Extracted {len(texts)} valid Spanish texts")
        return texts
        
    except Exception as e:
        print(f"Error loading corpus: {e}")
        print("\nTrying alternative loading method...")
        
        # Try loading without config name
        try:
            if corpus_name == "mc4":
                # Try alternative mC4 loading
                dataset = load_dataset(
                    "allenai/c4",
                    split=split,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                # Filter for Spanish if possible
                if hasattr(dataset, 'filter'):
                    # Try to filter by language if available
                    pass
            else:
                raise
        except Exception as e2:
            print(f"Failed to load corpus: {e2}")
            raise Exception(f"Could not load {corpus_name} corpus: {e2}")


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
    stride: Optional[int] = None
) -> dict:
    """Compute perplexity on a list of texts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of texts to evaluate
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run on
        stride: Stride for sliding window (if None, uses max_length)
    
    Returns:
        Dictionary with perplexity metrics
    """
    print(f"\nComputing perplexity on Spanish texts...")
    print(f"Batch size: {batch_size}, Max length: {max_length}")
    
    if stride is None:
        stride = max_length
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_valid_samples = 0
    
    # Process in batches
    for batch_idx in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_losses = []
        batch_token_counts = []
        
        for text in batch_texts:
            try:
                # Tokenize text
                encodings = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    max_length=None
                )
                
                input_ids = encodings["input_ids"].to(device)
                
                # Skip if empty
                if input_ids.shape[1] < 2:
                    continue
                
                # Use sliding window if text is longer than max_length
                seq_len = input_ids.shape[1]
                if seq_len <= max_length:
                    # Process entire sequence
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, labels=input_ids)
                        # outputs.loss is already averaged over all tokens
                        # We need to multiply by number of tokens to get total loss
                        num_tokens = seq_len - 1  # Exclude first token (no prediction for it)
                        if num_tokens > 0:
                            total_loss_val = outputs.loss.item() * num_tokens
                            batch_losses.append(total_loss_val)
                            batch_token_counts.append(num_tokens)
                            num_valid_samples += 1
                else:
                    # Use sliding window approach for long sequences
                    # Process overlapping windows
                    window_losses = []
                    window_tokens = []
                    
                    for start_idx in range(0, seq_len - max_length + 1, stride):
                        end_idx = min(start_idx + max_length, seq_len)
                        window_ids = input_ids[:, start_idx:end_idx]
                        
                        if window_ids.shape[1] < 2:
                            continue
                        
                        with torch.no_grad():
                            outputs = model(input_ids=window_ids, labels=window_ids)
                            # outputs.loss is averaged, multiply by token count
                            num_tokens = window_ids.shape[1] - 1
                            if num_tokens > 0:
                                total_loss_val = outputs.loss.item() * num_tokens
                                window_losses.append(total_loss_val)
                                window_tokens.append(num_tokens)
                    
                    if window_losses:
                        # Sum over windows (not average, since we want total loss)
                        total_window_loss = sum(window_losses)
                        total_window_tokens = sum(window_tokens)
                        if total_window_tokens > 0:
                            batch_losses.append(total_window_loss)
                            batch_token_counts.append(total_window_tokens)
                            num_valid_samples += 1
                        
            except Exception as e:
                print(f"Error computing perplexity for sample {batch_idx}: {e}")
                continue
        
        # Accumulate losses
        if batch_losses:
            for loss, num_tokens in zip(batch_losses, batch_token_counts):
                total_loss += loss
                total_tokens += num_tokens
    
    # Compute average loss and perplexity
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')
    
    results = {
        "corpus": "Spanish monolingual",
        "num_samples": num_valid_samples,
        "total_tokens": total_tokens,
        "average_loss": avg_loss,
        "perplexity": perplexity,
        "batch_size": batch_size,
        "max_length": max_length
    }
    
    print(f"\n{'='*60}")
    print(f"Perplexity Results:")
    print(f"{'='*60}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Number of samples: {num_valid_samples}")
    print(f"Total tokens: {total_tokens}")
    print(f"{'='*60}")
    
    return results


def save_results(results: dict, output_dir: str):
    """Save perplexity results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "perplexity_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity on Spanish monolingual corpus"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--corpus_name",
        type=str,
        default="mc4",
        choices=["mc4", "wikipedia", "oscar"],
        help="Spanish corpus to use (default: mc4)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use (default: validation)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./log/perplexity_eval",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for perplexity computation (default: 32 for H100)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for sliding window (default: max_length)"
    )
    
    args = parser.parse_args()
    
    # Load Spanish corpus
    texts = load_spanish_corpus(
        corpus_name=args.corpus_name,
        split=args.split,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir
    )
    
    if not texts:
        raise ValueError("No texts loaded from corpus")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    # Compute perplexity
    results = compute_perplexity(
        model, tokenizer,
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        stride=args.stride
    )
    
    # Add metadata
    results["model_path"] = args.model_path
    results["corpus_name"] = args.corpus_name
    results["split"] = args.split
    
    # Save results
    save_results(results, args.output_dir)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

