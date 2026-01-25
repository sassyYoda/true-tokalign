import json
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import argparse
from bert_score import BERTScorer
from tqdm import tqdm
import os
import torch
import numpy as np

def read_tsv(file_path):
    res = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            res.append([d.split(" ") for d in line.strip().split("\t")])
    return res

# BLEU-1
def eval_trans_matrix(
    trans_dict_path="./log/pythia2qwen2-7b/glove-MX1M-iter15-d300.json", 
    eval_file_path="./data/pretrain-dataset/pythia-2-qwen2-7b-MX1K-eval",
    bleu_weights=(1, 0, 0, 0),
):
    with open(trans_dict_path, "r") as f:
        trans = json.load(f)

    eval_data = read_tsv(eval_file_path)

    tgt_len = len(list(trans.keys()))

    td = trans

    tgt_len = len(list(trans.keys()))

    total_b = 0
    total_b1 = 0  # BLEU-1 only
    total_b2 = 0  # BLEU-2 only
    total_b3 = 0  # BLEU-3 only
    total_b4 = 0  # BLEU-4 only
    # for s in tqdm(eval_data):
    for s in eval_data:
        # src: source token id, e.g., pythia ids, tgt: target token id, e.g., qwen2-7b ids
        src, tgt = s[0], s[1]

        # using td dict by maping target ids to source ids
        pred = [str(td[tid]) for tid in tgt]
        
        # Compute with specified weights (weighted average)
        total_b += sentence_bleu([src], pred, bleu_weights)
        
        # Compute individual BLEU scores
        total_b1 += sentence_bleu([src], pred, (1, 0, 0, 0))  # BLEU-1
        total_b2 += sentence_bleu([src], pred, (0, 1, 0, 0))  # BLEU-2
        total_b3 += sentence_bleu([src], pred, (0, 0, 1, 0))  # BLEU-3
        total_b4 += sentence_bleu([src], pred, (0, 0, 0, 1))  # BLEU-4

        # using td dict by maping source ids to target ids
        # pred = [str(td[sid]) for sid in src]
        # total_b += sentence_bleu([tgt], pred, bleu_weights)

    num_examples = len(eval_data)
    
    # Calculate averages
    avg_bleu = total_b / num_examples
    avg_bleu1 = total_b1 / num_examples
    avg_bleu2 = total_b2 / num_examples
    avg_bleu3 = total_b3 / num_examples
    avg_bleu4 = total_b4 / num_examples
    
    # Print individual BLEU scores
    print(f"BLEU-1: {avg_bleu1:.6f}")
    print(f"BLEU-2: {avg_bleu2:.6f}")
    print(f"BLEU-3: {avg_bleu3:.6f}")
    print(f"BLEU-4: {avg_bleu4:.6f}")
    print(f"Average BLEU (weights {bleu_weights}): {avg_bleu:.6f}")

    return {
        "bleu": avg_bleu,
        "bleu1": avg_bleu1,
        "bleu2": avg_bleu2,
        "bleu3": avg_bleu3,
        "bleu4": avg_bleu4,
        "num_examples": num_examples
    }

# BERT-Score: De-tokenize target token IDs using alignment matrix, then compare with original text
def eval_bert_score(
    trans_dict_path="./log/pythia2qwen2-7b/glove-MX1M-iter15-d300.json",
    eval_file_path="./data/pretrain-dataset/pythia-2-qwen2-7b-MX1K-eval",
    source_tokenizer_path="EleutherAI/pythia-1b",
    target_tokenizer_path=None,
    model_name="roberta-base",
    batch_size=32,
    device="cuda",
    max_examples=None,
    **kwargs
):
    """
    Optimized BERTScore evaluation using BERTScorer class (load model once, reuse):
    1. Map target token IDs to source token IDs using alignment matrix
    2. De-tokenize mapped source token IDs → recovered text C'
    3. De-tokenize original source token IDs → original text C
    4. Compare C' and C using BERTScore with efficient batching
    """
    source_tok = AutoTokenizer.from_pretrained(source_tokenizer_path)
    
    with open(trans_dict_path, "r") as f:
        trans = json.load(f)

    eval_data = read_tsv(eval_file_path)
    td = trans
    
    # Limit number of examples if specified (for testing/debugging)
    if max_examples and max_examples > 0:
        eval_data = eval_data[:max_examples]
        print(f"Limiting evaluation to {max_examples} examples for testing/debugging")

    # Vectorized de-tokenization for better performance
    print("De-tokenizing texts for BERTScore evaluation...")
    recovered_texts = []  # C': de-tokenized from mapped source token IDs
    original_texts = []   # C: de-tokenized from original source token IDs
    
    # Batch decode for efficiency
    batch_decode_size = 100
    for batch_start in tqdm(range(0, len(eval_data), batch_decode_size), desc="De-tokenizing"):
        batch_end = min(batch_start + batch_decode_size, len(eval_data))
        batch_data = eval_data[batch_start:batch_end]
        
        # Process batch
        batch_recovered_ids = []
        batch_original_ids = []
        
        for s in batch_data:
            src_token_ids, tgt_token_ids = s[0], s[1]
            # Map target token IDs to source token IDs using alignment matrix
            mapped_src_token_ids = [int(td[tid]) for tid in tgt_token_ids]
            batch_recovered_ids.append(mapped_src_token_ids)
            batch_original_ids.append([int(sid) for sid in src_token_ids])
        
        # Batch decode (more efficient than individual decodes)
        batch_recovered = source_tok.batch_decode(batch_recovered_ids, skip_special_tokens=True)
        batch_original = source_tok.batch_decode(batch_original_ids, skip_special_tokens=True)
        
        recovered_texts.extend(batch_recovered)
        original_texts.extend(batch_original)
    
    print(f"Computing BERTScore for {len(recovered_texts)} examples...")
    
    # Clear GPU cache before starting
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before BERTScore: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    
    # Use BERTScorer class - load model ONCE and reuse (much more efficient!)
    # Try CPU first if GPU memory is limited, or use a smaller model
    print(f"Loading BERTScore model: {model_name} on {device}...")
    
    # Check GPU memory availability
    if device == "cuda" and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory available: {gpu_memory_gb:.2f} GB")
        
        # If GPU has less than 20GB, use CPU or smaller model
        if gpu_memory_gb < 20:
            print(f"GPU memory limited ({gpu_memory_gb:.2f} GB). Using CPU for better reliability...")
            device = "cpu"
    
    scorer = None
    try:
        # Start with smaller batch size for initial load
        initial_batch_size = min(batch_size, 8)
        scorer = BERTScorer(
            model_type=model_name,
            lang='en',
            device=device,
            batch_size=initial_batch_size,
            rescale_with_baseline=True,
            nthreads=4 if device == "cpu" else 1  # Use multiple threads on CPU
        )
        print(f"Model loaded successfully on {device}!")
    except Exception as e:
        print(f"Failed to load model on {device}: {e}")
        if device == "cuda":
            print("Falling back to CPU...")
            device = "cpu"
            try:
                scorer = BERTScorer(
                    model_type=model_name,
                    lang='en',
                    device="cpu",
                    batch_size=min(batch_size, 16),
                    rescale_with_baseline=True,
                    nthreads=4
                )
                print("Model loaded successfully on CPU!")
            except Exception as e2:
                print(f"Failed to load on CPU too: {e2}")
                # Try with an even smaller model
                if model_name != "distilbert-base-uncased":
                    print("Trying with distilbert-base-uncased (smallest model)...")
                    model_name = "distilbert-base-uncased"
                    scorer = BERTScorer(
                        model_type=model_name,
                        lang='en',
                        device="cpu",
                        batch_size=16,
                        rescale_with_baseline=True,
                        nthreads=4
                    )
                else:
                    raise
        else:
            raise
    
    # Process in batches - start conservative and adapt
    # Use smaller initial batch size, especially for GPU
    if device == "cuda":
        effective_batch_size = min(batch_size, 8)  # Start smaller on GPU
    else:
        effective_batch_size = min(batch_size, 16)  # Can use larger batches on CPU
    
    all_precision = []
    all_recall = []
    all_f1 = []
    
    num_batches = (len(recovered_texts) + effective_batch_size - 1) // effective_batch_size
    print(f"Processing {len(recovered_texts)} examples in {num_batches} batches of ~{effective_batch_size} examples each...")
    
    successful_batches = 0
    failed_batches = 0
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    for i in tqdm(range(0, len(recovered_texts), effective_batch_size), desc="BERTScore batches"):
        batch_recovered = recovered_texts[i:i+effective_batch_size]
        batch_original = original_texts[i:i+effective_batch_size]
        
        # Clear GPU cache before each batch
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        batch_success = False
        retry_batch_size = effective_batch_size
        
        # Retry with progressively smaller batches if needed
        while retry_batch_size >= 1 and not batch_success:
            try:
                # Process subset if retrying with smaller size
                if retry_batch_size < effective_batch_size:
                    current_recovered = batch_recovered[:retry_batch_size]
                    current_original = batch_original[:retry_batch_size]
                else:
                    current_recovered = batch_recovered
                    current_original = batch_original
                
                # Use scorer.score() - much more efficient than calling bert_score() repeatedly
                P, R, F1 = scorer.score(
                    current_recovered,
                    current_original,
                    verbose=False
                )
                
                # Convert to numpy arrays for efficient processing
                P_np = P.cpu().numpy() if isinstance(P, torch.Tensor) else np.array(P)
                R_np = R.cpu().numpy() if isinstance(R, torch.Tensor) else np.array(R)
                F1_np = F1.cpu().numpy() if isinstance(F1, torch.Tensor) else np.array(F1)
                
                # Extend lists efficiently
                all_precision.extend(P_np.tolist())
                all_recall.extend(R_np.tolist())
                all_f1.extend(F1_np.tolist())
                
                # Fill remaining with zeros if we processed fewer than expected
                if retry_batch_size < len(batch_recovered):
                    remaining = len(batch_recovered) - retry_batch_size
                    all_precision.extend([0.0] * remaining)
                    all_recall.extend([0.0] * remaining)
                    all_f1.extend([0.0] * remaining)
                
                # Delete tensors explicitly
                del P, R, F1, P_np, R_np, F1_np
                
                successful_batches += 1
                batch_success = True
                consecutive_failures = 0
                
                # Clear GPU cache after each batch
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError as e:
                consecutive_failures += 1
                if retry_batch_size > 1:
                    retry_batch_size = max(1, retry_batch_size // 2)
                    print(f"\nCUDA OOM at batch {i//effective_batch_size + 1}/{num_batches}, reducing to batch_size={retry_batch_size}...")
                    torch.cuda.empty_cache()
                else:
                    # Even batch_size=1 failed, skip this batch
                    print(f"\nCUDA OOM even with batch_size=1, skipping batch {i//effective_batch_size + 1}/{num_batches}...")
                    all_precision.extend([0.0] * len(batch_recovered))
                    all_recall.extend([0.0] * len(batch_recovered))
                    all_f1.extend([0.0] * len(batch_recovered))
                    failed_batches += 1
                    batch_success = True  # Mark as "handled" to break retry loop
                    
                    # If too many consecutive failures, switch to CPU
                    if consecutive_failures >= max_consecutive_failures and device == "cuda":
                        print(f"\nToo many consecutive failures ({consecutive_failures}). Switching to CPU...")
                        device = "cpu"
                        # Reload scorer on CPU
                        del scorer
                        torch.cuda.empty_cache()
                        scorer = BERTScorer(
                            model_type=model_name,
                            lang='en',
                            device="cpu",
                            batch_size=16,
                            rescale_with_baseline=True,
                            nthreads=4
                        )
                        effective_batch_size = 16
                        consecutive_failures = 0
                        print("Switched to CPU successfully!")
                    
            except Exception as e:
                failed_batches += 1
                consecutive_failures += 1
                print(f"\nWarning: Failed to process batch {i//effective_batch_size + 1}/{num_batches}: {e}")
                all_precision.extend([0.0] * len(batch_recovered))
                all_recall.extend([0.0] * len(batch_recovered))
                all_f1.extend([0.0] * len(batch_recovered))
                batch_success = True  # Mark as handled
                
                # Clear cache on any error
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Clean up scorer
    del scorer
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\nBERTScore processing complete: {successful_batches} successful batches, {failed_batches} failed batches")
    
    # Calculate averages using numpy for efficiency (only from successful examples)
    all_f1_np = np.array(all_f1)
    successful_mask = all_f1_np > 0.0
    successful_scores = all_f1_np[successful_mask]
    
    if len(successful_scores) > 0:
        # Use numpy for efficient calculation
        all_precision_np = np.array(all_precision)
        all_recall_np = np.array(all_recall)
        
        avg_precision = float(np.mean(all_precision_np[successful_mask]))
        avg_recall = float(np.mean(all_recall_np[successful_mask]))
        avg_f1 = float(np.mean(successful_scores))
        
        print(f"\nBERTScore results (from {len(successful_scores)}/{len(recovered_texts)} successful examples):")
        print(f"BERTScore Precision: {avg_precision:.6f}")
        print(f"BERTScore Recall: {avg_recall:.6f}")
        print(f"BERTScore F1: {avg_f1:.6f}")
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "num_examples": len(recovered_texts),
            "num_successful": int(len(successful_scores)),
            "precision_scores": all_precision,
            "recall_scores": all_recall,
            "f1_scores": all_f1
        }
    else:
        raise Exception(f"Failed to compute BERTScore for any examples. All {len(recovered_texts)} examples failed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluate-method", type=str, default="both", 
                       choices=["bleu", "bert-score", "bertscore", "both", "all"],
                       help="Evaluation method: 'bleu', 'bert-score', or 'both'")
    parser.add_argument("-m", "--one2one-matrix-path", type=str, default="./data/pythia2qwen2-7b/glove.json")
    parser.add_argument("-f", "--eval-file-path", type=str, default="./data/pretrain-dataset/pythia-2-qwen2-7b-MX1K-eval")
    parser.add_argument("-t", "--tokenizer-path", type=str, default="EleutherAI/pythia-1b",
                       help="Source tokenizer path (for de-tokenization)")
    parser.add_argument("--target-tokenizer-path", type=str, default=None,
                       help="Target tokenizer path (optional, for future use)")
    parser.add_argument("-b", "--bert-score-model-path", type=str, default="roberta-base",
                       help="BERTScore model name (default: roberta-base, much lighter. Options: roberta-base, distilbert-base-uncased, microsoft/deberta-base-mnli)")
    parser.add_argument("-w", "--bleu-weights", type=str, default="1,0,0,0")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save evaluation results JSON (default: same as eval_file_path parent)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for BERTScore (default: 32, model loaded once so larger batches are efficient)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for BERTScore (cuda/cpu)")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to evaluate (for testing/debugging)")

    args = parser.parse_args()

    results = {}
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.eval_file_path)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    eval_method = args.evaluate_method.lower()
    
    # Run BLEU evaluation
    if eval_method in ["bleu", "both", "all"]:
        print("\n" + "="*70)
        print("Running BLEU Evaluation")
        print("="*70)
        weights = tuple([float(i) for i in args.bleu_weights.split(",")])
        assert len(weights) == 4, "There are only 4 BLEU weights (BLEU-1 to 4)"
        bleu_results = eval_trans_matrix(
            trans_dict_path = args.one2one_matrix_path,
            eval_file_path = args.eval_file_path,
            bleu_weights = weights
        )
        results["bleu"] = bleu_results
    
    # Run BERTScore evaluation
    if eval_method in ["bert-score", "bertscore", "both", "all"]:
        print("\n" + "="*70)
        print("Running BERTScore Evaluation")
        print("="*70)
        try:
            bertscore_results = eval_bert_score(
                trans_dict_path = args.one2one_matrix_path,
                eval_file_path = args.eval_file_path,
                source_tokenizer_path = args.tokenizer_path,
                target_tokenizer_path = args.target_tokenizer_path,
                model_name = args.bert_score_model_path,
                batch_size = args.batch_size,
                device = args.device,
                max_examples = args.max_examples
            )
            results["bertscore"] = bertscore_results
        except Exception as e:
            print(f"\nERROR: BERTScore evaluation failed: {e}")
            print("This may be due to CUDA out of memory. Try:")
            print("  - Using a smaller model (e.g., 'microsoft/deberta-base-mnli' or 'roberta-base')")
            print("  - Reducing batch size (--batch-size 8 or --batch-size 4)")
            print("  - Using CPU instead (--device cpu)")
            print("\nContinuing with BLEU results only...")
            if "error" not in results:
                results["error"] = {}
            results["error"]["bertscore"] = str(e)
    
    # Save results to JSON (only if we have at least one successful evaluation)
    if results:
        output_file = os.path.join(output_dir, "evaluation_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to: {output_file}")
        
        # Print summary
        if "bleu" in results:
            print(f"\nSummary - BLEU Average: {results['bleu']['bleu']:.6f}")
        if "bertscore" in results:
            print(f"Summary - BERTScore F1: {results['bertscore']['f1']:.6f}")
    else:
        print("\nERROR: No evaluation results to save.")
