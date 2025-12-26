import json
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import argparse
from bert_score import score as bert_score
from tqdm import tqdm
import os
import torch

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
    model_name="microsoft/deberta-base-mnli",
    batch_size=8,
    device="cuda",
    max_examples=None,
    **kwargs
):
    """
    Evaluate using BERTScore:
    1. Map target token IDs to source token IDs using alignment matrix
    2. De-tokenize mapped source token IDs → recovered text C'
    3. De-tokenize original source token IDs → original text C
    4. Compare C' and C using BERTScore
    """
    source_tok = AutoTokenizer.from_pretrained(source_tokenizer_path)
    
    with open(trans_dict_path, "r") as f:
        trans = json.load(f)

    eval_data = read_tsv(eval_file_path)
    td = trans
    
    # Limit number of examples if specified (for testing/debugging)
    max_examples = kwargs.get('max_examples', None)
    if max_examples and max_examples > 0:
        eval_data = eval_data[:max_examples]
        print(f"Limiting evaluation to {max_examples} examples for testing/debugging")

    # Collect recovered texts (C') and original texts (C)
    recovered_texts = []  # C': de-tokenized from mapped source token IDs
    original_texts = []   # C: de-tokenized from original source token IDs
    
    print("De-tokenizing texts for BERTScore evaluation...")
    for s in tqdm(eval_data, desc="Processing examples"):
        src_token_ids, tgt_token_ids = s[0], s[1]
        
        # Map target token IDs to source token IDs using alignment matrix
        mapped_src_token_ids = [int(td[tid]) for tid in tgt_token_ids]
        
        # De-tokenize mapped source token IDs → recovered text C'
        recovered_text = source_tok.decode(mapped_src_token_ids, skip_special_tokens=True)
        recovered_texts.append(recovered_text)
        
        # De-tokenize original source token IDs → original text C
        original_src_token_ids = [int(sid) for sid in src_token_ids]
        original_text = source_tok.decode(original_src_token_ids, skip_special_tokens=True)
        original_texts.append(original_text)
    
    print(f"Computing BERTScore for {len(recovered_texts)} examples...")
    
    # Clear GPU cache before starting
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Process in chunks to avoid memory issues
    # Use smaller chunks for BERTScore to manage memory better
    chunk_size = min(batch_size * 4, 100)  # Process up to 100 examples at a time
    all_precision = []
    all_recall = []
    all_f1 = []
    
    num_chunks = (len(recovered_texts) + chunk_size - 1) // chunk_size
    print(f"Processing {len(recovered_texts)} examples in {num_chunks} chunks of ~{chunk_size} examples each...")
    
    for i in tqdm(range(0, len(recovered_texts), chunk_size), desc="BERTScore chunks"):
        chunk_recovered = recovered_texts[i:i+chunk_size]
        chunk_original = original_texts[i:i+chunk_size]
        
        try:
            # Compute BERTScore for this chunk
            P, R, F1 = bert_score(
                chunk_recovered,
                chunk_original,
                model_type=model_name,
                batch_size=batch_size,
                device=device,
                verbose=False  # Reduce verbosity for chunks
            )
            
            # Collect scores
            all_precision.extend(P.cpu().tolist())
            all_recall.extend(R.cpu().tolist())
            all_f1.extend(F1.cpu().tolist())
            
            # Clear GPU cache after each chunk
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\nWarning: Failed to process chunk {i//chunk_size + 1}/{num_chunks}: {e}")
            print(f"Skipping {len(chunk_recovered)} examples in this chunk...")
            # Add zeros for failed chunk
            all_precision.extend([0.0] * len(chunk_recovered))
            all_recall.extend([0.0] * len(chunk_recovered))
            all_f1.extend([0.0] * len(chunk_recovered))
            continue
    
    # Calculate averages
    if len(all_f1) > 0:
        avg_precision = sum(all_precision) / len(all_precision)
        avg_recall = sum(all_recall) / len(all_recall)
        avg_f1 = sum(all_f1) / len(all_f1)
        
        print(f"BERTScore Precision: {avg_precision:.6f}")
        print(f"BERTScore Recall: {avg_recall:.6f}")
        print(f"BERTScore F1: {avg_f1:.6f}")
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "num_examples": len(recovered_texts),
            "precision_scores": all_precision,
            "recall_scores": all_recall,
            "f1_scores": all_f1
        }
    else:
        raise Exception("Failed to compute BERTScore for any examples")

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
    parser.add_argument("-b", "--bert-score-model-path", type=str, default="microsoft/deberta-base-mnli",
                       help="BERTScore model name (default: deberta-base-mnli, lighter than deberta-xlarge)")
    parser.add_argument("-w", "--bleu-weights", type=str, default="1,0,0,0")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save evaluation results JSON (default: same as eval_file_path parent)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for BERTScore (default: 8, reduce if OOM)")
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
