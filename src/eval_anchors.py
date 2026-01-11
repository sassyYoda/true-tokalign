import json
import os
import argparse
import glob
from tqdm import tqdm

# Import evaluation functions from eval_matrix.py
from eval_matrix import eval_trans_matrix, eval_bert_score

def extract_seed_from_filename(filename):
    """Extract seed number from filename like 'alignment_matrix_seed_5.json'"""
    basename = os.path.basename(filename)
    if 'seed_' in basename:
        try:
            seed_str = basename.split('seed_')[1].split('.json')[0]
            return int(seed_str)
        except (ValueError, IndexError):
            return None
    return None

def find_alignment_matrices(matrix_dir):
    """Find all alignment matrix JSON files in the directory."""
    pattern = os.path.join(matrix_dir, "alignment_matrix_seed_*.json")
    matrix_files = glob.glob(pattern)
    matrix_files.sort()  # Sort for consistent ordering
    return matrix_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--matrix-dir", type=str, default="./alignment_matrix_seed_ablations",
                       help="Directory containing alignment matrix seed ablation files")
    parser.add_argument("-e", "--evaluate-method", type=str, default="both", 
                       choices=["bleu", "bert-score", "bertscore", "both", "all"],
                       help="Evaluation method: 'bleu', 'bert-score', or 'both'")
    parser.add_argument("-f", "--eval-file-path", type=str, default="./data/pretrain-dataset/pythia-2-qwen2-7b-MX1K-eval")
    parser.add_argument("-t", "--tokenizer-path", type=str, default="EleutherAI/pythia-1b",
                       help="Source tokenizer path (for de-tokenization)")
    parser.add_argument("--target-tokenizer-path", type=str, default=None,
                       help="Target tokenizer path (optional, for future use)")
    parser.add_argument("-b", "--bert-score-model-path", type=str, default="roberta-base",
                       help="BERTScore model name (default: roberta-base)")
    parser.add_argument("-w", "--bleu-weights", type=str, default="1,0,0,0")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for BERTScore")
    parser.add_argument("--device", type=str, default="cuda", help="Device for BERTScore (cuda/cpu)")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to evaluate")
    parser.add_argument("-o", "--output-file", type=str, default=None,
                       help="Output JSON file name (default: anchor_seed_evaluation_results.json in matrix-dir)")

    args = parser.parse_args()

    # Find all alignment matrices
    matrix_files = find_alignment_matrices(args.matrix_dir)
    
    if not matrix_files:
        print(f"ERROR: No alignment matrix files found in {args.matrix_dir}")
        print(f"Expected pattern: alignment_matrix_seed_*.json")
        exit(1)
    
    print(f"Found {len(matrix_files)} alignment matrices to evaluate")
    print(f"Matrix directory: {args.matrix_dir}")
    print(f"Evaluation method: {args.evaluate_method}")
    print(f"Evaluation data: {args.eval_file_path}")
    print("="*70)

    # Parse BLEU weights
    weights = tuple([float(i) for i in args.bleu_weights.split(",")])
    assert len(weights) == 4, "There are only 4 BLEU weights (BLEU-1 to 4)"

    eval_method = args.evaluate_method.lower()
    
    # Store all results
    all_results = {
        "evaluation_settings": {
            "eval_method": args.evaluate_method,
            "eval_file_path": args.eval_file_path,
            "tokenizer_path": args.tokenizer_path,
            "bleu_weights": args.bleu_weights,
            "bert_score_model": args.bert_score_model_path,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_examples": args.max_examples
        },
        "results_by_seed": {}
    }

    # Evaluate each matrix
    for matrix_file in tqdm(matrix_files, desc="Evaluating matrices"):
        seed = extract_seed_from_filename(matrix_file)
        if seed is None:
            print(f"Warning: Could not extract seed from {matrix_file}, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating seed {seed}")
        print(f"Matrix: {os.path.basename(matrix_file)}")
        print(f"{'='*70}")

        seed_results = {
            "matrix_file": os.path.basename(matrix_file),
            "seed": seed
        }

        # Run BLEU evaluation
        if eval_method in ["bleu", "both", "all"]:
            try:
                print(f"\nRunning BLEU evaluation for seed {seed}...")
                bleu_results = eval_trans_matrix(
                    trans_dict_path=matrix_file,
                    eval_file_path=args.eval_file_path,
                    bleu_weights=weights
                )
                seed_results["bleu"] = bleu_results
            except Exception as e:
                print(f"ERROR: BLEU evaluation failed for seed {seed}: {e}")
                seed_results["bleu_error"] = str(e)

        # Run BERTScore evaluation
        if eval_method in ["bert-score", "bertscore", "both", "all"]:
            try:
                print(f"\nRunning BERTScore evaluation for seed {seed}...")
                bertscore_results = eval_bert_score(
                    trans_dict_path=matrix_file,
                    eval_file_path=args.eval_file_path,
                    source_tokenizer_path=args.tokenizer_path,
                    target_tokenizer_path=args.target_tokenizer_path,
                    model_name=args.bert_score_model_path,
                    batch_size=args.batch_size,
                    device=args.device,
                    max_examples=args.max_examples
                )
                seed_results["bertscore"] = bertscore_results
            except Exception as e:
                print(f"ERROR: BERTScore evaluation failed for seed {seed}: {e}")
                print("Continuing with other seeds...")
                seed_results["bertscore_error"] = str(e)

        all_results["results_by_seed"][str(seed)] = seed_results

    # Calculate summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")

    summary = {}
    
    if eval_method in ["bleu", "both", "all"]:
        bleu_scores = []
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []
        
        for seed_str, seed_data in all_results["results_by_seed"].items():
            if "bleu" in seed_data and "bleu_error" not in seed_data:
                bleu_scores.append(seed_data["bleu"]["bleu"])
                bleu1_scores.append(seed_data["bleu"]["bleu1"])
                bleu2_scores.append(seed_data["bleu"]["bleu2"])
                bleu3_scores.append(seed_data["bleu"]["bleu3"])
                bleu4_scores.append(seed_data["bleu"]["bleu4"])
        
        if bleu_scores:
            summary["bleu"] = {
                "mean": sum(bleu_scores) / len(bleu_scores),
                "std": (sum([(x - sum(bleu_scores)/len(bleu_scores))**2 for x in bleu_scores]) / len(bleu_scores))**0.5,
                "min": min(bleu_scores),
                "max": max(bleu_scores),
                "scores_by_seed": {seed_str: all_results["results_by_seed"][seed_str]["bleu"]["bleu"] 
                                  for seed_str in all_results["results_by_seed"] 
                                  if "bleu" in all_results["results_by_seed"][seed_str] and "bleu_error" not in all_results["results_by_seed"][seed_str]}
            }
            summary["bleu1"] = {
                "mean": sum(bleu1_scores) / len(bleu1_scores),
                "std": (sum([(x - sum(bleu1_scores)/len(bleu1_scores))**2 for x in bleu1_scores]) / len(bleu1_scores))**0.5,
                "min": min(bleu1_scores),
                "max": max(bleu1_scores)
            }
            summary["bleu2"] = {
                "mean": sum(bleu2_scores) / len(bleu2_scores),
                "std": (sum([(x - sum(bleu2_scores)/len(bleu2_scores))**2 for x in bleu2_scores]) / len(bleu2_scores))**0.5,
                "min": min(bleu2_scores),
                "max": max(bleu2_scores)
            }
            summary["bleu3"] = {
                "mean": sum(bleu3_scores) / len(bleu3_scores),
                "std": (sum([(x - sum(bleu3_scores)/len(bleu3_scores))**2 for x in bleu3_scores]) / len(bleu3_scores))**0.5,
                "min": min(bleu3_scores),
                "max": max(bleu3_scores)
            }
            summary["bleu4"] = {
                "mean": sum(bleu4_scores) / len(bleu4_scores),
                "std": (sum([(x - sum(bleu4_scores)/len(bleu4_scores))**2 for x in bleu4_scores]) / len(bleu4_scores))**0.5,
                "min": min(bleu4_scores),
                "max": max(bleu4_scores)
            }
            
            print(f"\nBLEU Summary (across {len(bleu_scores)} seeds):")
            print(f"  Mean: {summary['bleu']['mean']:.6f}")
            print(f"  Std:  {summary['bleu']['std']:.6f}")
            print(f"  Min:  {summary['bleu']['min']:.6f}")
            print(f"  Max:  {summary['bleu']['max']:.6f}")

    if eval_method in ["bert-score", "bertscore", "both", "all"]:
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for seed_str, seed_data in all_results["results_by_seed"].items():
            if "bertscore" in seed_data and "bertscore_error" not in seed_data:
                f1_scores.append(seed_data["bertscore"]["f1"])
                precision_scores.append(seed_data["bertscore"]["precision"])
                recall_scores.append(seed_data["bertscore"]["recall"])
        
        if f1_scores:
            summary["bertscore"] = {
                "f1": {
                    "mean": sum(f1_scores) / len(f1_scores),
                    "std": (sum([(x - sum(f1_scores)/len(f1_scores))**2 for x in f1_scores]) / len(f1_scores))**0.5,
                    "min": min(f1_scores),
                    "max": max(f1_scores),
                    "scores_by_seed": {seed_str: all_results["results_by_seed"][seed_str]["bertscore"]["f1"] 
                                      for seed_str in all_results["results_by_seed"] 
                                      if "bertscore" in all_results["results_by_seed"][seed_str] and "bertscore_error" not in all_results["results_by_seed"][seed_str]}
                },
                "precision": {
                    "mean": sum(precision_scores) / len(precision_scores),
                    "std": (sum([(x - sum(precision_scores)/len(precision_scores))**2 for x in precision_scores]) / len(precision_scores))**0.5,
                    "min": min(precision_scores),
                    "max": max(precision_scores)
                },
                "recall": {
                    "mean": sum(recall_scores) / len(recall_scores),
                    "std": (sum([(x - sum(recall_scores)/len(recall_scores))**2 for x in recall_scores]) / len(recall_scores))**0.5,
                    "min": min(recall_scores),
                    "max": max(recall_scores)
                }
            }
            
            print(f"\nBERTScore Summary (across {len(f1_scores)} seeds):")
            print(f"  F1 Mean: {summary['bertscore']['f1']['mean']:.6f}")
            print(f"  F1 Std:  {summary['bertscore']['f1']['std']:.6f}")
            print(f"  F1 Min:  {summary['bertscore']['f1']['min']:.6f}")
            print(f"  F1 Max:  {summary['bertscore']['f1']['max']:.6f}")

    all_results["summary"] = summary

    # Determine output file path
    if args.output_file is None:
        output_file = os.path.join(args.matrix_dir, "anchor_seed_evaluation_results.json")
    else:
        if os.path.isabs(args.output_file):
            output_file = args.output_file
        else:
            output_file = os.path.join(args.matrix_dir, args.output_file)

    # Save results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")
