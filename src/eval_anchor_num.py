#!/usr/bin/env python
"""
Evaluate alignment matrices with different numbers of anchors and create comparison plots.
Combines evaluation and plotting functionality.
"""

import json
import os
import argparse
import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Import evaluation functions from eval_matrix.py
from eval_matrix import eval_trans_matrix, eval_bert_score

def extract_anchor_num_from_filename(filename):
    """Extract anchor number from filename like 'alignment_matrix_anchors_300.json'"""
    basename = os.path.basename(filename)
    if 'anchors_' in basename:
        try:
            anchor_str = basename.split('anchors_')[1].split('.json')[0]
            return int(anchor_str)
        except (ValueError, IndexError):
            return None
    return None

def find_alignment_matrices(matrix_dir):
    """Find all alignment matrix JSON files in the directory."""
    pattern = os.path.join(matrix_dir, "alignment_matrix_anchors_*.json")
    matrix_files = glob.glob(pattern)
    # Sort by anchor number
    matrix_files.sort(key=lambda x: extract_anchor_num_from_filename(x) or 0)
    return matrix_files

def plot_bleu_by_anchors(results_by_anchor, output_dir):
    """Plot BLEU scores across different anchor numbers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    anchor_nums = sorted([int(k) for k in results_by_anchor.keys()])
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    
    x = np.arange(len(anchor_nums))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        metric_key = metric.lower().replace('-', '')
        scores = [results_by_anchor[str(an)].get("bleu", {}).get(metric_key, 0) 
                  for an in anchor_nums]
        offset = (i - 1.5) * width
        ax.bar(x + offset, scores, width, label=metric, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Anchors', fontsize=12, fontweight='bold')
    ax.set_title('BLEU Scores by Number of Anchors', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(an) for an in anchor_nums])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'bleu_by_anchors.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved BLEU plot to: {output_path}")
    plt.close()

def plot_bertscore_by_anchors(results_by_anchor, output_dir):
    """Plot BERTScore metrics across different anchor numbers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    anchor_nums = sorted([int(k) for k in results_by_anchor.keys()])
    metrics = ['Precision', 'Recall', 'F1']
    
    x = np.arange(len(anchor_nums))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        metric_key = metric.lower()
        scores = [results_by_anchor[str(an)].get("bertscore", {}).get(metric_key, 0) 
                  for an in anchor_nums]
        offset = (i - 1) * width
        ax.bar(x + offset, scores, width, label=metric, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('BERTScore', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Anchors', fontsize=12, fontweight='bold')
    ax.set_title('BERTScore Metrics by Number of Anchors', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(an) for an in anchor_nums])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'bertscore_by_anchors.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved BERTScore plot to: {output_path}")
    plt.close()

def plot_combined_comparison(results_by_anchor, output_dir):
    """Plot combined comparison of BLEU and BERTScore across anchor numbers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    anchor_nums = sorted([int(k) for k in results_by_anchor.keys()])
    x = np.arange(len(anchor_nums))
    width = 0.15
    
    # BLEU subplot
    bleu_metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    colors_bleu = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, color) in enumerate(zip(bleu_metrics, colors_bleu)):
        metric_key = metric.lower().replace('-', '')
        scores = [results_by_anchor[str(an)].get("bleu", {}).get(metric_key, 0) 
                  for an in anchor_nums]
        offset = (i - 1.5) * width
        ax1.bar(x + offset, scores, width, label=metric, color=color, alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('BLEU Score', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Number of Anchors', fontsize=11, fontweight='bold')
    ax1.set_title('BLEU Scores', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(an) for an in anchor_nums])
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # BERTScore subplot
    bert_metrics = ['Precision', 'Recall', 'F1']
    colors_bert = ['#2ca02c', '#1f77b4', '#ff7f0e']
    
    for i, (metric, color) in enumerate(zip(bert_metrics, colors_bert)):
        metric_key = metric.lower()
        scores = [results_by_anchor[str(an)].get("bertscore", {}).get(metric_key, 0) 
                  for an in anchor_nums]
        offset = (i - 1) * width
        ax2.bar(x + offset, scores, width, label=metric, color=color, alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('BERTScore', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Number of Anchors', fontsize=11, fontweight='bold')
    ax2.set_title('BERTScore Metrics', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(an) for an in anchor_nums])
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.0])
    
    plt.suptitle('Token Alignment Performance by Number of Anchors', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'combined_anchor_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined comparison plot to: {output_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--matrix-dir", type=str, default="./anchor_num_evaluations",
                       help="Directory containing alignment matrix anchor number ablation files")
    parser.add_argument("-e", "--evaluate-method", type=str, default="both", 
                       choices=["bleu", "bert-score", "bertscore", "both", "all"],
                       help="Evaluation method: 'bleu', 'bert-score', or 'both'")
    parser.add_argument("-f", "--eval-file-path", type=str, default="./data/pretrain-dataset/pythia-2-qwen2-7b-glove-eval-mix")
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
                       help="Output JSON file name (default: anchor_num_evaluation_results.json in matrix-dir)")
    parser.add_argument("--no-plot", action='store_true', help="Skip plotting, only evaluate")

    args = parser.parse_args()

    # Find all alignment matrices
    matrix_files = find_alignment_matrices(args.matrix_dir)
    
    if not matrix_files:
        print(f"ERROR: No alignment matrix files found in {args.matrix_dir}")
        print(f"Expected pattern: alignment_matrix_anchors_*.json")
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
        "results_by_anchor": {}
    }

    # Evaluate each matrix
    for matrix_file in tqdm(matrix_files, desc="Evaluating matrices"):
        anchor_num = extract_anchor_num_from_filename(matrix_file)
        if anchor_num is None:
            print(f"Warning: Could not extract anchor number from {matrix_file}, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating {anchor_num} anchors")
        print(f"Matrix: {os.path.basename(matrix_file)}")
        print(f"{'='*70}")

        anchor_results = {
            "matrix_file": os.path.basename(matrix_file),
            "num_anchors": anchor_num
        }

        # Run BLEU evaluation
        if eval_method in ["bleu", "both", "all"]:
            try:
                print(f"\nRunning BLEU evaluation for {anchor_num} anchors...")
                bleu_results = eval_trans_matrix(
                    trans_dict_path=matrix_file,
                    eval_file_path=args.eval_file_path,
                    bleu_weights=weights
                )
                anchor_results["bleu"] = bleu_results
            except Exception as e:
                print(f"ERROR: BLEU evaluation failed for {anchor_num} anchors: {e}")
                anchor_results["bleu_error"] = str(e)

        # Run BERTScore evaluation
        if eval_method in ["bert-score", "bertscore", "both", "all"]:
            try:
                print(f"\nRunning BERTScore evaluation for {anchor_num} anchors...")
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
                anchor_results["bertscore"] = bertscore_results
            except Exception as e:
                print(f"ERROR: BERTScore evaluation failed for {anchor_num} anchors: {e}")
                print("Continuing with other anchor numbers...")
                anchor_results["bertscore_error"] = str(e)

        all_results["results_by_anchor"][str(anchor_num)] = anchor_results

    # Determine output file path
    if args.output_file is None:
        output_file = os.path.join(args.matrix_dir, "anchor_num_evaluation_results.json")
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

    # Generate plots if requested
    if not args.no_plot:
        print(f"\n{'='*70}")
        print("Generating comparison plots...")
        print(f"{'='*70}")
        
        # Filter out entries with errors
        results_by_anchor = {
            k: v for k, v in all_results["results_by_anchor"].items()
            if "bleu_error" not in v and "bertscore_error" not in v
        }
        
        if results_by_anchor:
            if eval_method in ["bleu", "both", "all"]:
                # Check if we have BLEU data
                has_bleu = any("bleu" in v for v in results_by_anchor.values())
                if has_bleu:
                    plot_bleu_by_anchors(results_by_anchor, args.matrix_dir)
            
            if eval_method in ["bert-score", "bertscore", "both", "all"]:
                # Check if we have BERTScore data
                has_bert = any("bertscore" in v for v in results_by_anchor.values())
                if has_bert:
                    plot_bertscore_by_anchors(results_by_anchor, args.matrix_dir)
            
            # Combined plot if we have both
            has_bleu = any("bleu" in v for v in results_by_anchor.values())
            has_bert = any("bertscore" in v for v in results_by_anchor.values())
            if has_bleu and has_bert:
                plot_combined_comparison(results_by_anchor, args.matrix_dir)
            
            print(f"\nPlots saved to: {args.matrix_dir}")
            print(f"  - bleu_by_anchors.png")
            print(f"  - bertscore_by_anchors.png")
            print(f"  - combined_anchor_comparison.png")
        else:
            print("No valid results to plot (all evaluations had errors).")
