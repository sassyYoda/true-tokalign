#!/usr/bin/env python
"""
Compare relative representation (relrep) seed ablation results with vanilla baseline.
Creates bar graphs comparing average BLEU and BERTScore metrics.
"""

import json
import argparse
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Vanilla baseline values (Pythia-Qwen GloVE)
VANILLA_BASELINE = {
    "bleu": {
        "bleu1": 0.520546,
        "bleu2": 0.388945,
        "bleu3": 0.307189,
        "bleu4": 0.249180
    },
    "bertscore": {
        "precision": 0.453112,
        "recall": 0.494601,
        "f1": 0.473979
    }
}

def load_results(results_file):
    """Load evaluation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_bleu_comparison(relrep_summary, output_dir):
    """Plot BLEU score comparison between relrep and vanilla."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    x = np.arange(len(metrics))
    width = 0.35
    
    # Extract relrep values from summary
    relrep_scores = [
        relrep_summary.get("bleu1", {}).get("mean", 0),
        relrep_summary.get("bleu2", {}).get("mean", 0),
        relrep_summary.get("bleu3", {}).get("mean", 0),
        relrep_summary.get("bleu4", {}).get("mean", 0)
    ]
    
    # Vanilla baseline values
    vanilla_scores = [
        VANILLA_BASELINE["bleu"]["bleu1"],
        VANILLA_BASELINE["bleu"]["bleu2"],
        VANILLA_BASELINE["bleu"]["bleu3"],
        VANILLA_BASELINE["bleu"]["bleu4"]
    ]
    
    # Create bars
    bars1 = ax.bar(x - width/2, vanilla_scores, width, label='Vanilla (Baseline)', 
                   color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, relrep_scores, width, label='RelRep (Mean)', 
                   color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax.set_title('BLEU Score Comparison: Vanilla vs RelRep', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(max(vanilla_scores), max(relrep_scores)) * 1.2])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'bleu_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved BLEU comparison plot to: {output_path}")
    plt.close()

def plot_bertscore_comparison(relrep_summary, output_dir):
    """Plot BERTScore comparison between relrep and vanilla."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Precision', 'Recall', 'F1']
    x = np.arange(len(metrics))
    width = 0.35
    
    # Extract relrep values from summary
    relrep_scores = [
        relrep_summary.get("bertscore", {}).get("precision", {}).get("mean", 0),
        relrep_summary.get("bertscore", {}).get("recall", {}).get("mean", 0),
        relrep_summary.get("bertscore", {}).get("f1", {}).get("mean", 0)
    ]
    
    # Vanilla baseline values
    vanilla_scores = [
        VANILLA_BASELINE["bertscore"]["precision"],
        VANILLA_BASELINE["bertscore"]["recall"],
        VANILLA_BASELINE["bertscore"]["f1"]
    ]
    
    # Create bars
    bars1 = ax.bar(x - width/2, vanilla_scores, width, label='Vanilla (Baseline)', 
                   color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, relrep_scores, width, label='RelRep (Mean)', 
                   color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('BERTScore', fontsize=12, fontweight='bold')
    ax.set_title('BERTScore Comparison: Vanilla vs RelRep', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'bertscore_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved BERTScore comparison plot to: {output_path}")
    plt.close()

def plot_combined_comparison(relrep_summary, output_dir):
    """Plot combined comparison of BLEU and BERTScore."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # BLEU subplot
    metrics_bleu = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    x_bleu = np.arange(len(metrics_bleu))
    width = 0.35
    
    relrep_bleu = [
        relrep_summary.get("bleu1", {}).get("mean", 0),
        relrep_summary.get("bleu2", {}).get("mean", 0),
        relrep_summary.get("bleu3", {}).get("mean", 0),
        relrep_summary.get("bleu4", {}).get("mean", 0)
    ]
    
    vanilla_bleu = [
        VANILLA_BASELINE["bleu"]["bleu1"],
        VANILLA_BASELINE["bleu"]["bleu2"],
        VANILLA_BASELINE["bleu"]["bleu3"],
        VANILLA_BASELINE["bleu"]["bleu4"]
    ]
    
    bars1_bleu = ax1.bar(x_bleu - width/2, vanilla_bleu, width, label='Vanilla', 
                         color='#1f77b4', alpha=0.7, edgecolor='black')
    bars2_bleu = ax1.bar(x_bleu + width/2, relrep_bleu, width, label='RelRep', 
                         color='#ff7f0e', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1_bleu, bars2_bleu]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('BLEU Score', fontsize=11, fontweight='bold')
    ax1.set_title('BLEU Scores', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_bleu)
    ax1.set_xticklabels(metrics_bleu, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, max(max(vanilla_bleu), max(relrep_bleu)) * 1.2])
    
    # BERTScore subplot
    metrics_bert = ['Precision', 'Recall', 'F1']
    x_bert = np.arange(len(metrics_bert))
    
    relrep_bert = [
        relrep_summary.get("bertscore", {}).get("precision", {}).get("mean", 0),
        relrep_summary.get("bertscore", {}).get("recall", {}).get("mean", 0),
        relrep_summary.get("bertscore", {}).get("f1", {}).get("mean", 0)
    ]
    
    vanilla_bert = [
        VANILLA_BASELINE["bertscore"]["precision"],
        VANILLA_BASELINE["bertscore"]["recall"],
        VANILLA_BASELINE["bertscore"]["f1"]
    ]
    
    bars1_bert = ax2.bar(x_bert - width/2, vanilla_bert, width, label='Vanilla', 
                         color='#2ca02c', alpha=0.7, edgecolor='black')
    bars2_bert = ax2.bar(x_bert + width/2, relrep_bert, width, label='RelRep', 
                         color='#ff7f0e', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1_bert, bars2_bert]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylabel('BERTScore', fontsize=11, fontweight='bold')
    ax2.set_title('BERTScore Metrics', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_bert)
    ax2.set_xticklabels(metrics_bert)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.0])
    
    plt.suptitle('Token Alignment Comparison: Vanilla vs RelRep (Mean)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'combined_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined comparison plot to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare relrep seed ablation results with vanilla baseline')
    parser.add_argument('--results-file', type=str, 
                       default='./alignment_matrix_seed_ablations/anchor_seed_evaluation_results.json',
                       help='Path to anchor_seed_evaluation_results.json file')
    parser.add_argument('--output-dir', type=str, default='./alignment_matrix_seed_ablations',
                       help='Directory to save plots (default: ./alignment_matrix_seed_ablations)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.results_file):
        print(f"ERROR: Results file not found: {args.results_file}")
        print("Please run eval_anchors.py first to generate the evaluation results.")
        return
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    try:
        results = load_results(args.results_file)
    except Exception as e:
        print(f"ERROR: Failed to load results file: {e}")
        return
    
    # Extract summary statistics
    if "summary" not in results:
        print("ERROR: No summary statistics found in results file.")
        print("Make sure the evaluation completed successfully.")
        return
    
    relrep_summary = results["summary"]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving plots to: {args.output_dir}\n")
    
    # Generate comparison plots
    if "bleu" in relrep_summary or "bleu1" in relrep_summary:
        plot_bleu_comparison(relrep_summary, args.output_dir)
    
    if "bertscore" in relrep_summary:
        plot_bertscore_comparison(relrep_summary, args.output_dir)
    
    plot_combined_comparison(relrep_summary, args.output_dir)
    
    print("\nAll comparison plots generated successfully!")
    print(f"\nPlots saved:")
    print(f"  - bleu_comparison.png")
    print(f"  - bertscore_comparison.png")
    print(f"  - combined_comparison.png")

if __name__ == '__main__':
    main()
