#!/usr/bin/env python
"""
Generate plots and visualizations from evaluation results.
Saves plots to the figure/ directory.
"""

import json
import argparse
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(results_file):
    """Load evaluation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_bleu_scores(results, output_dir):
    """Plot BLEU score breakdown."""
    if "bleu" not in results:
        print("No BLEU results found, skipping BLEU plots.")
        return
    
    bleu_data = results["bleu"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'Average BLEU']
    scores = [
        bleu_data['bleu1'],
        bleu_data['bleu2'],
        bleu_data['bleu3'],
        bleu_data['bleu4'],
        bleu_data['bleu']
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(metrics, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax.set_title('BLEU Score Evaluation Results', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(scores) * 1.2])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'bleu_scores.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved BLEU plot to: {output_path}")
    plt.close()

def plot_bertscore_metrics(results, output_dir):
    """Plot BERTScore metrics (Precision, Recall, F1)."""
    if "bertscore" not in results:
        print("No BERTScore results found, skipping BERTScore plots.")
        return
    
    bertscore_data = results["bertscore"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Precision', 'Recall', 'F1']
    scores = [
        bertscore_data['precision'],
        bertscore_data['recall'],
        bertscore_data['f1']
    ]
    
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    bars = ax.bar(metrics, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('BERTScore', fontsize=12, fontweight='bold')
    ax.set_title('BERTScore Evaluation Results', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'bertscore_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved BERTScore plot to: {output_path}")
    plt.close()

def plot_combined_comparison(results, output_dir):
    """Plot combined comparison of BLEU and BERTScore."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # BLEU subplot
    if "bleu" in results:
        bleu_data = results["bleu"]
        bleu_metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
        bleu_scores = [
            bleu_data['bleu1'],
            bleu_data['bleu2'],
            bleu_data['bleu3'],
            bleu_data['bleu4']
        ]
        
        ax1.bar(bleu_metrics, bleu_scores, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('BLEU Score', fontsize=11, fontweight='bold')
        ax1.set_title('BLEU Scores', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, max(bleu_scores) * 1.2 if bleu_scores else 1.0])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, (metric, score) in enumerate(zip(bleu_metrics, bleu_scores)):
            ax1.text(i, score, f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No BLEU data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('BLEU Scores', fontsize=12)
    
    # BERTScore subplot
    if "bertscore" in results:
        bertscore_data = results["bertscore"]
        bertscore_metrics = ['Precision', 'Recall', 'F1']
        bertscore_scores = [
            bertscore_data['precision'],
            bertscore_data['recall'],
            bertscore_data['f1']
        ]
        
        ax2.bar(bertscore_metrics, bertscore_scores, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('BERTScore', fontsize=11, fontweight='bold')
        ax2.set_title('BERTScore Metrics', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1.0])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (metric, score) in enumerate(zip(bertscore_metrics, bertscore_scores)):
            ax2.text(i, score, f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No BERTScore data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('BERTScore Metrics', fontsize=12)
    
    plt.suptitle('Token Alignment Evaluation Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'evaluation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined comparison plot to: {output_path}")
    plt.close()

def plot_score_distributions(results, output_dir):
    """Plot distribution of individual BERTScore scores."""
    if "bertscore" not in results or "f1_scores" not in results["bertscore"]:
        print("No BERTScore distribution data found, skipping distribution plots.")
        return
    
    bertscore_data = results["bertscore"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['precision', 'recall', 'f1']
    metric_labels = ['Precision', 'Recall', 'F1']
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        scores = bertscore_data[f'{metric}_scores']
        
        ax.hist(scores, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(scores):.4f}')
        ax.axvline(np.median(scores), color='blue', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(scores):.4f}')
        
        ax.set_xlabel(f'{label} Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('BERTScore Score Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'bertscore_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved BERTScore distribution plot to: {output_path}")
    plt.close()

def plot_summary_table(results, output_dir):
    """Create a summary table visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    
    if "bleu" in results:
        bleu_data = results["bleu"]
        table_data.append(['BLEU-1', f"{bleu_data['bleu1']:.6f}"])
        table_data.append(['BLEU-2', f"{bleu_data['bleu2']:.6f}"])
        table_data.append(['BLEU-3', f"{bleu_data['bleu3']:.6f}"])
        table_data.append(['BLEU-4', f"{bleu_data['bleu4']:.6f}"])
        table_data.append(['Average BLEU', f"{bleu_data['bleu']:.6f}"])
        table_data.append(['', ''])  # Separator
    
    if "bertscore" in results:
        bertscore_data = results["bertscore"]
        table_data.append(['BERTScore Precision', f"{bertscore_data['precision']:.6f}"])
        table_data.append(['BERTScore Recall', f"{bertscore_data['recall']:.6f}"])
        table_data.append(['BERTScore F1', f"{bertscore_data['f1']:.6f}"])
    
    if table_data:
        # Remove trailing separator if present
        if table_data[-1] == ['', '']:
            table_data.pop()
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Score'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('Evaluation Results Summary', fontsize=14, fontweight='bold', pad=20)
        
        output_path = os.path.join(output_dir, 'evaluation_summary_table.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary table to: {output_path}")
        plt.close()
    else:
        print("No evaluation data found for summary table.")

def main():
    parser = argparse.ArgumentParser(description='Generate plots from evaluation results')
    parser.add_argument('--results-file', type=str, required=True,
                       help='Path to evaluation_results.json file')
    parser.add_argument('--output-dir', type=str, default='./figure',
                       help='Directory to save plots (default: ./figure)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.results_file):
        print(f"ERROR: Results file not found: {args.results_file}")
        print("Please run the evaluation first.")
        return
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    try:
        results = load_results(args.results_file)
    except Exception as e:
        print(f"ERROR: Failed to load results file: {e}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving plots to: {args.output_dir}\n")
    
    # Generate all plots
    plot_bleu_scores(results, args.output_dir)
    plot_bertscore_metrics(results, args.output_dir)
    plot_combined_comparison(results, args.output_dir)
    plot_score_distributions(results, args.output_dir)
    plot_summary_table(results, args.output_dir)
    
    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()

