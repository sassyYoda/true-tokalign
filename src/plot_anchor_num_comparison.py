#!/usr/bin/env python
"""
Create comparison plots for anchor number ablation results.
Shows separate plots for each BLEU metric with bars for each anchor number.
"""

import json
import argparse
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_file):
    """Load evaluation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_bleu_by_metric(results_by_anchor, output_dir):
    """Plot separate BLEU metric plots, each showing bars for different anchor numbers."""
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    metric_keys = ['bleu1', 'bleu2', 'bleu3', 'bleu4']
    
    # Get anchor numbers and sort them
    anchor_nums = sorted([int(k) for k in results_by_anchor.keys()])
    
    # Define colors for each anchor number
    colors = {
        100: '#1f77b4',   # blue
        300: '#ff7f0e',   # orange
        500: '#2ca02c',   # green
        1000: '#d62728'   # red
    }
    # If we have other anchor numbers, use a colormap
    if len(anchor_nums) > len(colors):
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')
        for i, an in enumerate(anchor_nums):
            if an not in colors:
                colors[an] = cmap(i / len(anchor_nums))
    
    # Create a separate plot for each BLEU metric
    for metric, metric_key in zip(metrics, metric_keys):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract scores for this metric across all anchor numbers
        scores = []
        anchor_labels = []
        bar_colors = []
        
        for anchor_num in anchor_nums:
            anchor_str = str(anchor_num)
            if anchor_str in results_by_anchor:
                result = results_by_anchor[anchor_str]
                if "bleu" in result and metric_key in result["bleu"]:
                    scores.append(result["bleu"][metric_key])
                    anchor_labels.append(str(anchor_num))
                    bar_colors.append(colors.get(anchor_num, '#9467bd'))
        
        if not scores:
            print(f"Warning: No BLEU data found for {metric}, skipping...")
            plt.close()
            continue
        
        # Create bars
        x = np.arange(len(anchor_labels))
        bars = ax.bar(x, scores, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Anchors', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} by Number of Anchors', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(anchor_labels)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set y-axis limits
        if scores:
            y_max = max(scores) * 1.15
            ax.set_ylim([0, y_max])
        
        plt.tight_layout()
        
        # Save plot
        filename = metric.lower().replace('-', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f'{filename}_by_anchors.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric} plot to: {output_path}")
        plt.close()

def plot_all_bleu_combined(results_by_anchor, output_dir):
    """Plot all BLEU metrics in a single figure with subplots."""
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    metric_keys = ['bleu1', 'bleu2', 'bleu3', 'bleu4']
    
    # Get anchor numbers and sort them
    anchor_nums = sorted([int(k) for k in results_by_anchor.keys()])
    
    # Define colors for each anchor number
    colors = {
        100: '#1f77b4',   # blue
        300: '#ff7f0e',   # orange
        500: '#2ca02c',   # green
        1000: '#d62728'   # red
    }
    # If we have other anchor numbers, use a colormap
    if len(anchor_nums) > len(colors):
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')
        for i, an in enumerate(anchor_nums):
            if an not in colors:
                colors[an] = cmap(i / len(anchor_nums))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, metric_key) in enumerate(zip(metrics, metric_keys)):
        ax = axes[idx]
        
        # Extract scores for this metric across all anchor numbers
        scores = []
        anchor_labels = []
        bar_colors = []
        
        for anchor_num in anchor_nums:
            anchor_str = str(anchor_num)
            if anchor_str in results_by_anchor:
                result = results_by_anchor[anchor_str]
                if "bleu" in result and metric_key in result["bleu"]:
                    scores.append(result["bleu"][metric_key])
                    anchor_labels.append(str(anchor_num))
                    bar_colors.append(colors.get(anchor_num, '#9467bd'))
        
        if scores:
            # Create bars
            x = np.arange(len(anchor_labels))
            bars = ax.bar(x, scores, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.4f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel('BLEU Score', fontsize=11, fontweight='bold')
            ax.set_xlabel('Number of Anchors', fontsize=11, fontweight='bold')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(anchor_labels)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Set y-axis limits
            y_max = max(scores) * 1.15
            ax.set_ylim([0, y_max])
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric, fontsize=12)
    
    plt.suptitle('BLEU Scores by Number of Anchors', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'all_bleu_by_anchors.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined BLEU plot to: {output_path}")
    plt.close()

def plot_bertscore_by_metric(results_by_anchor, output_dir):
    """Plot separate BERTScore metric plots, each showing bars for different anchor numbers."""
    metrics = ['Precision', 'Recall', 'F1']
    metric_keys = ['precision', 'recall', 'f1']
    
    # Get anchor numbers and sort them
    anchor_nums = sorted([int(k) for k in results_by_anchor.keys()])
    
    # Define colors for each anchor number
    colors = {
        100: '#1f77b4',   # blue
        300: '#ff7f0e',   # orange
        500: '#2ca02c',   # green
        1000: '#d62728'   # red
    }
    # If we have other anchor numbers, use a colormap
    if len(anchor_nums) > len(colors):
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')
        for i, an in enumerate(anchor_nums):
            if an not in colors:
                colors[an] = cmap(i / len(anchor_nums))
    
    # Create a separate plot for each BERTScore metric
    for metric, metric_key in zip(metrics, metric_keys):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract scores for this metric across all anchor numbers
        scores = []
        anchor_labels = []
        bar_colors = []
        
        for anchor_num in anchor_nums:
            anchor_str = str(anchor_num)
            if anchor_str in results_by_anchor:
                result = results_by_anchor[anchor_str]
                if "bertscore" in result and metric_key in result["bertscore"]:
                    scores.append(result["bertscore"][metric_key])
                    anchor_labels.append(str(anchor_num))
                    bar_colors.append(colors.get(anchor_num, '#9467bd'))
        
        if not scores:
            print(f"Warning: No BERTScore data found for {metric}, skipping...")
            plt.close()
            continue
        
        # Create bars
        x = np.arange(len(anchor_labels))
        bars = ax.bar(x, scores, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('BERTScore', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Anchors', fontsize=12, fontweight='bold')
        ax.set_title(f'BERTScore {metric} by Number of Anchors', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(anchor_labels)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        # Save plot
        filename = f'bertscore_{metric_key}_by_anchors.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved BERTScore {metric} plot to: {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot anchor number ablation comparison results')
    parser.add_argument('--results-file', type=str, 
                       default='./anchor_num_evaluations/anchor_num_evaluation_results.json',
                       help='Path to anchor_num_evaluation_results.json file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (default: same as results file directory)')
    parser.add_argument('--no-bertscore', action='store_true',
                       help='Skip BERTScore plots')
    parser.add_argument('--combined-only', action='store_true',
                       help='Only create combined BLEU plot, skip individual plots')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.results_file):
        print(f"ERROR: Results file not found: {args.results_file}")
        print("Please run eval_anchor_num.py first to generate the evaluation results.")
        return
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_file)
        if not args.output_dir:
            args.output_dir = '.'
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    try:
        results = load_results(args.results_file)
    except Exception as e:
        print(f"ERROR: Failed to load results file: {e}")
        return
    
    # Extract results by anchor
    if "results_by_anchor" not in results:
        print("ERROR: No 'results_by_anchor' found in results file.")
        print("Make sure the evaluation completed successfully.")
        return
    
    results_by_anchor = results["results_by_anchor"]
    
    # Filter out entries with errors
    valid_results = {}
    for k, v in results_by_anchor.items():
        if "bleu_error" not in v and "bertscore_error" not in v:
            valid_results[k] = v
    
    if not valid_results:
        print("ERROR: No valid results found (all evaluations had errors).")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving plots to: {args.output_dir}\n")
    
    # Generate BLEU plots
    if not args.combined_only:
        plot_bleu_by_metric(valid_results, args.output_dir)
    
    plot_all_bleu_combined(valid_results, args.output_dir)
    
    # Generate BERTScore plots
    if not args.no_bertscore:
        plot_bertscore_by_metric(valid_results, args.output_dir)
    
    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()
