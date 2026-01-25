"""
Aggregate results from multiple CulturaX perplexity evaluation trials.
Creates a summary table similar to the TokAlign paper.
"""

import argparse
import json
import os
from typing import Dict, List
import numpy as np
import pandas as pd
from pathlib import Path

# Language resource levels
LANGUAGE_RESOURCES = {
    "high": ["ar", "de", "en", "ja", "zh"],
    "medium": ["bn", "ko", "th", "uk", "vi"],
    "low": ["ta", "te", "ur"]
}

ALL_LANGUAGES = [lang for langs in LANGUAGE_RESOURCES.values() for lang in langs]


def load_trial_results(results_dir: str) -> List[Dict]:
    """Load all trial results from a directory.
    
    Looks for files matching: culturax_perplexity_results_*.json
    You can manually rename files to include trial numbers if running multiple trials.
    """
    results = []
    results_dir = Path(results_dir)
    
    # Find all JSON result files
    for json_file in sorted(results_dir.glob("culturax_perplexity_results*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)
                print(f"Loaded: {json_file.name}")
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def aggregate_trials(trial_results: List[Dict]) -> Dict:
    """Aggregate multiple trial results, computing means and stds."""
    if not trial_results:
        raise ValueError("No trial results to aggregate")
    
    # Group by language
    language_data = {lang: [] for lang in ALL_LANGUAGES}
    
    for trial in trial_results:
        for lang, lang_results in trial.get("languages", {}).items():
            if lang in language_data:
                language_data[lang].append(lang_results["perplexity"])
    
    # Compute statistics
    aggregated = {}
    for lang in ALL_LANGUAGES:
        if language_data[lang]:
            aggregated[lang] = {
                "mean": np.mean(language_data[lang]),
                "std": np.std(language_data[lang]),
                "values": language_data[lang],
                "n_trials": len(language_data[lang])
            }
        else:
            aggregated[lang] = {
                "mean": np.nan,
                "std": np.nan,
                "values": [],
                "n_trials": 0
            }
    
    # Compute averages by resource level
    by_resource = {}
    for level, langs in LANGUAGE_RESOURCES.items():
        level_values = []
        for lang in langs:
            if lang in aggregated and not np.isnan(aggregated[lang]["mean"]):
                level_values.append(aggregated[lang]["mean"])
        
        if level_values:
            by_resource[level] = {
                "mean": np.mean(level_values),
                "std": np.std(level_values) if len(level_values) > 1 else 0.0
            }
        else:
            by_resource[level] = {"mean": np.nan, "std": np.nan}
    
    # Overall average
    all_values = [v["mean"] for v in aggregated.values() if not np.isnan(v["mean"])]
    overall_avg = np.mean(all_values) if all_values else np.nan
    
    return {
        "languages": aggregated,
        "by_resource": by_resource,
        "overall_average": overall_avg,
        "n_trials": len(trial_results),
        "model_path": trial_results[0]["model_path"] if trial_results else None
    }


def create_summary_table(aggregated: Dict, output_path: str):
    """Create a formatted summary table."""
    model_name = os.path.basename(aggregated["model_path"]) if aggregated["model_path"] else "Unknown"
    
    # Create table data
    table_data = []
    
    # Header
    header = ["Language", "Perplexity (Mean)", "Std Dev", "N Trials"]
    table_data.append(header)
    
    # By resource level
    for level in ["high", "medium", "low"]:
        level_name = level.capitalize() + " Resource"
        table_data.append([level_name, "", "", ""])
        
        for lang in LANGUAGE_RESOURCES[level]:
            if lang in aggregated["languages"]:
                lang_data = aggregated["languages"][lang]
                if not np.isnan(lang_data["mean"]):
                    mean_str = f"{lang_data['mean']:.4f}"
                    std_str = f"{lang_data['std']:.4f}" if lang_data['std'] > 0 else "-"
                    n_str = str(lang_data["n_trials"])
                    table_data.append([f"  {lang}", mean_str, std_str, n_str])
        
        # Level average
        if level in aggregated["by_resource"]:
            level_avg = aggregated["by_resource"][level]
            if not np.isnan(level_avg["mean"]):
                avg_str = f"{level_avg['mean']:.4f}"
                std_str = f"{level_avg['std']:.4f}" if level_avg['std'] > 0 else "-"
                table_data.append([f"  Average", avg_str, std_str, str(aggregated["n_trials"])])
    
    # Overall average
    if not np.isnan(aggregated["overall_average"]):
        table_data.append(["Overall Average", f"{aggregated['overall_average']:.4f}", "", ""])
    
    # Save as CSV
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    df.to_csv(output_path, index=False)
    
    # Also save as formatted text
    txt_path = output_path.replace(".csv", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Number of Trials: {aggregated['n_trials']}\n")
        f.write("="*80 + "\n\n")
        
        for row in table_data:
            f.write(f"{row[0]:20s} {row[1]:15s} {row[2]:15s} {row[3]:10s}\n")
    
    print(f"Summary table saved to {output_path}")
    print(f"Formatted text saved to {txt_path}")
    
    return output_path


def create_latex_table(aggregated: Dict, output_path: str):
    """Create a LaTeX table similar to the TokAlign paper."""
    model_name = os.path.basename(aggregated["model_path"]) if aggregated["model_path"] else "Model"
    
    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Normalized Perplexity on CulturaX}")
    latex_lines.append("\\label{tab:culturax-perplexity}")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(ALL_LANGUAGES) + "c}")
    latex_lines.append("\\toprule")
    
    # Header row
    header = "Model & " + " & ".join(ALL_LANGUAGES) + " & Avg $\\downarrow$ \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Data row
    row_values = [model_name]
    for lang in ALL_LANGUAGES:
        if lang in aggregated["languages"]:
            val = aggregated["languages"][lang]["mean"]
            if not np.isnan(val):
                row_values.append(f"{val:.2f}")
            else:
                row_values.append("-")
        else:
            row_values.append("-")
    
    # Overall average
    if not np.isnan(aggregated["overall_average"]):
        row_values.append(f"{aggregated['overall_average']:.2f}")
    else:
        row_values.append("-")
    
    latex_lines.append(" & ".join(row_values) + " \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    
    print(f"LaTeX table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple CulturaX perplexity evaluation trials"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing trial result JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save aggregated results (default: same as results_dir)"
    )
    
    args = parser.parse_args()
    
    # Load trial results
    print(f"Loading trial results from {args.results_dir}...")
    trial_results = load_trial_results(args.results_dir)
    
    if not trial_results:
        raise ValueError(f"No trial results found in {args.results_dir}")
    
    print(f"Loaded {len(trial_results)} trial(s)")
    
    # Aggregate
    print("Aggregating results...")
    aggregated = aggregate_trials(trial_results)
    
    # Save aggregated results
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, "aggregated_culturax_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    print(f"Aggregated results saved to {json_path}")
    
    # Create summary table
    csv_path = os.path.join(output_dir, "aggregated_culturax_summary.csv")
    create_summary_table(aggregated, csv_path)
    
    # Create LaTeX table
    latex_path = os.path.join(output_dir, "aggregated_culturax_table.tex")
    create_latex_table(aggregated, latex_path)
    
    # Print summary
    print("\n" + "="*80)
    print("Aggregation Summary")
    print("="*80)
    print(f"Model: {os.path.basename(aggregated['model_path'])}")
    print(f"Number of trials: {aggregated['n_trials']}")
    print(f"\nBy Resource Level:")
    for level in ["high", "medium", "low"]:
        if level in aggregated["by_resource"]:
            avg = aggregated["by_resource"][level]["mean"]
            if not np.isnan(avg):
                print(f"  {level.capitalize()}: {avg:.4f}")
    if not np.isnan(aggregated["overall_average"]):
        print(f"\nOverall Average: {aggregated['overall_average']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()

