"""
Utility functions for ARISE evaluation
Handles data loading, result saving, and visualization
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load dataset from JSONL file

    Args:
        dataset_path: Path to JSONL dataset file

    Returns:
        List of problem dictionaries
    """
    data = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} problems from {dataset_path}")
    return data


def save_results(results: Dict, output_path: Union[str, Path]) -> None:
    """
    Save evaluation results to JSON file

    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy values for JSON serialization
    results = convert_numpy_to_python(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def convert_numpy_to_python(obj):
    """
    Recursively convert numpy values to Python types for JSON serialization

    Args:
        obj: Object to convert

    Returns:
        Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj


def create_output_dir(base_dir: str = "results/") -> Path:
    """
    Create timestamped output directory

    Args:
        base_dir: Base directory for results

    Returns:
        Path to created directory
    """
    base_path = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_path / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created output directory: {output_path}")
    return output_path


def load_all_results(results_dir: str) -> Dict[str, Dict]:
    """
    Load all result files from a directory

    Args:
        results_dir: Directory containing result files

    Returns:
        Dictionary mapping model names to results
    """
    results_path = Path(results_dir)
    all_results = {}

    for json_file in results_path.glob("*.json"):
        if json_file.name != "all_results.json":
            with open(json_file, "r") as f:
                results = json.load(f)
                model_name = json_file.stem.replace("_results", "")
                all_results[model_name] = results

    return all_results


def create_comparison_dataframe(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison DataFrame from multiple model results

    Args:
        results_dict: Dictionary of model results

    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []

    for model_name, results in results_dict.items():
        row = {"Model": model_name, "ARISE Score": results.get("arise_score", 0), "Traditional Scaling": results.get("traditional_scaling_metric", 0), "Final Accuracy": results.get("accuracy_by_level", [0])[-1] if results.get("accuracy_by_level") else 0, "Avg Tokens": np.mean(results.get("tokens_by_level", [0])) if results.get("tokens_by_level") else 0, "Negative Samples": results.get("negative_scaling_analysis", {}).get("num_negative_samples", 0)}
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df = df.sort_values("ARISE Score", ascending=False)

    return df


def plot_scaling_curves(results_dict: Dict[str, Dict], save_path: Optional[str] = None) -> None:
    """
    Plot scaling curves for multiple models

    Args:
        results_dict: Dictionary of model results
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, results in results_dict.items():
        if "accuracy_by_level" in results and "tokens_by_level" in results:
            tokens = results["tokens_by_level"]
            accuracies = results["accuracy_by_level"]

            # Accuracy vs Tokens
            ax1.plot(tokens, accuracies, marker="o", label=model_name)

            # ARISE contributions by level (if available)
            if "sample_results" in results:
                arise_by_level = []
                for level in range(1, len(tokens)):
                    level_arise = []
                    for sample in results["sample_results"]:
                        if len(sample["accuracies"]) > level:
                            # Calculate contribution for this level transition
                            delta_a = sample["accuracies"][level] - sample["accuracies"][level - 1]
                            if delta_a != 0:
                                token_ratio = sample["tokens"][level - 1] / sample["tokens"][level]
                                contribution = delta_a * (token_ratio ** np.sign(delta_a))
                                level_arise.append(contribution)
                    arise_by_level.append(np.mean(level_arise) if level_arise else 0)

                ax2.plot(range(1, len(arise_by_level) + 1), arise_by_level, marker="s", label=model_name)

    ax1.set_xlabel("Average Tokens")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Scaling Curves: Accuracy vs Tokens")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Scaling Level")
    ax2.set_ylabel("ARISE Contribution")
    ax2.set_title("ARISE Contributions by Level")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_arise_comparison(results_dict: Dict[str, Dict], save_path: Optional[str] = None) -> None:
    """
    Create bar plot comparing ARISE and traditional scaling metrics

    Args:
        results_dict: Dictionary of model results
        save_path: Optional path to save figure
    """
    models = []
    arise_scores = []
    traditional_scores = []

    for model_name, results in results_dict.items():
        models.append(model_name)
        arise_scores.append(results.get("arise_score", 0))
        traditional_scores.append(results.get("traditional_scaling_metric", 0))

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, arise_scores, width, label="ARISE", color="#2ecc71")
    bars2 = ax.bar(x + width / 2, traditional_scores, width, label="Traditional", color="#3498db")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("ARISE vs Traditional Scaling Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def analyze_sample_transitions(sample_results: List[Dict]) -> Dict:
    """
    Analyze sample-level accuracy transitions

    Args:
        sample_results: List of sample results

    Returns:
        Transition analysis dictionary
    """
    transitions = {"always_correct": [], "always_incorrect": [], "improved": [], "degraded": [], "mixed": []}

    for i, sample in enumerate(sample_results):
        accs = sample["accuracies"]

        if all(a == 1 for a in accs):
            transitions["always_correct"].append(i)
        elif all(a == 0 for a in accs):
            transitions["always_incorrect"].append(i)
        else:
            has_improvement = any(accs[j] > accs[j - 1] for j in range(1, len(accs)))
            has_degradation = any(accs[j] < accs[j - 1] for j in range(1, len(accs)))

            if has_improvement and not has_degradation:
                transitions["improved"].append(i)
            elif has_degradation and not has_improvement:
                transitions["degraded"].append(i)
            else:
                transitions["mixed"].append(i)

    # Calculate statistics
    total = len(sample_results)
    stats = {"counts": {k: len(v) for k, v in transitions.items()}, "percentages": {k: len(v) / total * 100 for k, v in transitions.items()}, "indices": transitions}

    return stats


def generate_latex_table(comparison_df: pd.DataFrame, caption: str = "Model Performance Comparison", label: str = "tab:comparison") -> str:
    """
    Generate LaTeX table from comparison DataFrame

    Args:
        comparison_df: DataFrame with comparison metrics
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table string
    """
    latex = comparison_df.to_latex(index=False, float_format="%.4f", column_format="l" + "c" * (len(comparison_df.columns) - 1), caption=caption, label=label)

    # Add booktabs styling
    latex = latex.replace("\\toprule", "\\toprule\n\\midrule")
    latex = latex.replace("\\bottomrule", "\\midrule\n\\bottomrule")

    return latex


def calculate_statistical_significance(results1: List[float], results2: List[float], test_type: str = "paired") -> Dict:
    """
    Calculate statistical significance between two sets of results

    Args:
        results1: First set of results
        results2: Second set of results
        test_type: Type of test ('paired' or 'independent')

    Returns:
        Dictionary with test statistics
    """
    from scipy import stats

    if test_type == "paired":
        statistic, p_value = stats.ttest_rel(results1, results2)
    else:
        statistic, p_value = stats.ttest_ind(results1, results2)

    # Calculate effect size (Cohen's d)
    mean_diff = np.mean(results1) - np.mean(results2)
    pooled_std = np.sqrt((np.std(results1) ** 2 + np.std(results2) ** 2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    return {"statistic": statistic, "p_value": p_value, "cohens_d": cohens_d, "significant": p_value < 0.05, "mean_diff": mean_diff}


def export_for_paper(results_dict: Dict[str, Dict], output_dir: str = "paper_results/") -> None:
    """
    Export results in formats suitable for paper submission

    Args:
        results_dict: Dictionary of all model results
        output_dir: Directory to save exported files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create comparison DataFrame
    df = create_comparison_dataframe(results_dict)

    # Save as CSV
    df.to_csv(output_path / "comparison.csv", index=False)

    # Save as LaTeX
    with open(output_path / "comparison_table.tex", "w") as f:
        f.write(generate_latex_table(df))

    # Create and save plots
    plot_scaling_curves(results_dict, output_path / "scaling_curves.pdf")
    plot_arise_comparison(results_dict, output_path / "metric_comparison.pdf")

    # Generate summary statistics
    summary = {"best_arise_model": df.iloc[0]["Model"], "best_arise_score": df.iloc[0]["ARISE Score"], "arise_range": [df["ARISE Score"].min(), df["ARISE Score"].max()], "arise_mean": df["ARISE Score"].mean(), "arise_std": df["ARISE Score"].std()}

    with open(output_path / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Paper results exported to {output_path}")
