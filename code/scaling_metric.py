"""
Traditional Scaling Metric Implementation
Used for comparison with ARISE metric
"""

import numpy as np
from typing import List, Dict, Tuple


class TraditionalScalingMetric:
    """
    Implementation of traditional slope-based scaling metric
    """

    def __init__(self):
        pass

    def calculate(self, baseline_accuracy: float, scaled_accuracy: float, baseline_tokens: float, scaled_tokens: float) -> float:
        """
        Calculate traditional scaling metric (slope-based)

        Args:
            baseline_accuracy: Accuracy at baseline
            scaled_accuracy: Accuracy after scaling
            baseline_tokens: Tokens at baseline
            scaled_tokens: Tokens after scaling

        Returns:
            Scaling metric value
        """
        if scaled_tokens <= baseline_tokens:
            return 0.0

        # Calculate slope: Δaccuracy / Δtokens
        accuracy_change = scaled_accuracy - baseline_accuracy
        token_change = scaled_tokens - baseline_tokens

        scaling_metric = accuracy_change / token_change

        return scaling_metric

    def calculate_from_results(self, sample_results: List[Dict]) -> Tuple[float, Dict]:
        """
        Calculate scaling metric from sample results

        Args:
            sample_results: List of sample results with accuracies and tokens

        Returns:
            Tuple of (scaling_metric, statistics)
        """
        if not sample_results:
            return 0.0, {}

        # Get baseline and final statistics
        baseline_accuracies = []
        final_accuracies = []
        baseline_tokens = []
        final_tokens = []

        for sample in sample_results:
            if "accuracies" in sample and "tokens" in sample:
                if len(sample["accuracies"]) >= 2:
                    baseline_accuracies.append(sample["accuracies"][0])
                    final_accuracies.append(sample["accuracies"][-1])
                    baseline_tokens.append(sample["tokens"][0])
                    final_tokens.append(sample["tokens"][-1])

        if not baseline_accuracies:
            return 0.0, {}

        # Calculate aggregate statistics
        baseline_acc = np.mean(baseline_accuracies)
        final_acc = np.mean(final_accuracies)
        baseline_tok = np.mean(baseline_tokens)
        final_tok = np.mean(final_tokens)

        # Calculate scaling metric
        scaling_metric = self.calculate(baseline_acc, final_acc, baseline_tok, final_tok)

        statistics = {"scaling_metric": scaling_metric, "baseline_accuracy": baseline_acc, "final_accuracy": final_acc, "accuracy_improvement": final_acc - baseline_acc, "baseline_tokens": baseline_tok, "final_tokens": final_tok, "token_increase": final_tok - baseline_tok, "efficiency": (final_acc - baseline_acc) / (final_tok - baseline_tok) if final_tok > baseline_tok else 0}

        return scaling_metric, statistics

    def compare_models(self, model_results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Compare multiple models using traditional scaling metric

        Args:
            model_results: Dictionary mapping model names to results

        Returns:
            DataFrame with comparison
        """
        import pandas as pd

        comparison_data = []

        for model_name, results in model_results.items():
            metric, stats = self.calculate_from_results(results)

            row = {"Model": model_name, "Scaling Metric": metric, "Baseline Acc": stats.get("baseline_accuracy", 0), "Final Acc": stats.get("final_accuracy", 0), "Acc Improvement": stats.get("accuracy_improvement", 0), "Token Increase": stats.get("token_increase", 0)}
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("Scaling Metric", ascending=False)

        return df


class ScalingMetricAnalyzer:
    """
    Analyzer for comparing ARISE and traditional scaling metrics
    """

    def __init__(self):
        self.traditional = TraditionalScalingMetric()

    def analyze_metric_behavior(self, sample_results: List[Dict]) -> Dict:
        """
        Analyze behavior of different metrics

        Args:
            sample_results: Sample results to analyze

        Returns:
            Analysis of metric behaviors
        """
        from arise_metric import ARISEMetric

        arise_metric = ARISEMetric()

        # Calculate both metrics
        arise_score, arise_stats = arise_metric.calculate_dataset_arise(sample_results)
        traditional_score, traditional_stats = self.traditional.calculate_from_results(sample_results)

        # Identify cases where metrics disagree
        disagreements = []

        for i, sample in enumerate(sample_results):
            if len(sample["accuracies"]) >= 2:
                # Check if sample improved or degraded
                improved = sample["accuracies"][-1] > sample["accuracies"][0]
                degraded = sample["accuracies"][-1] < sample["accuracies"][0]

                # Calculate sample-level ARISE
                sample_arise = arise_metric.calculate_sample_arise(sample["accuracies"], sample["tokens"])

                # Check for metric disagreements
                if degraded and sample_arise < 0 and traditional_score > 0:
                    disagreements.append({"sample_idx": i, "type": "negative_scaling_penalty", "arise": sample_arise, "description": "ARISE penalizes degradation, traditional does not"})

        analysis = {"arise_score": arise_score, "traditional_score": traditional_score, "correlation": np.corrcoef([s.get("arise_score", 0) for s in sample_results], [traditional_score] * len(sample_results))[0, 1] if len(sample_results) > 1 else 0, "disagreements": disagreements, "num_disagreements": len(disagreements), "arise_stats": arise_stats, "traditional_stats": traditional_stats}

        return analysis

    def plot_metric_comparison(self, sample_results: List[Dict], save_path: Optional[str] = None) -> None:
        """
        Plot comparison between ARISE and traditional metrics

        Args:
            sample_results: Sample results
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        from arise_metric import ARISEMetric

        arise_metric = ARISEMetric()

        # Calculate metrics for different scaling levels
        arise_scores = []
        traditional_scores = []

        max_levels = max(len(s["accuracies"]) for s in sample_results)

        for level in range(2, max_levels + 1):
            # Create truncated results up to this level
            truncated = []
            for sample in sample_results:
                if len(sample["accuracies"]) >= level:
                    truncated.append({"accuracies": sample["accuracies"][:level], "tokens": sample["tokens"][:level]})

            if truncated:
                # Calculate ARISE
                arise_score, _ = arise_metric.calculate_dataset_arise(truncated)
                arise_scores.append(arise_score)

                # Calculate traditional
                trad_score, _ = self.traditional.calculate_from_results(truncated)
                traditional_scores.append(trad_score)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot metric progression
        levels = list(range(2, len(arise_scores) + 2))

        ax1.plot(levels, arise_scores, "g-", marker="o", label="ARISE")
        ax1.plot(levels, traditional_scores, "b-", marker="s", label="Traditional")
        ax1.set_xlabel("Scaling Level")
        ax1.set_ylabel("Metric Score")
        ax1.set_title("Metric Progression with Scaling")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot comparing metrics
        ax2.scatter(traditional_scores, arise_scores, alpha=0.6, s=50)
        ax2.set_xlabel("Traditional Scaling Metric")
        ax2.set_ylabel("ARISE Score")
        ax2.set_title("ARISE vs Traditional Metric")

        # Add diagonal reference line
        min_val = min(min(traditional_scores), min(arise_scores))
        max_val = max(max(traditional_scores), max(arise_scores))
        ax2.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.3)

        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
