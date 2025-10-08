"""
ARISE: Adaptive Resolution-Aware Scaling Evaluation Metric
Core implementation of the ARISE metric for test-time scaling evaluation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class ARISEMetric:
    """
    ARISE metric implementation for test-time scaling evaluation
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize ARISE metric calculator

        Args:
            epsilon: Small value for numerical stability
        """
        self.epsilon = epsilon

    def calculate_sample_arise(self, accuracies: List[int], tokens: List[int]) -> float:
        """
        Calculate ARISE score for a single sample across scaling iterations

        Args:
            accuracies: Binary accuracy values [0 or 1] at each iteration
            tokens: Token consumption at each iteration

        Returns:
            ARISE score for the sample
        """
        if len(accuracies) != len(tokens):
            raise ValueError("Accuracies and tokens must have same length")

        if len(accuracies) < 2:
            return 0.0

        arise_score = 0.0

        for j in range(1, len(accuracies)):
            # Calculate accuracy change
            delta_a = accuracies[j] - accuracies[j - 1]

            if delta_a == 0:
                # No change in accuracy, no contribution
                continue

            # Calculate token ratio
            if tokens[j] <= 0 or tokens[j - 1] <= 0:
                continue

            token_ratio = tokens[j - 1] / tokens[j]

            # Apply weight function with sign-based exponent
            sign = np.sign(delta_a)
            weight = token_ratio**sign

            # Calculate contribution
            contribution = delta_a * weight
            arise_score += contribution

        return arise_score

    def calculate_dataset_arise(self, sample_results: List[Dict]) -> Tuple[float, Dict]:
        """
        Calculate ARISE score for entire dataset

        Args:
            sample_results: List of dicts with 'accuracies' and 'tokens' for each sample

        Returns:
            Tuple of (overall_arise_score, detailed_statistics)
        """
        sample_scores = []
        transition_stats = {"improved": 0, "degraded": 0, "always_correct": 0, "always_incorrect": 0, "mixed": 0}  # 0->1 transitions  # 1->0 transitions

        for result in sample_results:
            # Calculate sample ARISE
            score = self.calculate_sample_arise(result["accuracies"], result["tokens"])
            sample_scores.append(score)

            # Track transition patterns
            accs = result["accuracies"]
            if all(a == 1 for a in accs):
                transition_stats["always_correct"] += 1
            elif all(a == 0 for a in accs):
                transition_stats["always_incorrect"] += 1
            else:
                # Check for improvements and degradations
                improved = False
                degraded = False
                for j in range(1, len(accs)):
                    if accs[j] > accs[j - 1]:
                        improved = True
                        transition_stats["improved"] += 1
                    elif accs[j] < accs[j - 1]:
                        degraded = True
                        transition_stats["degraded"] += 1

                if improved or degraded:
                    transition_stats["mixed"] += 1

        # Calculate overall ARISE
        overall_arise = np.mean(sample_scores) if sample_scores else 0.0

        # Compile statistics
        statistics = {"overall_arise": overall_arise, "sample_scores": sample_scores, "mean": overall_arise, "std": np.std(sample_scores) if sample_scores else 0.0, "min": np.min(sample_scores) if sample_scores else 0.0, "max": np.max(sample_scores) if sample_scores else 0.0, "num_samples": len(sample_scores), "transition_stats": transition_stats}

        return overall_arise, statistics

    def compare_scaling_approaches(self, baseline_results: List[Dict], scaled_results: List[Dict]) -> Dict:
        """
        Compare two scaling approaches using ARISE

        Args:
            baseline_results: Results from baseline approach
            scaled_results: Results from scaled approach

        Returns:
            Comparison statistics
        """
        # Ensure paired samples
        if len(baseline_results) != len(scaled_results):
            raise ValueError("Results must have same number of samples")

        # Combine results for ARISE calculation
        combined_results = []
        for base, scaled in zip(baseline_results, scaled_results):
            combined = {"accuracies": [base["accuracy"], scaled["accuracy"]], "tokens": [base["tokens"], scaled["tokens"]]}
            combined_results.append(combined)

        arise_score, stats = self.calculate_dataset_arise(combined_results)

        # Calculate additional comparison metrics
        accuracy_improvement = np.mean([scaled["accuracy"] - base["accuracy"] for base, scaled in zip(baseline_results, scaled_results)])

        token_increase = np.mean([scaled["tokens"] - base["tokens"] for base, scaled in zip(baseline_results, scaled_results)])

        comparison = {"arise_score": arise_score, "accuracy_improvement": accuracy_improvement, "token_increase": token_increase, "arise_statistics": stats}

        return comparison


class NegativeScalingAnalyzer:
    """
    Analyzer for negative scaling behaviors
    """

    def __init__(self):
        self.arise_metric = ARISEMetric()

    def analyze_negative_scaling(self, sample_results: List[Dict]) -> Dict:
        """
        Analyze samples exhibiting negative scaling

        Args:
            sample_results: Sample evaluation results

        Returns:
            Analysis of negative scaling patterns
        """
        negative_samples = []
        severe_degradations = []

        for i, result in enumerate(sample_results):
            accs = result["accuracies"]
            tokens = result["tokens"]

            # Check for any degradation
            for j in range(1, len(accs)):
                if accs[j] < accs[j - 1]:
                    # Found degradation
                    token_ratio = tokens[j] / tokens[j - 1]
                    penalty = -token_ratio  # Negative ARISE contribution

                    negative_samples.append({"sample_idx": i, "iteration": j, "accuracy_drop": accs[j - 1] - accs[j], "token_waste": tokens[j] - tokens[j - 1], "token_ratio": token_ratio, "arise_penalty": penalty})

                    # Flag severe degradations (>50% token increase)
                    if token_ratio > 1.5:
                        severe_degradations.append(i)

        analysis = {"num_negative_samples": len(set(s["sample_idx"] for s in negative_samples)), "total_degradations": len(negative_samples), "severe_degradations": len(severe_degradations), "avg_token_waste": np.mean([s["token_waste"] for s in negative_samples]) if negative_samples else 0, "avg_arise_penalty": np.mean([s["arise_penalty"] for s in negative_samples]) if negative_samples else 0, "negative_samples": negative_samples}

        return analysis


def calculate_traditional_scaling_metric(baseline_results: List[Dict], scaled_results: List[Dict]) -> float:
    """
    Calculate traditional slope-based scaling metric for comparison

    Args:
        baseline_results: Baseline evaluation results
        scaled_results: Scaled evaluation results

    Returns:
        Traditional scaling metric value
    """
    # Calculate aggregate statistics
    baseline_acc = np.mean([r["accuracy"] for r in baseline_results])
    scaled_acc = np.mean([r["accuracy"] for r in scaled_results])

    baseline_tokens = np.mean([r["tokens"] for r in baseline_results])
    scaled_tokens = np.mean([r["tokens"] for r in scaled_results])

    # Calculate slope
    if scaled_tokens > baseline_tokens:
        scaling_metric = (scaled_acc - baseline_acc) / (scaled_tokens - baseline_tokens)
    else:
        scaling_metric = 0.0

    return scaling_metric
