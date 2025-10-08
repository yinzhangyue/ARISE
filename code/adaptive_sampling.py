"""
Adaptive Sampling Module for ARISE
Implements dynamic sampling based on observed variance patterns
"""

import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for adaptive sampling"""

    m_min: int = 3  # Minimum samples
    m_max: int = 10  # Maximum samples
    threshold: float = 0.5  # Convergence threshold (CV)
    epsilon: float = 1e-8  # Numerical stability


class AdaptiveSampler:
    """
    Implements adaptive sampling mechanism for ARISE evaluation
    """

    def __init__(self, config: SamplingConfig = None):
        """
        Initialize adaptive sampler

        Args:
            config: Sampling configuration
        """
        self.config = config or SamplingConfig()

    def calculate_coefficient_variation(self, values: List[float]) -> float:
        """
        Calculate coefficient of variation (CV)

        Args:
            values: List of values

        Returns:
            Coefficient of variation
        """
        if len(values) < 2:
            return 0.0

        mean = np.mean(values)
        std = np.std(values)

        # Avoid division by zero
        cv = std / (mean + self.config.epsilon)
        return cv

    def probe_initial_samples(self, eval_function: Callable, problem: Dict) -> Tuple[List[Dict], Dict]:
        """
        Conduct initial probing phase

        Args:
            eval_function: Function to evaluate a problem
            problem: Problem dictionary

        Returns:
            Tuple of (initial_results, statistics)
        """
        initial_results = []
        accuracies = []
        tokens = []

        # Collect minimum samples
        for _ in range(self.config.m_min):
            result = eval_function(problem)
            initial_results.append(result)
            accuracies.append(result["accuracy"])
            tokens.append(result["tokens"])

        # Calculate initial statistics
        stats = {"accuracy_mean": np.mean(accuracies), "accuracy_std": np.std(accuracies), "accuracy_cv": self.calculate_coefficient_variation(accuracies), "tokens_mean": np.mean(tokens), "tokens_std": np.std(tokens), "tokens_cv": self.calculate_coefficient_variation(tokens), "combined_cv": self.calculate_coefficient_variation(accuracies) + self.calculate_coefficient_variation(tokens)}

        return initial_results, stats

    def adaptive_sample(self, eval_function: Callable, problem: Dict, verbose: bool = False) -> Tuple[Dict, int]:
        """
        Perform adaptive sampling for a single problem

        Args:
            eval_function: Function to evaluate problem
            problem: Problem to evaluate
            verbose: Whether to log detailed information

        Returns:
            Tuple of (final_result, num_samples)
        """
        # Initial probing
        results, stats = self.probe_initial_samples(eval_function, problem)

        accuracies = [r["accuracy"] for r in results]
        tokens = [r["tokens"] for r in results]

        num_samples = self.config.m_min
        combined_cv = stats["combined_cv"]

        # Continue sampling if variance is high
        while num_samples < self.config.m_max and combined_cv > self.config.threshold:
            # Get additional sample
            result = eval_function(problem)
            results.append(result)
            accuracies.append(result["accuracy"])
            tokens.append(result["tokens"])
            num_samples += 1

            # Recalculate CV
            acc_cv = self.calculate_coefficient_variation(accuracies)
            tok_cv = self.calculate_coefficient_variation(tokens)
            combined_cv = acc_cv + tok_cv

            if verbose:
                logger.info(f"Sample {num_samples}: CV={combined_cv:.3f} " f"(acc_cv={acc_cv:.3f}, tok_cv={tok_cv:.3f})")

        # Compute final statistics
        final_result = {"accuracy": np.mean(accuracies), "tokens": np.mean(tokens), "accuracy_std": np.std(accuracies), "tokens_std": np.std(tokens), "num_samples": num_samples, "converged": combined_cv <= self.config.threshold, "final_cv": combined_cv}

        return final_result, num_samples

    def allocate_budget(self, initial_cvs: Dict[Tuple[int, int], float], total_budget: int) -> Dict[Tuple[int, int], int]:
        """
        Allocate sampling budget proportionally to variance

        Args:
            initial_cvs: Dictionary mapping (sample_idx, iteration) to CV
            total_budget: Total sampling budget

        Returns:
            Dictionary mapping (sample_idx, iteration) to allocated samples
        """
        n_configurations = len(initial_cvs)
        base_budget = n_configurations * self.config.m_min

        if total_budget <= base_budget:
            # Not enough budget for adaptive allocation
            return {key: self.config.m_min for key in initial_cvs}

        remaining_budget = total_budget - base_budget
        total_cv = sum(initial_cvs.values())

        allocations = {}

        if total_cv > 0:
            for key, cv in initial_cvs.items():
                # Proportional allocation based on CV
                extra_samples = int(remaining_budget * cv / total_cv)
                allocations[key] = min(self.config.m_min + extra_samples, self.config.m_max)
        else:
            # Uniform allocation if no variance
            allocations = {key: self.config.m_min for key in initial_cvs}

        return allocations


class MultiIterationSampler:
    """
    Handles adaptive sampling across multiple scaling iterations
    """

    def __init__(self, config: SamplingConfig = None):
        """
        Initialize multi-iteration sampler

        Args:
            config: Sampling configuration
        """
        self.config = config or SamplingConfig()
        self.sampler = AdaptiveSampler(config)

    def evaluate_with_scaling(self, eval_function: Callable, problems: List[Dict], scaling_levels: List[int], total_budget: int = None) -> List[Dict]:
        """
        Evaluate problems across multiple scaling iterations with adaptive sampling

        Args:
            eval_function: Function to evaluate problem at given scaling level
            problems: List of problems to evaluate
            scaling_levels: List of scaling iterations
            total_budget: Optional total sampling budget

        Returns:
            List of evaluation results
        """
        results = []

        # First pass: collect initial CVs
        initial_cvs = {}

        for i, problem in enumerate(problems):
            problem_results = {"problem_idx": i, "accuracies": [], "tokens": [], "scaling_results": []}

            for j, level in enumerate(scaling_levels):
                # Define evaluation function for this configuration
                def eval_fn(p):
                    return eval_function(p, scaling_level=level)

                # Get initial statistics
                _, stats = self.sampler.probe_initial_samples(eval_fn, problem)
                initial_cvs[(i, j)] = stats["combined_cv"]

            results.append(problem_results)

        # Allocate budget if specified
        if total_budget:
            allocations = self.sampler.allocate_budget(initial_cvs, total_budget)
        else:
            allocations = {key: self.config.m_max for key in initial_cvs}

        # Second pass: perform adaptive sampling with allocated budget
        for i, problem in enumerate(problems):
            for j, level in enumerate(scaling_levels):
                # Get allocation for this configuration
                max_samples = allocations.get((i, j), self.config.m_max)

                # Configure sampler with allocated budget
                temp_config = SamplingConfig(m_min=self.config.m_min, m_max=max_samples, threshold=self.config.threshold, epsilon=self.config.epsilon)
                temp_sampler = AdaptiveSampler(temp_config)

                # Define evaluation function
                def eval_fn(p):
                    return eval_function(p, scaling_level=level)

                # Perform adaptive sampling
                result, num_samples = temp_sampler.adaptive_sample(eval_fn, problem, verbose=False)

                # Store results
                results[i]["accuracies"].append(result["accuracy"])
                results[i]["tokens"].append(result["tokens"])
                results[i]["scaling_results"].append({"scaling_level": level, "result": result, "num_samples": num_samples})

        return results


class StabilityAnalyzer:
    """
    Analyzes stability of ARISE evaluations
    """

    def __init__(self):
        pass

    def analyze_stability(self, repeated_evaluations: List[List[Dict]]) -> Dict:
        """
        Analyze stability across repeated evaluations

        Args:
            repeated_evaluations: List of evaluation runs

        Returns:
            Stability analysis
        """
        # Extract ARISE scores from each run
        arise_scores = []
        for evaluation in repeated_evaluations:
            scores = [r.get("arise_score", 0) for r in evaluation]
            arise_scores.append(np.mean(scores))

        # Calculate stability metrics
        analysis = {"mean": np.mean(arise_scores), "std": np.std(arise_scores), "cv": np.std(arise_scores) / (np.mean(arise_scores) + 1e-8), "min": np.min(arise_scores), "max": np.max(arise_scores), "range": np.max(arise_scores) - np.min(arise_scores), "num_runs": len(arise_scores), "scores": arise_scores}

        return analysis

    def compare_sampling_strategies(self, single_sample_results: List[Dict], multi_sample_results: List[Dict], adaptive_results: List[Dict]) -> Dict:
        """
        Compare different sampling strategies

        Args:
            single_sample_results: Results from single sampling
            multi_sample_results: Results from fixed multiple sampling
            adaptive_results: Results from adaptive sampling

        Returns:
            Comparison of strategies
        """
        strategies = {"single": single_sample_results, "multi": multi_sample_results, "adaptive": adaptive_results}

        comparison = {}

        for name, results in strategies.items():
            # Analyze each strategy
            stability = self.analyze_stability([results])
            comparison[name] = {"stability": stability, "total_samples": sum(r.get("num_samples", 1) for r in results), "avg_samples_per_problem": np.mean([r.get("num_samples", 1) for r in results])}

        # Calculate improvement metrics
        if "single" in comparison and "adaptive" in comparison:
            single_cv = comparison["single"]["stability"]["cv"]
            adaptive_cv = comparison["adaptive"]["stability"]["cv"]

            comparison["variance_reduction"] = {"absolute": single_cv - adaptive_cv, "relative": (single_cv - adaptive_cv) / single_cv * 100 if single_cv > 0 else 0}

        return comparison
