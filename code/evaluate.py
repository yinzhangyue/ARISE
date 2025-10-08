"""
Main evaluation script for ARISE
Coordinates model evaluation, metric calculation, and result analysis
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from arise_metric import ARISEMetric, NegativeScalingAnalyzer
from adaptive_sampling import AdaptiveSampler, SamplingConfig, MultiIterationSampler
from models import ModelWrapper, get_model_wrapper
from utils import load_dataset, save_results, create_output_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARISEEvaluator:
    """
    Main evaluator class for ARISE experiments
    """

    def __init__(self, model_name: str, dataset_path: str, adaptive_sampling: bool = True, sampling_config: Optional[SamplingConfig] = None):
        """
        Initialize ARISE evaluator

        Args:
            model_name: Name of model to evaluate
            dataset_path: Path to evaluation dataset
            adaptive_sampling: Whether to use adaptive sampling
            sampling_config: Configuration for adaptive sampling
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.adaptive_sampling = adaptive_sampling

        # Initialize components
        self.model = get_model_wrapper(model_name)
        self.arise_metric = ARISEMetric()
        self.negative_analyzer = NegativeScalingAnalyzer()

        # Setup sampling
        self.sampling_config = sampling_config or SamplingConfig()
        if adaptive_sampling:
            self.sampler = AdaptiveSampler(self.sampling_config)
            self.multi_sampler = MultiIterationSampler(self.sampling_config)
        else:
            self.sampler = None
            self.multi_sampler = None

        # Load dataset
        self.dataset = load_dataset(dataset_path)

    def evaluate_single_problem(self, problem: Dict, scaling_level: int = 1, temperature: float = 0.7) -> Dict:
        """
        Evaluate a single problem at specified scaling level

        Args:
            problem: Problem dictionary with 'question' and 'answer'
            scaling_level: Scaling iteration (1=baseline, 2+=scaled)
            temperature: Sampling temperature

        Returns:
            Evaluation result dictionary
        """
        # Generate response
        response = self.model.generate(prompt=problem["question"], scaling_level=scaling_level, temperature=temperature)

        # Check correctness
        predicted_answer = self.model.extract_answer(response["text"])
        correct = self.check_answer(predicted_answer, problem["answer"])

        result = {"question": problem["question"], "true_answer": problem["answer"], "predicted_answer": predicted_answer, "accuracy": 1 if correct else 0, "tokens": response["total_tokens"], "completion_tokens": response["completion_tokens"], "scaling_level": scaling_level, "response_text": response["text"]}

        return result

    def check_answer(self, predicted: str, true: str) -> bool:
        """
        Check if predicted answer matches true answer

        Args:
            predicted: Predicted answer
            true: True answer

        Returns:
            Whether answers match
        """
        # Normalize answers
        predicted = str(predicted).strip().lower()
        true = str(true).strip().lower()

        # Direct match
        if predicted == true:
            return True

        # Try numeric comparison
        try:
            pred_num = float(predicted.replace(",", ""))
            true_num = float(true.replace(",", ""))
            return abs(pred_num - true_num) < 1e-6
        except:
            pass

        return False

    def evaluate_with_scaling(self, scaling_iterations: int = 3, max_problems: Optional[int] = None, temperature: float = 0.7) -> Dict:
        """
        Evaluate model across multiple scaling iterations

        Args:
            scaling_iterations: Number of scaling levels to evaluate
            max_problems: Maximum number of problems to evaluate
            temperature: Sampling temperature

        Returns:
            Complete evaluation results
        """
        problems = self.dataset[:max_problems] if max_problems else self.dataset
        scaling_levels = list(range(1, scaling_iterations + 1))

        logger.info(f"Evaluating {len(problems)} problems across {scaling_iterations} scaling levels")

        all_results = []

        for problem in tqdm(problems, desc="Evaluating problems"):
            problem_results = {"question": problem["question"], "answer": problem["answer"], "accuracies": [], "tokens": [], "scaling_results": []}

            for level in scaling_levels:
                if self.adaptive_sampling:
                    # Use adaptive sampling
                    def eval_fn(p):
                        return self.evaluate_single_problem(p, level, temperature)

                    result, num_samples = self.sampler.adaptive_sample(eval_fn, problem, verbose=False)

                    problem_results["accuracies"].append(result["accuracy"])
                    problem_results["tokens"].append(result["tokens"])
                    problem_results["scaling_results"].append({"level": level, "accuracy": result["accuracy"], "tokens": result["tokens"], "num_samples": num_samples, "converged": result["converged"]})
                else:
                    # Single evaluation
                    result = self.evaluate_single_problem(problem, level, temperature)
                    problem_results["accuracies"].append(result["accuracy"])
                    problem_results["tokens"].append(result["tokens"])
                    problem_results["scaling_results"].append({"level": level, "accuracy": result["accuracy"], "tokens": result["tokens"], "num_samples": 1})

            # Calculate sample ARISE score
            sample_arise = self.arise_metric.calculate_sample_arise(problem_results["accuracies"], problem_results["tokens"])
            problem_results["arise_score"] = sample_arise

            all_results.append(problem_results)

        # Calculate overall metrics
        overall_arise, arise_stats = self.arise_metric.calculate_dataset_arise(all_results)

        # Analyze negative scaling
        negative_analysis = self.negative_analyzer.analyze_negative_scaling(all_results)

        # Calculate traditional scaling metric for comparison
        baseline_results = [{"accuracy": r["accuracies"][0], "tokens": r["tokens"][0]} for r in all_results]
        final_results = [{"accuracy": r["accuracies"][-1], "tokens": r["tokens"][-1]} for r in all_results]

        from arise_metric import calculate_traditional_scaling_metric

        traditional_metric = calculate_traditional_scaling_metric(baseline_results, final_results)

        # Compile final results
        evaluation_results = {"model": self.model_name, "dataset": self.dataset_path, "num_problems": len(problems), "scaling_iterations": scaling_iterations, "adaptive_sampling": self.adaptive_sampling, "arise_score": overall_arise, "traditional_scaling_metric": traditional_metric, "arise_statistics": arise_stats, "negative_scaling_analysis": negative_analysis, "sample_results": all_results, "accuracy_by_level": [np.mean([r["accuracies"][i] for r in all_results]) for i in range(scaling_iterations)], "tokens_by_level": [np.mean([r["tokens"][i] for r in all_results]) for i in range(scaling_iterations)]}

        return evaluation_results

    def run_stability_analysis(self, num_runs: int = 5, scaling_iterations: int = 3, max_problems: int = 50) -> Dict:
        """
        Run multiple evaluations to assess metric stability

        Args:
            num_runs: Number of independent runs
            scaling_iterations: Number of scaling levels
            max_problems: Number of problems to evaluate

        Returns:
            Stability analysis results
        """
        logger.info(f"Running {num_runs} independent evaluations for stability analysis")

        arise_scores = []
        traditional_scores = []

        for run_idx in range(num_runs):
            logger.info(f"Run {run_idx + 1}/{num_runs}")

            results = self.evaluate_with_scaling(scaling_iterations=scaling_iterations, max_problems=max_problems)

            arise_scores.append(results["arise_score"])
            traditional_scores.append(results["traditional_scaling_metric"])

        stability_results = {"num_runs": num_runs, "arise": {"scores": arise_scores, "mean": np.mean(arise_scores), "std": np.std(arise_scores), "cv": np.std(arise_scores) / (np.mean(arise_scores) + 1e-8)}, "traditional": {"scores": traditional_scores, "mean": np.mean(traditional_scores), "std": np.std(traditional_scores), "cv": np.std(traditional_scores) / (np.mean(traditional_scores) + 1e-8)}}

        return stability_results


def main():
    """
    Main entry point for evaluation script
    """
    parser = argparse.ArgumentParser(description="ARISE Evaluation")

    # Model and dataset arguments
    parser.add_argument("--models", nargs="+", default=["gpt-4"], help="Models to evaluate")
    parser.add_argument("--dataset", type=str, default="data/AIME.jsonl", help="Path to evaluation dataset")

    # Scaling arguments
    parser.add_argument("--scaling_iterations", type=int, default=3, help="Number of scaling iterations")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    # Sampling arguments
    parser.add_argument("--adaptive_sampling", action="store_true", help="Use adaptive sampling")
    parser.add_argument("--m_min", type=int, default=3, help="Minimum samples for adaptive sampling")
    parser.add_argument("--m_max", type=int, default=10, help="Maximum samples for adaptive sampling")
    parser.add_argument("--threshold", type=float, default=0.5, help="Convergence threshold for adaptive sampling")

    # Analysis arguments
    parser.add_argument("--stability_analysis", action="store_true", help="Run stability analysis")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for stability analysis")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/", help="Output directory for results")
    parser.add_argument("--max_problems", type=int, default=None, help="Maximum number of problems to evaluate")

    # vLLM arguments
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000", help="Base URL for vLLM server")

    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_dir(args.output_dir)

    # Setup sampling config
    sampling_config = SamplingConfig(m_min=args.m_min, m_max=args.m_max, threshold=args.threshold)

    # Evaluate each model
    all_results = {}

    for model_name in args.models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating {model_name}")
        logger.info(f"{'='*50}")

        evaluator = ARISEEvaluator(model_name=model_name, dataset_path=args.dataset, adaptive_sampling=args.adaptive_sampling, sampling_config=sampling_config)

        if args.stability_analysis:
            # Run stability analysis
            results = evaluator.run_stability_analysis(num_runs=args.num_runs, scaling_iterations=args.scaling_iterations, max_problems=args.max_problems or 50)
        else:
            # Single evaluation
            results = evaluator.evaluate_with_scaling(scaling_iterations=args.scaling_iterations, max_problems=args.max_problems, temperature=args.temperature)

        all_results[model_name] = results

        # Save individual model results
        save_results(results, output_dir / f"{model_name.replace('/', '-')}_results.json")

        # Print summary
        if args.stability_analysis:
            print(f"\n{model_name} Stability Analysis:")
            print(f"  ARISE: {results['arise']['mean']:.4f} ± {results['arise']['std']:.4f}")
            print(f"  Traditional: {results['traditional']['mean']:.4f} ± {results['traditional']['std']:.4f}")
            print(f"  ARISE CV: {results['arise']['cv']:.3f}")
            print(f"  Traditional CV: {results['traditional']['cv']:.3f}")
        else:
            print(f"\n{model_name} Results:")
            print(f"  ARISE Score: {results['arise_score']:.4f}")
            print(f"  Traditional Scaling: {results['traditional_scaling_metric']:.6f}")
            print(f"  Negative Scaling Samples: {results['negative_scaling_analysis']['num_negative_samples']}")
            print(f"  Accuracy by Level: {[f'{acc:.2%}' for acc in results['accuracy_by_level']]}")
            print(f"  Tokens by Level: {[f'{tok:.0f}' for tok in results['tokens_by_level']]}")

    # Save combined results
    save_results(all_results, output_dir / "all_results.json")

    # Generate comparison table
    if len(args.models) > 1:
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)

        comparison_data = []
        for model, results in all_results.items():
            if args.stability_analysis:
                comparison_data.append({"Model": model, "ARISE Mean": results["arise"]["mean"], "ARISE Std": results["arise"]["std"], "Traditional Mean": results["traditional"]["mean"], "Traditional Std": results["traditional"]["std"]})
            else:
                comparison_data.append({"Model": model, "ARISE Score": results["arise_score"], "Traditional Scaling": results["traditional_scaling_metric"], "Negative Samples": results["negative_scaling_analysis"]["num_negative_samples"]})

        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        df.to_csv(output_dir / "comparison.csv", index=False)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
