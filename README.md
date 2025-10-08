# ARISE: Adaptive Resolution-Aware Scaling Evaluation

Official implementation of [**ARISE: An Adaptive Resolution-Aware Metric for Test-Time Scaling Evaluation in Large Reasoning Models**](https://arxiv.org/abs/2510.06014)

## Overview

ARISE is a novel metric designed to systematically evaluate test-time scaling capabilities of large reasoning models. Unlike traditional metrics, ARISE incorporates:

- **Sample-level awareness**: Tracks individual sample trajectories across scaling levels
- **Negative scaling correction**: Properly penalizes performance degradation  
- **Adaptive sampling**: Dynamically adjusts evaluation runs based on variance

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
openai>=1.0.0
vllm>=0.2.0
tqdm>=4.62.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## Repository Structure

```
arise/
├── code/
│   ├── arise_metric.py      # Core ARISE metric implementation
│   ├── adaptive_sampling.py # Adaptive sampling mechanism
│   ├── evaluate.py          # Main evaluation script
│   ├── models.py            # Model wrappers for API calls
│   ├── scaling_metric.py    # Traditional scaling metric
│   └── utils.py            # Utility functions
├── data/
│   ├── AIME.jsonl          # AIME dataset
│   └── HMMT.jsonl          # HMMT dataset
├── configs/
│   └── default_config.yaml # Default configuration
├── results/                # Evaluation results (generated)
└── scripts/
    └── run_experiments.sh   # Experiment runner script
```

## Quick Start

### 1. Evaluate a Model with ARISE

```python
from code.evaluate import ARISEEvaluator

# Initialize evaluator
evaluator = ARISEEvaluator(
    model_name="gpt-4",
    dataset_path="data/AIME.jsonl",
    adaptive_sampling=True
)

# Run evaluation
results = evaluator.evaluate(
    scaling_iterations=3,
    m_min=3,
    m_max=10,
    threshold=0.5
)

print(f"ARISE Score: {results['arise_score']:.4f}")
```

### 2. Compare Multiple Models

```bash
python code/evaluate.py \
    --models gpt-4 claude-3-opus deepseek-r1 \
    --dataset data/AIME.jsonl \
    --output_dir results/ \
    --adaptive_sampling \
    --scaling_iterations 3
```

### 3. Use with vLLM for Open-Source Models

```bash
# Start vLLM server
vllm serve Qwen/Qwen3-32B --max_model_len 32768 \
    --enable-reasoning --reasoning-parser qwen3

# Run evaluation
python code/evaluate.py \
    --models vllm:Qwen3-32B \
    --vllm_base_url http://localhost:8000 \
    --dataset data/HMMT.jsonl
```

## ARISE Metric Details

The ARISE score for a sample is computed as:

```
ARISE_i = Σ_j Δa_i^(j) · W_i^(j)
```

Where:
- `Δa_i^(j)` = accuracy change between iterations
- `W_i^(j)` = token ratio weight function

Key features:
- **Positive scaling**: Rewards efficient token usage for improvements
- **Negative scaling**: Penalizes wasteful computation for degradations
- **Sample-level tracking**: Captures individual trajectory patterns

## Adaptive Sampling

The adaptive sampling mechanism dynamically allocates evaluation runs based on variance:

```python
# Configure adaptive sampling
config = {
    'm_min': 3,        # Minimum samples
    'm_max': 10,       # Maximum samples  
    'threshold': 0.5,  # Convergence threshold
    'epsilon': 1e-8    # Numerical stability
}
```

## Supported Models

### API-based Models
- OpenAI: GPT-4, GPT-5, o1, o3
- Anthropic: Claude Opus 4.1, Claude Sonnet 4
- DeepSeek: DeepSeek-Reasoner, DeepSeek-Chat

### Open-Source Models (via vLLM)
- Qwen3 series (0.6B - 235B)
- Custom models with vLLM support

## Datasets

The repository includes two mathematical reasoning datasets:

- **AIME.jsonl**: American Invitational Mathematics Examination problems
- **HMMT.jsonl**: Harvard-MIT Mathematics Tournament problems

Dataset format:
```json
{
  "question": "Problem statement...",
  "answer": "Numerical answer"
}
```

## Citation

```bibtex
@misc{yin2025ariseadaptiveresolutionawaremetric,
      title={ARISE: An Adaptive Resolution-Aware Metric for Test-Time Scaling Evaluation in Large Reasoning Models}, 
      author={Zhangyue Yin and Qiushi Sun and Zhiyuan Zeng and Zhiyuan Yu and Qipeng Guo and Xuanjing Huang and Xipeng Qiu},
      year={2025},
      eprint={2510.06014},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.06014}, 
}
```
