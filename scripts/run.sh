#!/bin/bash

# ARISE Experiment Runner Script
# Usage: ./run_experiments.sh [experiment_type] [dataset]

set -e  # Exit on error

# Default values
EXPERIMENT_TYPE=${1:-"basic"}
DATASET=${2:-"data/AIME.jsonl"}
OUTPUT_DIR="results/"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}ARISE Evaluation Framework${NC}"
echo "================================"
echo ""

# Function to run basic evaluation
run_basic_evaluation() {
    echo -e "${YELLOW}Running Basic Evaluation${NC}"
    echo "Dataset: $DATASET"
    echo ""
    
    python code/evaluate.py \
        --models gpt-4 claude-opus-4-1 \
        --dataset "$DATASET" \
        --scaling_iterations 3 \
        --adaptive_sampling \
        --m_min 3 \
        --m_max 10 \
        --threshold 0.5 \
        --output_dir "$OUTPUT_DIR"
}

# Function to run comprehensive evaluation
run_comprehensive_evaluation() {
    echo -e "${YELLOW}Running Comprehensive Evaluation${NC}"
    echo "Dataset: $DATASET"
    echo ""
    
    python code/evaluate.py \
        --models gpt-4 gpt-5 o1 o3 \
            claude-opus-4-1 claude-opus-4 claude-sonnet-4 \
            deepseek-r1 deepseek-chat \
        --dataset "$DATASET" \
        --scaling_iterations 4 \
        --adaptive_sampling \
        --m_min 3 \
        --m_max 10 \
        --threshold 0.5 \
        --max_problems 100 \
        --output_dir "$OUTPUT_DIR"
}

# Function to run stability analysis
run_stability_analysis() {
    echo -e "${YELLOW}Running Stability Analysis${NC}"
    echo "Dataset: $DATASET"
    echo ""
    
    python code/evaluate.py \
        --models gpt-5 claude-opus-4-1 \
        --dataset "$DATASET" \
        --scaling_iterations 3 \
        --stability_analysis \
        --num_runs 5 \
        --adaptive_sampling \
        --max_problems 50 \
        --output_dir "$OUTPUT_DIR"
}

# Function to run vLLM evaluation
run_vllm_evaluation() {
    echo -e "${YELLOW}Running vLLM Model Evaluation${NC}"
    echo "Dataset: $DATASET"
    echo ""
    
    # Check if vLLM server is running
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}vLLM server is running${NC}"
    else
        echo -e "${RED}Error: vLLM server is not running${NC}"
        echo "Please start vLLM server with:"
        echo "vllm serve MODEL_NAME --max_model_len 32768 --enable-reasoning --reasoning-parser qwen3"
        exit 1
    fi
    
    python code/evaluate.py \
        --models vllm:Qwen3-32B vllm:Qwen3-235B-A22B \
        --dataset "$DATASET" \
        --scaling_iterations 3 \
        --adaptive_sampling \
        --vllm_base_url http://localhost:8000 \
        --output_dir "$OUTPUT_DIR"
}

# Function to run quick test
run_quick_test() {
    echo -e "${YELLOW}Running Quick Test${NC}"
    echo "Dataset: $DATASET"
    echo ""
    
    python code/evaluate.py \
        --models gpt-4 \
        --dataset "$DATASET" \
        --scaling_iterations 2 \
        --max_problems 10 \
        --output_dir "$OUTPUT_DIR"
}

# Function to compare metrics
compare_metrics() {
    echo -e "${YELLOW}Running Metric Comparison${NC}"
    echo ""
    
    python -c "
import sys
sys.path.append('code')
from utils import load_all_results, plot_arise_comparison, export_for_paper
from pathlib import Path

# Load latest results
results_dirs = sorted(Path('$OUTPUT_DIR').iterdir())
if results_dirs:
    latest_dir = results_dirs[-1]
    results = load_all_results(latest_dir)
    
    # Create comparison plots
    plot_arise_comparison(results, 'results/metric_comparison.png')
    
    # Export for paper
    export_for_paper(results, 'paper_results/')
    
    print('Comparison plots saved to results/')
    print('Paper-ready exports saved to paper_results/')
else:
    print('No results found to compare')
"
}

# Function to start vLLM server (example)
start_vllm_server() {
    MODEL_NAME=${1:-"Qwen/Qwen3-32B"}
    echo -e "${YELLOW}Starting vLLM server with $MODEL_NAME${NC}"
    
    vllm serve "$MODEL_NAME" \
        --max_model_len 32768 \
        --enable-reasoning \
        --reasoning-parser qwen3 &
    
    # Wait for server to start
    echo "Waiting for vLLM server to start..."
    sleep 10
    
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}vLLM server started successfully${NC}"
    else
        echo -e "${RED}Failed to start vLLM server${NC}"
        exit 1
    fi
}

# Function to generate report
generate_report() {
    echo -e "${YELLOW}Generating Evaluation Report${NC}"
    echo ""
    
    python -c "
import sys
sys.path.append('code')
from utils import load_all_results, create_comparison_dataframe
from pathlib import Path
import pandas as pd

# Load latest results
results_dirs = sorted(Path('$OUTPUT_DIR').iterdir())
if results_dirs:
    latest_dir = results_dirs[-1]
    results = load_all_results(latest_dir)
    
    # Create comparison
    df = create_comparison_dataframe(results)
    
    print('='*60)
    print('ARISE Evaluation Report')
    print('='*60)
    print()
    print(df.to_string(index=False))
    print()
    
    # Save report
    df.to_csv('results/evaluation_report.csv', index=False)
    print('Full report saved to results/evaluation_report.csv')
else:
    print('No results found')
"
}

# Main execution
case "$EXPERIMENT_TYPE" in
    basic)
        run_basic_evaluation
        ;;
    comprehensive)
        run_comprehensive_evaluation
        ;;
    stability)
        run_stability_analysis
        ;;
    vllm)
        run_vllm_evaluation
        ;;
    quick)
        run_quick_test
        ;;
    compare)
        compare_metrics
        ;;
    report)
        generate_report
        ;;
    start-vllm)
        start_vllm_server "$2"
        ;;
    *)
        echo "Usage: $0 [experiment_type] [dataset]"
        echo ""
        echo "Experiment types:"
        echo "  basic         - Basic evaluation with 2 models"
        echo "  comprehensive - Full evaluation with all models"
        echo "  stability     - Run stability analysis"
        echo "  vllm         - Evaluate vLLM models"
        echo "  quick        - Quick test with 10 problems"
        echo "  compare      - Compare metrics and generate plots"
        echo "  report       - Generate evaluation report"
        echo "  start-vllm   - Start vLLM server (provide model name)"
        echo ""
        echo "Datasets:"
        echo "  data/AIME.jsonl - AIME mathematics problems (default)"
        echo "  data/HMMT.jsonl - HMMT mathematics problems"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Experiment completed successfully!${NC}"
echo "Results saved to: $OUTPUT_DIR"