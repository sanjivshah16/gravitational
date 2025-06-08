"""
Main experiment runner for attention mechanism comparison

This script orchestrates the complete experiment pipeline:
1. Data preparation
2. Model training and evaluation
3. Results analysis and visualization
"""

import os
import json
import torch
import numpy as np
import warnings
from datetime import datetime
import argparse

# Silence known NumPy 2.0 warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

# Import components
from data_pipeline import create_imdb_dataloaders, create_listops_dataloaders
from train_and_evaluate import run_experiment
from analyze_results import analyze_results


def select_device():
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Metal (MPS) GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (no GPU available)")
    return device


def run_complete_experiment(selected_tasks):
    """Run the complete experiment pipeline for selected tasks."""
    print("🚀 Starting attention mechanism comparison experiment...")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Choose device
    device = select_device()

    # Use fixed results directory so skipping logic works
    results_dir = "./results_current"
    os.makedirs(results_dir, exist_ok=True)

    # Base model configuration
    base_model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512,
        'max_seq_length': 512,
        'dropout': 0.1
    }

    # Specific model configs
    model_configs = {
        'conventional': base_model_config.copy(),
        'gravitational': base_model_config.copy(),
        'multi_timescale': {**base_model_config.copy(), 'n_timescales': 3},
        'gravitational_multi_timescale': {**base_model_config.copy(), 'n_timescales': 3}
    }

    # Training configuration
    training_config = {
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 20,
        'max_grad_norm': 1.0,
        'subset_size': 1000  # Set to None for full dataset
    }

    # Save configurations
    with open(os.path.join(results_dir, 'configs.json'), 'w') as f:
        json.dump({
            'model_configs': model_configs,
            'training_config': training_config
        }, f, indent=2)

    # Determine which tasks to run
    tasks = ['imdb', 'listops'] if selected_tasks == 'all' else [selected_tasks]
    all_results = {}

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"# Running experiment for {task.upper()} task")
        print(f"{'#'*60}\n")

        task_results = run_experiment(
            task_name=task,
            model_configs=model_configs,
            training_config=training_config,
            results_dir=results_dir
        )

        all_results[task] = task_results

    print("\n✅ All experiments completed!")

    # Analyze results
    print("\n📊 Analyzing results...")
    summary_df, output_dir = analyze_results(results_dir)

    print("\n📈 Summary of Results:")
    print(summary_df)

    print(f"\n🗂️ Results saved to: {results_dir}")
    print(f"📁 Analysis output in: {output_dir}")

    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['imdb', 'listops', 'all'], default='all',
                        help="Which task to run: imdb, listops, or all")
    args = parser.parse_args()
    run_complete_experiment(args.task)
