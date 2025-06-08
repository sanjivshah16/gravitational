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


def run_complete_experiment():
    """Run the complete experiment pipeline."""
    print("🚀 Starting attention mechanism comparison experiment...")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Choose device
    device = select_device()

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Base model configuration
    base_model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512,
        'max_seq_length': 1024,  # Adjust if needed for ListOps/IMDB
        'dropout': 0.1
    }

    # Specific model configs (4 attention types)
    model_configs = {
        'conventional': base_model_config.copy(),
        'gravitational': base_model_config.copy(),
        'multi_timescale': {**base_model_config.copy(), 'n_timescales': 3},
        'gravitational_multi_timescale': {**base_model_config.copy(), 'n_timescales': 3}
    }

    # Training configuration
    training_config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 20,
        'max_grad_norm': 1.0,
        'subset_size': 1000  # Use None for full dataset
    }

    # Save configurations to file
    with open(os.path.join(results_dir, 'configs.json'), 'w') as f:
        json.dump({
            'model_configs': model_configs,
            'training_config': training_config
        }, f, indent=2)

    # Run tasks
    tasks = ['imdb', 'listops']
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

    # Post-run analysis
    print("\n📊 Analyzing results...")
    summary_df, output_dir = analyze_results(results_dir)

    print("\n📈 Summary of Results:")
    print(summary_df)

    print(f"\n🗂️ Results saved to: {results_dir}")
    print(f"📁 Analysis output in: {output_dir}")

    return results_dir


if __name__ == "__main__":
    run_complete_experiment()
