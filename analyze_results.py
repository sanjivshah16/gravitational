"""
Results Analysis and Visualization for Attention Mechanism Comparison

This script runs the complete analysis and visualization pipeline for comparing
different attention mechanisms on LRA benchmark tasks.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import the evaluator
from evaluate_performance import AttentionModelEvaluator


def analyze_results(results_dir):
    """
    Analyze and visualize results from all models and tasks.
    
    Args:
        results_dir: Directory containing results
    
    Returns:
        Summary DataFrame and path to output directory
    """
    print(f"Analyzing results from {results_dir}...")
    
    # Create evaluator
    evaluator = AttentionModelEvaluator(results_dir)
    
    # Create output directory for analysis
    output_dir = os.path.join(results_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation and visualization
    summary_df = evaluator.evaluate_and_visualize(output_dir)
    
    # Save summary as markdown table
    with open(os.path.join(output_dir, 'summary_table.md'), 'w') as f:
        f.write("# Summary of Results\n\n")
        f.write(summary_df.to_markdown(index=False))
    
    # Generate additional comparative analysis
    generate_comparative_analysis(evaluator, output_dir)
    
    return summary_df, output_dir


def generate_comparative_analysis(evaluator, output_dir):
    """
    Generate additional comparative analysis beyond the basic metrics.
    
    Args:
        evaluator: AttentionModelEvaluator instance
        output_dir: Directory to save analysis outputs
    """
    # Calculate relative improvements
    for task in evaluator.tasks:
        if task not in evaluator.results:
            continue
        
        # Get baseline (conventional) accuracy
        if 'conventional' not in evaluator.results[task]:
            continue
            
        baseline_acc = evaluator.results[task]['conventional']['final_accuracy']
        
        # Calculate improvements for other models
        improvements = {}
        for model_name in evaluator.model_names:
            if model_name == 'conventional' or model_name not in evaluator.results[task]:
                continue
                
            model_acc = evaluator.results[task][model_name]['final_accuracy']
            rel_improvement = (model_acc - baseline_acc) / baseline_acc * 100
            improvements[model_name] = rel_improvement
        
        # Create and save improvement plot
        plt.figure(figsize=(10, 6))
        models = list(improvements.keys())
        values = list(improvements.values())
        
        bars = plt.bar(
            [evaluator.model_display_names[m] for m in models],
            values,
            color=[evaluator.colors[m] for m in models]
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f"{height:.1f}%",
                ha='center', va='bottom'
            )
        
        plt.title(f'Accuracy Improvement over Conventional Attention - {task.upper()}', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Relative Improvement (%)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{task}_relative_improvement.png'), dpi=300, bbox_inches='tight')
    
    # Create radar chart comparing all models across key metrics
    create_radar_chart(evaluator, output_dir)


def create_radar_chart(evaluator, output_dir):
    """
    Create radar charts comparing models across multiple metrics.
    
    Args:
        evaluator: AttentionModelEvaluator instance
        output_dir: Directory to save analysis outputs
    """
    for task in evaluator.tasks:
        if task not in evaluator.results:
            continue
            
        # Check if we have all models for this task
        models_present = [m for m in evaluator.model_names if m in evaluator.results[task]]
        if len(models_present) < 2:
            continue
        
        # Define metrics to compare (normalize each to 0-1 scale)
        metrics = {
            'Accuracy': [],
            'Speed': [],  # Inverse of convergence epoch
            'Efficiency': [],  # Inverse of training time
            'Memory': []  # Inverse of memory usage
        }
        
        # Extract raw values for normalization
        raw_values = {
            'Accuracy': [],
            'Convergence': [],
            'Time': [],
            'Memory': []
        }
        
        for model_name in models_present:
            results = evaluator.results[task][model_name]
            raw_values['Accuracy'].append(results['final_accuracy'])
            raw_values['Convergence'].append(results['convergence_epoch'])
            raw_values['Time'].append(np.mean(results['training_times']))
            raw_values['Memory'].append(np.max(results['peak_memory_usage']))
        
        # Normalize values
        for model_name in models_present:
            results = evaluator.results[task][model_name]
            
            # Accuracy (higher is better)
            acc = results['final_accuracy']
            acc_norm = (acc - min(raw_values['Accuracy'])) / (max(raw_values['Accuracy']) - min(raw_values['Accuracy']) + 1e-10)
            metrics['Accuracy'].append(acc_norm)
            
            # Speed - inverse of convergence epoch (lower epoch is better)
            conv = results['convergence_epoch']
            # Invert and normalize
            speed_norm = 1 - (conv - min(raw_values['Convergence'])) / (max(raw_values['Convergence']) - min(raw_values['Convergence']) + 1e-10)
            metrics['Speed'].append(speed_norm)
            
            # Efficiency - inverse of training time (lower time is better)
            time = np.mean(results['training_times'])
            # Invert and normalize
            eff_norm = 1 - (time - min(raw_values['Time'])) / (max(raw_values['Time']) - min(raw_values['Time']) + 1e-10)
            metrics['Efficiency'].append(eff_norm)
            
            # Memory - inverse of memory usage (lower memory is better)
            mem = np.max(results['peak_memory_usage'])
            # Invert and normalize
            mem_norm = 1 - (mem - min(raw_values['Memory'])) / (max(raw_values['Memory']) - min(raw_values['Memory']) + 1e-10)
            metrics['Memory'].append(mem_norm)
        
        # Create radar chart
        labels = list(metrics.keys())
        num_models = len(models_present)
        
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, model_name in enumerate(models_present):
            values = [metrics[metric][i] for metric in labels]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=evaluator.model_display_names[model_name], 
                   color=evaluator.colors[model_name], marker=evaluator.markers[model_name], markersize=8)
            ax.fill(angles, values, alpha=0.1, color=evaluator.colors[model_name])
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        
        ax.set_ylim(0, 1)
        ax.set_title(f'Model Comparison - {task.upper()}', fontsize=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task}_radar_comparison.png'), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # Get results directory from command line or use default
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find most recent results directory
        results_dirs = [d for d in os.listdir('.') if d.startswith('results_')]
        if results_dirs:
            results_dirs.sort(reverse=True)
            results_dir = results_dirs[0]
        else:
            print("No results directory found. Please specify a directory.")
            sys.exit(1)
    
    # Run analysis
    summary_df, output_dir = analyze_results(results_dir)
    
    # Print summary
    print("\nSummary of Results:")
    print(summary_df)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
