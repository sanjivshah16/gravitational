"""
Performance Evaluation and Visualization for Attention Mechanism Comparison

This file implements the performance evaluation and visualization functionality
for comparing different attention mechanisms on LRA benchmark tasks.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import pandas as pd
from typing import Dict, List, Any


class AttentionModelEvaluator:
    """Evaluator class for attention mechanism comparison."""
    
    def __init__(self, results_dir):
        """
        Args:
            results_dir: Directory containing results
        """
        self.results_dir = results_dir
        self.tasks = ['imdb', 'listops']
        self.model_names = ['conventional', 'gravitational', 'multi_timescale', 'gravitational_multi_timescale']
        self.model_display_names = {
            'conventional': 'Conventional',
            'gravitational': 'Gravitational',
            'multi_timescale': 'Multi-Timescale',
            'gravitational_multi_timescale': 'Gravitational + Multi-Timescale'
        }
        self.colors = {
            'conventional': '#1f77b4',  # blue
            'gravitational': '#ff7f0e',  # orange
            'multi_timescale': '#2ca02c',  # green
            'gravitational_multi_timescale': '#d62728'  # red
        }
        self.markers = {
            'conventional': 'o',
            'gravitational': 's',
            'multi_timescale': '^',
            'gravitational_multi_timescale': 'D'
        }
        
        # Load results
        self.results = self._load_results()
        
    def _load_results(self):
        """Load results from JSON files."""
        results = {}
        
        for task in self.tasks:
            task_dir = os.path.join(self.results_dir, task)
            if not os.path.exists(task_dir):
                print(f"Warning: Results directory for task {task} not found.")
                continue
                
            results[task] = {}
            
            for model_name in self.model_names:
                result_file = os.path.join(task_dir, f"{model_name}_results.json")
                if not os.path.exists(result_file):
                    print(f"Warning: Results file for {model_name} on {task} not found.")
                    continue
                    
                with open(result_file, 'r') as f:
                    results[task][model_name] = json.load(f)
        
        return results
    
    def evaluate_accuracy(self):
        """Evaluate and compare accuracy across models and tasks."""
        accuracy_data = {
            'Task': [],
            'Model': [],
            'Final Accuracy': []
        }
        
        for task in self.tasks:
            if task not in self.results:
                continue
                
            for model_name in self.model_names:
                if model_name not in self.results[task]:
                    continue
                    
                accuracy_data['Task'].append(task.upper())
                accuracy_data['Model'].append(self.model_display_names[model_name])
                accuracy_data['Final Accuracy'].append(self.results[task][model_name]['final_accuracy'])
        
        return pd.DataFrame(accuracy_data)
    
    def evaluate_convergence(self):
        """Evaluate and compare convergence speed across models and tasks."""
        convergence_data = {
            'Task': [],
            'Model': [],
            'Convergence Epoch': []
        }
        
        for task in self.tasks:
            if task not in self.results:
                continue
                
            for model_name in self.model_names:
                if model_name not in self.results[task]:
                    continue
                    
                convergence_data['Task'].append(task.upper())
                convergence_data['Model'].append(self.model_display_names[model_name])
                convergence_data['Convergence Epoch'].append(self.results[task][model_name]['convergence_epoch'])
        
        return pd.DataFrame(convergence_data)
    
    def evaluate_efficiency(self):
        """Evaluate and compare computational efficiency across models and tasks."""
        efficiency_data = {
            'Task': [],
            'Model': [],
            'Avg Training Time (s/epoch)': [],
            'Peak Memory Usage (GB)': []
        }
        
        for task in self.tasks:
            if task not in self.results:
                continue
                
            for model_name in self.model_names:
                if model_name not in self.results[task]:
                    continue
                    
                avg_time = np.mean(self.results[task][model_name]['training_times'])
                peak_memory = np.max(self.results[task][model_name]['peak_memory_usage'])
                
                efficiency_data['Task'].append(task.upper())
                efficiency_data['Model'].append(self.model_display_names[model_name])
                efficiency_data['Avg Training Time (s/epoch)'].append(avg_time)
                efficiency_data['Peak Memory Usage (GB)'].append(peak_memory)
        
        return pd.DataFrame(efficiency_data)
    
    def plot_accuracy_comparison(self, save_path=None):
        """Plot accuracy comparison across models and tasks."""
        accuracy_df = self.evaluate_accuracy()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Final Accuracy', hue='Task', data=accuracy_df)
        plt.title('Final Accuracy Comparison', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Task')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_convergence_comparison(self, save_path=None):
        """Plot convergence comparison across models and tasks."""
        convergence_df = self.evaluate_convergence()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Convergence Epoch', hue='Task', data=convergence_df)
        plt.title('Convergence Speed Comparison', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Epochs to 90% Max Performance', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Task')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_efficiency_comparison(self, save_path=None):
        """Plot computational efficiency comparison across models and tasks."""
        efficiency_df = self.evaluate_efficiency()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training time plot
        sns.barplot(x='Model', y='Avg Training Time (s/epoch)', hue='Task', data=efficiency_df, ax=axes[0])
        axes[0].set_title('Training Time Comparison', fontsize=16)
        axes[0].set_xlabel('Model', fontsize=14)
        axes[0].set_ylabel('Avg Time per Epoch (s)', fontsize=14)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0].legend(title='Task')
        
        # Memory usage plot
        sns.barplot(x='Model', y='Peak Memory Usage (GB)', hue='Task', data=efficiency_df, ax=axes[1])
        axes[1].set_title('Memory Usage Comparison', fontsize=16)
        axes[1].set_xlabel('Model', fontsize=14)
        axes[1].set_ylabel('Peak Memory Usage (GB)', fontsize=14)
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        axes[1].legend(title='Task')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_learning_curves(self, task, save_path=None):
        """Plot learning curves for all models on a specific task."""
        if task not in self.results:
            print(f"Warning: Results for task {task} not found.")
            return None
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training and validation accuracy
        for model_name in self.model_names:
            if model_name not in self.results[task]:
                continue
                
            model_results = self.results[task][model_name]
            epochs = range(1, len(model_results['train_accuracies']) + 1)
            
            axes[0].plot(epochs, model_results['train_accuracies'], 
                        color=self.colors[model_name], marker=self.markers[model_name],
                        linestyle='-', label=f"{self.model_display_names[model_name]} (Train)")
            axes[0].plot(epochs, model_results['val_accuracies'], 
                        color=self.colors[model_name], marker=self.markers[model_name],
                        linestyle='--', label=f"{self.model_display_names[model_name]} (Val)")
        
        axes[0].set_title(f'Learning Curves - {task.upper()}', fontsize=16)
        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('Accuracy', fontsize=14)
        axes[0].grid(linestyle='--', alpha=0.7)
        axes[0].legend()
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Training and validation loss
        for model_name in self.model_names:
            if model_name not in self.results[task]:
                continue
                
            model_results = self.results[task][model_name]
            epochs = range(1, len(model_results['train_losses']) + 1)
            
            axes[1].plot(epochs, model_results['train_losses'], 
                        color=self.colors[model_name], marker=self.markers[model_name],
                        linestyle='-', label=f"{self.model_display_names[model_name]} (Train)")
            axes[1].plot(epochs, model_results['val_losses'], 
                        color=self.colors[model_name], marker=self.markers[model_name],
                        linestyle='--', label=f"{self.model_display_names[model_name]} (Val)")
        
        axes[1].set_title(f'Loss Curves - {task.upper()}', fontsize=16)
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].set_ylabel('Loss', fontsize=14)
        axes[1].grid(linestyle='--', alpha=0.7)
        axes[1].legend()
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_attention_patterns(self, task, save_path=None):
        """Plot attention patterns for all models on a specific task."""
        if task not in self.results:
            print(f"Warning: Results for task {task} not found.")
            return None
            
        # Load attention patterns
        attention_patterns = {}
        for model_name in self.model_names:
            pattern_file = os.path.join(self.results_dir, task, f"{model_name}_attention_patterns.npy")
            if not os.path.exists(pattern_file):
                print(f"Warning: Attention pattern file for {model_name} on {task} not found.")
                continue
                
            attention_patterns[model_name] = np.load(pattern_file, allow_pickle=True)
        
        # Determine number of models with patterns
        n_models = len(attention_patterns)
        if n_models == 0:
            print("No attention patterns found.")
            return None
            
        # Determine layout based on models and timescales
        has_timescales = any('multi_timescale' in model for model in attention_patterns.keys())
        
        if has_timescales:
            # For models with timescales, we need more subplots
            max_timescales = 3  # Assuming max 3 timescales
            fig, axes = plt.subplots(n_models, max_timescales, figsize=(15, 4 * n_models))
            
            row = 0
            for model_name, patterns in attention_patterns.items():
                if 'multi_timescale' in model_name:
                    # Multi-timescale models have patterns for each timescale
                    for t in range(len(patterns[0])):
                        ax = axes[row, t] if n_models > 1 else axes[t]
                        im = ax.imshow(patterns[0][t], cmap='viridis')
                        ax.set_title(f"{self.model_display_names[model_name]}\nTimescale {t+1}")
                        ax.set_xlabel('Key Position')
                        ax.set_ylabel('Query Position')
                        plt.colorbar(im, ax=ax)
                else:
                    # Single-timescale models have one pattern
                    ax = axes[row, 0] if n_models > 1 else axes[0]
                    im = ax.imshow(patterns[0], cmap='viridis')
                    ax.set_title(f"{self.model_display_names[model_name]}")
                    ax.set_xlabel('Key Position')
                    ax.set_ylabel('Query Position')
                    plt.colorbar(im, ax=ax)
                    
                    # Hide unused subplots
                    for t in range(1, max_timescales):
                        ax = axes[row, t] if n_models > 1 else axes[t]
                        ax.axis('off')
                
                row += 1
        else:
            # For models without timescales, we need one subplot per model
            fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
            
            for i, (model_name, patterns) in enumerate(attention_patterns.items()):
                ax = axes[i] if n_models > 1 else axes
                im = ax.imshow(patterns[0], cmap='viridis')
                ax.set_title(f"{self.model_display_names[model_name]}")
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'Attention Patterns - {task.upper()}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def generate_summary_table(self):
        """Generate a summary table of key metrics for all models and tasks."""
        summary_data = {
            'Task': [],
            'Model': [],
            'Final Accuracy': [],
            'Convergence Epoch': [],
            'Avg Training Time (s/epoch)': [],
            'Peak Memory Usage (GB)': []
        }
        
        for task in self.tasks:
            if task not in self.results:
                continue
                
            for model_name in self.model_names:
                if model_name not in self.results[task]:
                    continue
                    
                model_results = self.results[task][model_name]
                
                summary_data['Task'].append(task.upper())
                summary_data['Model'].append(self.model_display_names[model_name])
                summary_data['Final Accuracy'].append(f"{model_results['final_accuracy']:.4f}")
                summary_data['Convergence Epoch'].append(model_results['convergence_epoch'])
                summary_data['Avg Training Time (s/epoch)'].append(f"{np.mean(model_results['training_times']):.2f}")
                summary_data['Peak Memory Usage (GB)'].append(f"{np.max(model_results['peak_memory_usage']):.2f}")
        
        return pd.DataFrame(summary_data)
    
    def evaluate_and_visualize(self, output_dir):
        """Run all evaluations and save visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save summary table
        summary_df = self.generate_summary_table()
        summary_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
        
        # Plot and save accuracy comparison
        self.plot_accuracy_comparison(save_path=os.path.join(output_dir, 'accuracy_comparison.png'))
        
        # Plot and save convergence comparison
        self.plot_convergence_comparison(save_path=os.path.join(output_dir, 'convergence_comparison.png'))
        
        # Plot and save efficiency comparison
        self.plot_efficiency_comparison(save_path=os.path.join(output_dir, 'efficiency_comparison.png'))
        
        # Plot and save learning curves for each task
        for task in self.tasks:
            if task in self.results:
                self.plot_learning_curves(task, save_path=os.path.join(output_dir, f'{task}_learning_curves.png'))
                
        # Plot and save attention patterns for each task
        for task in self.tasks:
            if task in self.results:
                self.plot_attention_patterns(task, save_path=os.path.join(output_dir, f'{task}_attention_patterns.png'))
        
        return summary_df


if __name__ == "__main__":
    # Set the results directory
    results_dir = "./results_20250608_104500"  # Replace with actual results directory
    
    # Create evaluator
    evaluator = AttentionModelEvaluator(results_dir)
    
    # Run evaluation and visualization
    output_dir = os.path.join(results_dir, 'evaluation')
    summary_df = evaluator.evaluate_and_visualize(output_dir)
    
    # Print summary
    print("\nSummary of Results:")
    print(summary_df)
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")
