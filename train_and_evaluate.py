"""
Training and Evaluation Script for Attention Mechanism Comparison

This file implements the training, validation, and evaluation functionality
for comparing different attention mechanisms on LRA benchmark tasks.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# Import model implementations
from conventional_attention import create_conventional_model
from gravitational_attention import create_gravitational_model
from multi_timescale_attention import create_multi_timescale_model
from gravitational_multi_timescale_attention import create_gravitational_multi_timescale_model

# Import data pipeline
from data_pipeline import create_imdb_dataloaders, create_listops_dataloaders, create_padding_mask


class AttentionModelTrainer:
    """Trainer class for attention mechanism comparison."""

    def __init__(self, model_name, model, train_dataloader, test_dataloader,
                 device, config, results_dir):
        """
        Args:
            model_name: Name of the model
            model: Model instance
            train_dataloader: Training data loader
            test_dataloader: Test data loader
            device: Device to run on (cuda or cpu)
            config: Configuration dictionary
            results_dir: Directory to save results
        """
        self.model_name = model_name
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.config = config
        self.results_dir = results_dir

        # Create optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Create loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_losses = []
        self.training_times = []
        self.peak_memory_usage = []
        self.attention_patterns = []

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
    
        # Track peak memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
    
        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} Training"):
            # ✅ Unpack batch tuple
            input_ids, labels = batch
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
    
            # Optional: Skip mask unless required
            mask = None
            if hasattr(self.model, "use_mask") and self.model.use_mask:
                mask = create_padding_mask(input_ids, pad_idx=1).to(self.device)
    
            self.optimizer.zero_grad()
    
            # Forward pass
            logits, _ = self.model(input_ids, mask)
    
            # Loss and backward
            loss = self.criterion(logits, labels)
            loss.backward()
    
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
    
            self.optimizer.step()
    
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        # Epoch metrics
        epoch_loss = total_loss / len(self.train_dataloader)
        epoch_accuracy = correct / total
        epoch_time = time.time() - start_time
    
        # Memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        else:
            peak_memory = 0
    
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)
        self.training_times.append(epoch_time)
        self.peak_memory_usage.append(peak_memory)
    
        return epoch_loss, epoch_accuracy, epoch_time, peak_memory


    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc=f"Epoch {epoch+1} Validation"):
                # ✅ Unpack tuple
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
    
                # Optional: only compute mask if model uses it
                mask = None
                if hasattr(self.model, "use_mask") and self.model.use_mask:
                    mask = create_padding_mask(input_ids, pad_idx=1).to(self.device)
    
                # Forward pass
                logits, attention_weights = self.model(input_ids, mask)
    
                # Loss
                loss = self.criterion(logits, labels)
    
                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
                # ✅ Save attention patterns only once at last epoch
                if epoch == self.config['epochs'] - 1 and len(self.attention_patterns) == 0:
                    if isinstance(attention_weights[0], torch.Tensor):
                        attn_pattern = attention_weights[0][0, 0].cpu().numpy()
                    else:
                        attn_pattern = [aw[0, 0, 0].cpu().numpy() for aw in attention_weights]
                    self.attention_patterns.append(attn_pattern)
    
        val_loss = total_loss / len(self.test_dataloader)
        val_accuracy = correct / total
    
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
    
        self.scheduler.step(val_accuracy)
    
        return val_loss, val_accuracy


    def train(self):
        """Train the model for the specified number of epochs."""
        print(f"Starting training for {self.model_name}...")

        for epoch in range(self.config['epochs']):
            # Train for one epoch
            train_loss, train_acc, epoch_time, peak_memory = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate(epoch)

            # Print metrics
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Time: {epoch_time:.2f}s, Peak Memory: {peak_memory:.2f}GB")

            # Early stopping
            if epoch > 10 and val_acc < self.val_accuracies[-10]:
                print("Early stopping triggered")
                break

        # Save results
        self.save_results()

        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'training_times': self.training_times,
            'peak_memory_usage': self.peak_memory_usage,
            'attention_patterns': self.attention_patterns,
            'final_accuracy': self.val_accuracies[-1],
            'convergence_epoch': self.get_convergence_epoch()
        }

    def get_convergence_epoch(self):
        """Get the epoch at which the model reached 90% of its maximum performance."""
        max_acc = max(self.val_accuracies)
        threshold = 0.9 * max_acc

        for i, acc in enumerate(self.val_accuracies):
            if acc >= threshold:
                return i

        return len(self.val_accuracies) - 1

    def save_results(self):
        """Save training results and model."""
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        # Save metrics
        results = {
            'model_name': self.model_name,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'training_times': self.training_times,
            'peak_memory_usage': self.peak_memory_usage,
            'final_accuracy': self.val_accuracies[-1],
            'convergence_epoch': self.get_convergence_epoch(),
            'config': self.config
        }

        with open(os.path.join(self.results_dir, f"{self.model_name}_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        # Save model
        torch.save(self.model.state_dict(), os.path.join(self.results_dir, f"{self.model_name}_model.pt"))

        # Save attention patterns as numpy arrays
        if self.attention_patterns:
            np.save(os.path.join(self.results_dir, f"{self.model_name}_attention_patterns.npy"),
                   self.attention_patterns)


def run_experiment(task_name, model_configs, training_config, results_dir):
    """
    Run experiment comparing different attention mechanisms on a specific task.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    task_results_dir = os.path.join(results_dir, task_name)
    os.makedirs(task_results_dir, exist_ok=True)

    print(f"Loading {task_name} data...")

    # ✅ Extract max_seq_length from one of the sub-configs
    max_length = model_configs['conventional']['max_seq_length']

    if task_name == 'imdb':
        dataloaders = create_imdb_dataloaders(
            batch_size=training_config['batch_size'],
            max_length=max_length,
            subset_size=training_config.get('subset_size')
        )
        num_classes = 2
    elif task_name == 'listops':
        dataloaders = create_listops_dataloaders(
            batch_size=training_config['batch_size'],
            max_length=max_length,
            subset_size=training_config.get('subset_size')
        )
        num_classes = 10
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # ✅ Unpack dataloaders dictionary
    train_dataloader = dataloaders["train"]
    test_dataloader = dataloaders["test"]
    vocab_size = dataloaders["vocab_size"]

    for config in model_configs.values():
        config['vocab_size'] = vocab_size
        config['num_classes'] = num_classes
        config['pad_idx'] = 1

    models = {
        'conventional': create_conventional_model(model_configs['conventional']),
        'gravitational': create_gravitational_model(model_configs['gravitational']),
        'multi_timescale': create_multi_timescale_model(model_configs['multi_timescale']),
        'gravitational_multi_timescale': create_gravitational_multi_timescale_model(
            model_configs['gravitational_multi_timescale']
        )
    }

    results = {}
    for model_name, model in models.items():
        result_path = os.path.join(task_results_dir, f"{model_name}_results.json")
        model_path = os.path.join(task_results_dir, f"{model_name}_model.pt")
    
        if os.path.exists(result_path) and os.path.exists(model_path):
            print(f"⏭️ Skipping {model_name} on {task_name} (results and model already exist)")
            with open(result_path, "r") as f:
                results[model_name] = json.load(f)
            continue
    
        print(f"\n{'='*50}")
        print(f"Training {model_name} model on {task_name}")
        print(f"{'='*50}")
    
        trainer = AttentionModelTrainer(
            model_name=model_name,
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            device=device,
            config=training_config,
            results_dir=task_results_dir
        )
    
        model_results = trainer.train()
        results[model_name] = model_results



    return results



if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Reuse a fixed results directory unless overridden
    results_dir = "./results_current"
    os.makedirs(results_dir, exist_ok=True)


    # Define model configurations
    base_model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512,
        'max_seq_length': 1024,  # Reduced for MVP
        'dropout': 0.1
    }

    # Create specific model configs
    model_configs = {
        'conventional': base_model_config.copy(),
        'gravitational': base_model_config.copy(),
        'multi_timescale': {**base_model_config.copy(), 'n_timescales': 3},
        'gravitational_multi_timescale': {**base_model_config.copy(), 'n_timescales': 3}
    }

    # Define training configuration
    training_config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 20,
        'max_grad_norm': 1.0,
        'subset_size': 1000  # Use a small subset for MVP
    }

    # Save configurations
    with open(os.path.join(results_dir, 'configs.json'), 'w') as f:
        json.dump({
            'model_configs': model_configs,
            'training_config': training_config
        }, f, indent=2)

    # Run experiments for each task
    tasks = ['imdb', 'listops']
    all_results = {}

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"# Running experiment for {task} task")
        print(f"{'#'*60}\n")

        task_results = run_experiment(
            task_name=task,
            model_configs=model_configs,
            training_config=training_config,
            results_dir=results_dir
        )

        all_results[task] = task_results

    print("\nAll experiments completed!")
