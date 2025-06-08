# Attention Mechanism Comparison MVP Report

## Overview

This report presents a Minimum Viable Product (MVP) implementation comparing four different attention mechanisms in transformer models:

1. **Conventional Attention**: Standard scaled dot-product attention as described in the original "Attention Is All You Need" paper
2. **Gravitational Attention**: A novel approach where tokens interact based on their "masses" and distances, similar to gravitational fields
3. **Multi-Timescale Attention**: An approach that processes information at multiple temporal scales simultaneously
4. **Gravitational + Multi-Timescale Attention**: A combined approach integrating both gravitational principles and multi-timescale processing

The comparison is performed on selected tasks from the Long Range Arena (LRA) benchmark, specifically designed to evaluate models on their ability to handle long-range dependencies.

## Benchmark Selection

The Long Range Arena (LRA) benchmark was selected for this comparison because:

1. It specifically tests long-range dependencies, which is crucial for evaluating multi-timescale approaches
2. It includes diverse tasks that test different aspects of attention mechanisms
3. It is a standard benchmark in the field, making results comparable to published literature

For this MVP, we focus on two tasks from LRA:

1. **Text Classification (IMDb)**: Tests document-level understanding with sequences up to 4,096 tokens
2. **ListOps**: Tests hierarchical structure understanding with deeply nested operations

These tasks provide a good balance of different types of dependencies (semantic vs. hierarchical) and computational requirements.

## Implementation Details

### Model Architectures

All four attention variants share the same overall transformer encoder architecture, differing only in their attention mechanism:

- **Conventional Attention**: Implements standard scaled dot-product attention with multiple heads
- **Gravitational Attention**: Replaces dot-product with a gravitational formula where tokens have "masses" and interact based on their distances
- **Multi-Timescale Attention**: Processes information at multiple temporal scales with different discount factors, combining results through a Laplace-like decoder
- **Gravitational + Multi-Timescale Attention**: Integrates gravitational principles with multi-timescale processing

### Key Implementation Features

#### Gravitational Attention
- Tokens are assigned learnable "masses" through projection layers
- Attention scores are calculated using an inverse square law: G * (m_q * m_k) / r²
- Distances are computed in embedding space using Euclidean distance

#### Multi-Timescale Attention
- Multiple discount factors (γ) are applied to capture dependencies at different timescales
- Each timescale has its own set of query, key, and value projections
- A learned inverse mapping (Laplace-like decoder) combines information across timescales

#### Gravitational + Multi-Timescale Attention
- Tokens have multiple "masses" corresponding to different timescales
- Gravitational interactions are computed separately for each timescale
- Temporal discounting is applied based on position differences
- Results from all timescales are combined through a Laplace-like decoder

### Data Pipeline

The data pipeline includes:
- Custom dataset classes for IMDb and ListOps tasks
- Tokenization and vocabulary building
- Padding and truncation to fixed sequence lengths
- Batch creation with attention masks

## Experimental Setup

### Model Configuration
- Embedding dimension: 128
- Number of heads: 4
- Number of layers: 2
- Feed-forward dimension: 512
- Number of timescales (for multi-timescale models): 3
- Dropout: 0.1

### Training Configuration
- Batch size: 32
- Learning rate: 1e-4
- Weight decay: 1e-5
- Maximum epochs: 20
- Early stopping: If validation accuracy doesn't improve for 10 epochs
- Gradient clipping: 1.0

### Evaluation Metrics
- Accuracy: Primary metric for both tasks
- Convergence speed: Epochs to reach 90% of maximum performance
- Training time: Average seconds per epoch
- Memory usage: Peak memory consumption during training
- Attention pattern visualization: Qualitative analysis of what each model attends to

## Results and Analysis

[Note: In a real implementation, this section would contain actual results from running the experiments. For this MVP report, we describe the expected analysis.]

### Performance Comparison

The performance comparison includes:

1. **Accuracy Comparison**: Bar charts comparing final accuracy across models and tasks
2. **Convergence Speed**: Analysis of how quickly each model reaches 90% of its maximum performance
3. **Computational Efficiency**: Comparison of training time and memory usage
4. **Learning Curves**: Plots showing training and validation accuracy/loss over epochs
5. **Attention Pattern Visualization**: Heatmaps showing what different models attend to
6. **Relative Improvement**: Percentage improvement over conventional attention
7. **Radar Charts**: Multi-dimensional comparison across accuracy, speed, efficiency, and memory usage

### Key Findings

[Note: Placeholder for actual findings. In a real implementation, this would be based on experimental results.]

Expected findings might include:
- How gravitational attention compares to conventional attention in capturing long-range dependencies
- Whether multi-timescale processing provides benefits for different types of tasks
- If the combined approach offers the best of both worlds or introduces too much complexity
- Trade-offs between performance and computational efficiency

## Code Structure

The implementation consists of the following Python files:

1. `conventional_attention.py`: Implementation of standard transformer attention
2. `gravitational_attention.py`: Implementation of gravitational attention mechanism
3. `multi_timescale_attention.py`: Implementation of multi-timescale attention mechanism
4. `gravitational_multi_timescale_attention.py`: Implementation of combined approach
5. `data_pipeline.py`: Data loading and preprocessing for LRA tasks
6. `train_and_evaluate.py`: Training and evaluation functionality
7. `evaluate_performance.py`: Performance metrics calculation and visualization
8. `analyze_results.py`: Comparative analysis and visualization
9. `run_experiment.py`: Main script to run the complete experiment

## Running the MVP

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- NumPy, Pandas, Matplotlib, Seaborn
- tqdm

### Installation
```bash
pip install torch numpy pandas matplotlib seaborn tqdm
```

### Running the Experiment
```bash
# Run the complete experiment
python run_experiment.py

# Analyze results from a specific run
python analyze_results.py ./results_TIMESTAMP
```

### Expected Output
The experiment will create a `results_TIMESTAMP` directory containing:
- Model checkpoints
- Training and validation metrics
- Attention pattern visualizations
- Performance comparison plots
- Summary tables

## Conclusion and Future Work

This MVP provides a framework for comparing different attention mechanisms, with a focus on gravitational and multi-timescale approaches. The implementation allows for fair comparison across standard benchmark tasks and provides comprehensive evaluation metrics.

Future work could include:
1. Scaling to larger models and more complex tasks
2. Exploring different gravitational formulations (e.g., different distance metrics)
3. Testing with more diverse timescales and discount factors
4. Implementing adaptive timescale selection
5. Extending to generative tasks and autoregressive models

## References

1. Vaswani, A., et al. (2017). "Attention is all you need." NeurIPS.
2. Tay, Y., et al. (2020). "Long Range Arena: A Benchmark for Efficient Transformers." arXiv.
3. Masset, P., et al. (2025). "Multi-timescale reinforcement learning in the brain." Nature.
