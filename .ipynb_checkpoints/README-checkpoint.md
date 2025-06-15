# Internal Coherence Maximization (ICM) Implementation

A PyTorch implementation of the Internal Coherence Maximization algorithm from the paper "Unsupervised Elicitation of Language Models" by Wen et al. (2025). This implementation supports both vLLM and transformers backends for efficient inference.

## Overview

ICM is an unsupervised algorithm that fine-tunes pretrained language models on their own generated labels without external supervision. It works by:

1. **Mutual Predictability**: Finding labels where the model can infer each label from all others
2. **Logical Consistency**: Enforcing task-specific consistency constraints
3. **Simulated Annealing**: Iteratively improving the label set using temperature-based acceptance

## Features

- 🚀 **Dual Backend Support**: Optimized vLLM backend for production, transformers for compatibility
- 🔧 **Modular Design**: Easily extensible components for different tasks
- 📊 **Built-in Tasks**: Support for truthfulness, math correctness, and comparison tasks
- 🧪 **Comprehensive Testing**: Unit tests and integration tests included
- 📈 **Performance Tracking**: Detailed metrics and experiment logging

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Basic Installation

```bash
# Install core dependencies
pip install torch transformers tqdm numpy pandas

# For vLLM backend (recommended for performance)
pip install vllm

# For the default Qwen3 models
pip install transformers>=4.51.0
```

### Docker Installation (Recommended)

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN pip install --upgrade pip && \
    pip install vllm==0.7.0 transformers>=4.51.0 \
    tqdm numpy pandas psutil
```

## Quick Start

### 1. Basic Usage

```python
from icm_implementation import ICM, ICMConfig, create_truthfulness_dataset

# Create dataset
data = [
    ("Is the Earth round?", "Yes, the Earth is spherical", None),
    ("Is the Earth flat?", "No, the Earth is round", None),
    ("Is 2+2=4?", "Yes, 2+2 equals 4", None),
    ("Is 2+2=5?", "No, 2+2 equals 4, not 5", None),
]

dataset = create_truthfulness_dataset(data)

# Configure ICM
config = ICMConfig(
    model_name="Qwen/Qwen3-4B",  # or any HF model
    backend="auto",  # uses vLLM if available
    initial_examples=2,
    alpha=50.0
)

# Run ICM
icm = ICM(config)
labeled_data = icm.run(dataset)

# Check results
for data_point, label in labeled_data:
    print(f"Input: {data_point.input_text}")
    print(f"Label: {config.label_names[label]}\n")
```

### 2. Running Experiments

```bash
# Run all tasks with default model
python icm_examples.py --task all

# Run specific task with custom model
python icm_examples.py --task math --model meta-llama/Llama-3.2-1B

# Quick test with small dataset
python icm_examples.py --task truthfulness --small

# Compare backends
python icm_examples.py --compare-backends
```

### 3. Custom Tasks

```python
from icm_implementation import ICM, ICMConfig, DataPoint, LogicalConsistency

class CustomConsistency(LogicalConsistency):
    def check_consistency(self, x_i, y_i, x_j, y_j):
        # Implement your consistency logic
        return True  # or False based on your constraints

# Create custom dataset
dataset = [
    DataPoint(
        id=i,
        input_text="Your task-specific input",
        metadata={"custom_field": value}
    )
    for i, value in enumerate(your_data)
]

# Run with custom consistency
config = ICMConfig(num_labels=3, label_names=["A", "B", "C"])
icm = ICM(config)
icm.consistency_checker = CustomConsistency()
results = icm.run(dataset)
```

## Architecture

### Core Components

1. **ModelBackend**: Abstract interface for model inference
   - `VLLMBackend`: High-performance batch inference
   - `TransformersBackend`: Compatible with any HuggingFace model

2. **LogicalConsistency**: Handles task-specific consistency checking
   - General consistency (default)
   - Asymmetry consistency (for comparisons)
   - Math correctness consistency

3. **ICM Algorithm**: Main algorithm implementation
   - Simulated annealing with temperature scheduling
   - Consistency fixing subroutine
   - Score calculation and tracking

### Configuration Options

```python
@dataclass
class ICMConfig:
    # Model settings
    model_name: str = "Qwen/Qwen3-4B"
    backend: str = "auto"  # "vllm", "transformers", or "auto"
    
    # Algorithm parameters
    initial_examples: int = 8        # K in the paper
    initial_temperature: float = 10.0  # T_0
    final_temperature: float = 0.01    # T_min
    cooling_rate: float = 0.99         # β
    alpha: float = 50.0                # Mutual predictability weight
    
    # Inference settings
    max_context_length: int = 32768
    max_new_tokens: int = 64
    temperature: float = 0.1
    top_p: float = 0.95
```

## Supported Tasks

### 1. Truthfulness (TruthfulQA-style)
```python
dataset = create_truthfulness_dataset([
    (question, claim, is_true),  # is_true can be None
    ...
])
```

### 2. Mathematical Correctness (GSM8K-style)
```python
dataset = create_math_correctness_dataset([
    (problem, solution, answer, is_correct),
    ...
])
```

### 3. Comparison (Alpaca-style)
```python
dataset = create_comparison_dataset([
    (query, response_a, response_b, a_is_better),
    ...
])
```

## Performance Optimization

### Memory Management
- Use smaller `max_context_length` for limited GPU memory
- Adjust `initial_examples` based on dataset size
- Use `backend="transformers"` with CPU for testing

### Speed Optimization
- Use vLLM backend for 5-10x speedup
- Batch size is automatically optimized
- Reduce `max_iterations` for faster results

### Model Selection
- Qwen3-4B: Best balance of performance and efficiency
- Qwen3-1.7B: For resource-constrained environments
- Llama-3.2-1B: Alternative lightweight option

## Testing

```bash
# Run all tests
python icm_test_suite.py

# Run specific test class
python -m unittest icm_test_suite.TestLogicalConsistency

# Run with verbose output
python icm_test_suite.py -v
```

## Experiment Tracking

Results are automatically saved to `icm_results/` with:
- Detailed JSON logs for each experiment
- Summary CSV with key metrics
- Score history and acceptance rates

## Limitations

1. **Context Length**: Limited by model's context window for in-context examples
2. **Concept Salience**: Only works for concepts the model already understands
3. **Compute Requirements**: Requires multiple forward passes per label

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{wen2025unsupervised,
  title={Unsupervised Elicitation of Language Models},
  author={Wen, Jiaxin and others},
  journal={arXiv preprint arXiv:2505.15134},
  year={2025}
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   config.max_context_length = 4096  # Reduce context
   config.backend = "transformers"   # Use CPU
   ```

2. **vLLM Import Error**
   ```bash
   # Install with specific CUDA version
   pip install vllm --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Slow Performance**
   - Ensure vLLM backend is being used
   - Check GPU utilization with `nvidia-smi`
   - Reduce dataset size or max_iterations

## Contributing

Contributions are welcome! Areas for improvement:
- Additional consistency types
- Support for more model architectures
- Multi-GPU support
- Additional evaluation metrics

## License

This implementation is provided for research purposes. Please ensure you comply with the licenses of the models you use (Qwen3, Llama, etc.).