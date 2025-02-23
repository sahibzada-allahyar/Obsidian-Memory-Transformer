# Obsidian Memory Transformer


A novel LLM architecture written in highly optimized low-level C++/CUDA with a new Long-Term Memory (LTM) mechanism for large context windows. This is a high-performance implementation of a Transformer model with long-term memory capabilities, inspired by Google's Titan architecture. This project provides efficient CUDA implementations of FlashAttention and memory-augmented Transformer blocks, along with Python bindings for easy integration.

## Features

- **Long-term Memory**: Novel memory mechanism for handling extended context windows efficiently
- **FlashAttention**: Memory-efficient attention implementation with minimal memory access
- **High Performance**:
  - Optimized CUDA kernels
  - Mixed precision training (FP16/BF16)
  - Quantization support (INT8/INT4)
  - Fused operations for better throughput
- **Distributed Training**:
  - Data parallelism
  - Tensor parallelism
  - Pipeline parallelism
  - Multi-node support via MPI
- **Python Integration**:
  - HuggingFace-compatible interface
  - Easy-to-use training API
  - Efficient inference engine

## Installation

### Prerequisites

- CUDA Toolkit (>= 11.0)
- CMake (>= 3.15)
- C++17 compatible compiler
- Python (>= 3.7)
- PyTorch (>= 1.9.0)

### Installing from PyPI

```bash
pip install ltm-transformer
```

### Building from Source

1. Clone the repository:
```bash
git clone https://github.com/singularityresearch/ltm-transformer.git
cd ltm-transformer
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Build and install:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
make install
```

## Quick Start

### Python

```python
from ltm import TitanModel, TitanConfig, InferenceEngine

# Initialize model
config = TitanConfig(
    hidden_size=768,
    num_attention_heads=12,
    memory_slots=512,
    use_flash_attention=True
)
model = TitanModel(config)

# Training
from ltm import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./outputs",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4
    ),
    train_dataset=dataset
)
trainer.train()

# Inference
engine = InferenceEngine(
    model=model,
    config=InferenceConfig(
        use_flash_attention=True,
        use_memory_cache=True,
        max_sequence_length=2048
    )
)

output = engine.generate(
    input_ids=tokenizer.encode("Hello, how are"),
    max_new_tokens=50
)
```

### C++

```cpp
#include "ltm/transformer/titan_inspired_block.cuh"

// Configure model
ltm::transformer::TitanBlockConfig config;
config.hidden_dim = 768;
config.num_heads = 12;
config.memory_slots = 512;
config.use_flash_attention = true;

// Create model
auto model = std::make_unique<ltm::transformer::TitanBlock<float>>(config);

// Run inference
torch::Tensor input = /* ... */;
auto output = model->forward(input);
```

## Architecture

The LTM Transformer extends the standard Transformer architecture with:

1. **Memory Bank**: A trainable matrix storing compressed representations of past context
2. **Compression Gate**: Mechanism for compressing and storing relevant information
3. **Memory Attention**: Efficient attention between current context and memory bank
4. **FlashAttention**: Memory-efficient attention implementation

For detailed architecture information, see [docs/design/architecture.md](docs/design/architecture.md).

## Performance

### Memory Usage

| Context Length | Standard Transformer | LTM Transformer |
|---------------|---------------------|-----------------|
| 2K tokens     | 4 GB                | 2 GB           |
| 8K tokens     | 64 GB               | 4 GB           |
| 32K tokens    | 1024 GB             | 8 GB           |

### Training Speed

- 1.5x faster training compared to standard Transformers
- 4x reduction in memory bandwidth usage
- Linear scaling up to 64 GPUs

For detailed benchmarks, see [docs/performance/optimization.md](docs/performance/optimization.md).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Build with testing enabled:
```bash
mkdir build && cd build
cmake -DBUILD_TESTING=ON ..
make -j$(nproc)
```

3. Run tests:
```bash
ctest --output-on-failure
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{allahyar2025ltm,
    title={LTM Transformer: Long-term Memory Transformer with Titan-inspired Architecture},
    author={Allahyar, Sahibzada},
    journal={arXiv preprint arXiv:2025.xxxxx},
    year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google's Titan architecture for inspiration
- FlashAttention paper for efficient attention implementation
- HuggingFace team for transformer implementations
- NVIDIA for CUDA optimization guidelines

## Contact

- Sahibzada A - sahibzada@singularityresearchlabs.com
- Project Link: https://github.com/Sahibzada-A/Obsidian-Memory-Transformer
