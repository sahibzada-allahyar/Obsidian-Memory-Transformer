# LTM Transformer Usage Guide

This guide provides practical examples and instructions for using the LTM Transformer library. It covers common tasks like training models, running inference, and optimizing performance.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Training](#training)
- [Inference](#inference)
- [Distributed Training](#distributed-training)
- [Performance Optimization](#performance-optimization)
- [Advanced Features](#advanced-features)

## Installation

### From PyPI

```bash
pip install ltm-transformer
```

### From Source

```bash
git clone https://github.com/singularityresearch/ltm-transformer.git
cd ltm-transformer
pip install -e .
```

## Basic Usage

### Creating a Model

```python
from ltm import TitanModel, TitanConfig

# Initialize configuration
config = TitanConfig(
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    memory_slots=512,
    memory_dim=64,
    use_flash_attention=True
)

# Create model
model = TitanModel(config)
```

### Simple Forward Pass

```python
import torch

# Prepare input
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=None,  # Optional attention mask
    use_cache=True       # Enable KV caching
)

# Access outputs
hidden_states = outputs.hidden_states
memory_states = outputs.memory_states
```

## Training

### Basic Training Loop

```python
from ltm import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Configure training
training_args = TrainingArguments(
    output_dir="./outputs",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    max_steps=100000,
    warmup_steps=10000,
    logging_steps=100,
    save_steps=1000,
    fp16=True,
    gradient_checkpointing=True
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

# Start training
trainer.train()
```

### Custom Training Loop

```python
import torch.optim as optim

# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update memory bank
        model.update_memory_bank(outputs.hidden_states)
```

## Inference

### Text Generation

```python
from ltm import InferenceEngine, InferenceConfig
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure inference
config = InferenceConfig(
    max_sequence_length=2048,
    use_flash_attention=True,
    use_memory_cache=True,
    batch_size=1
)

# Create inference engine
engine = InferenceEngine(model, tokenizer, config)

# Generate text
output = engine.generate(
    input_ids=tokenizer.encode("Once upon a time"),
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)

# Decode output
text = tokenizer.decode(output[0])
```

### Streaming Generation

```python
# Stream tokens one by one
for token in engine.stream_generate(
    input_ids=tokenizer.encode("The future of AI"),
    max_new_tokens=100
):
    print(tokenizer.decode([token]), end="", flush=True)
```

### Batch Processing

```python
# Process multiple inputs in parallel
inputs = [
    "First prompt",
    "Second prompt",
    "Third prompt"
]

# Tokenize inputs
input_ids = [tokenizer.encode(text) for text in inputs]

# Add to batch processor
for ids in input_ids:
    engine.batch_processor.add_request(ids)

# Process batch
results = list(engine.batch_processor.process_batch())
```

## Distributed Training

### Multi-GPU Training

```python
from ltm import DistributedTrainer

# Initialize distributed training
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    fp16=True,
    gradient_checkpointing=True,
    
    # Distributed settings
    local_rank=-1,  # Set by torch.distributed.launch
    tensor_parallel_size=4,
    pipeline_parallel_size=2
)

trainer = DistributedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Launch training
trainer.train()
```

### Multi-Node Training

```bash
# On node 1 (master)
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train.py

# On node 2
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train.py
```

## Performance Optimization

### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision training
from torch.cuda.amp import autocast

with autocast():
    outputs = model(input_ids)
```

### Quantization

```python
from ltm import QuantizedEngine

# Quantize model to INT8
engine = QuantizedEngine(
    model,
    config=InferenceConfig(
        use_quantization=True,
        quantization_bits=8
    )
)

# Run inference with quantized model
outputs = engine.generate(input_ids)
```

### Caching

```python
# Enable KV caching
engine = InferenceEngine(
    model,
    config=InferenceConfig(
        use_cache=True,
        use_kv_cache=True,
        use_memory_cache=True
    )
)

# First forward pass
outputs = engine.generate(
    input_ids,
    use_cache=True
)

# Subsequent passes will reuse cached states
next_outputs = engine.generate(
    new_input_ids,
    past_key_values=outputs.past_key_values
)
```

## Advanced Features

### Custom Memory Bank

```python
from ltm import MemoryBank, CompressionGate

# Configure custom memory bank
memory_bank = MemoryBank(
    num_slots=1024,
    slot_dim=128,
    update_rate=0.9
)

# Configure compression gate
compression_gate = CompressionGate(
    input_dim=768,
    compressed_dim=128,
    num_heads=4
)

# Use custom components
model = TitanModel(
    config=TitanConfig(
        memory_bank=memory_bank,
        compression_gate=compression_gate
    )
)
```

### Memory Visualization

```python
from ltm.utils import visualize_memory

# Get memory attention patterns
attention_patterns = model.get_memory_attention_patterns()

# Visualize memory usage
visualize_memory(
    attention_patterns,
    save_path="memory_visualization.png"
)
```

### Export Model

```python
# Save model and configuration
model.save_pretrained("./my_model")

# Save tokenizer
tokenizer.save_pretrained("./my_model")

# Load model
from ltm import TitanModel
model = TitanModel.from_pretrained("./my_model")
```

## Best Practices

1. **Memory Management**
   - Use gradient checkpointing for large models
   - Enable mixed precision training
   - Monitor memory usage with `nvidia-smi`

2. **Performance**
   - Use FlashAttention when possible
   - Enable memory caching for inference
   - Batch inputs when processing multiple sequences

3. **Training**
   - Start with small models for testing
   - Use learning rate warmup
   - Monitor memory bank usage
   - Save checkpoints regularly

4. **Inference**
   - Use quantization for deployment
   - Enable KV caching for autoregressive generation
   - Batch requests when possible

5. **Distributed Training**
   - Use tensor parallelism for large models
   - Enable pipeline parallelism when appropriate
   - Monitor GPU utilization across nodes

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```python
   # Solution: Enable gradient checkpointing
   model.gradient_checkpointing_enable()
   
   # Or reduce batch size
   training_args.per_device_train_batch_size //= 2
   ```

2. **Slow Training**
   ```python
   # Solution: Enable Flash Attention
   config.use_flash_attention = True
   
   # Use larger batch size with gradient accumulation
   training_args.gradient_accumulation_steps *= 2
   ```

3. **Memory Bank Issues**
   ```python
   # Solution: Monitor memory usage
   memory_stats = model.get_memory_stats()
   print(f"Memory utilization: {memory_stats['utilization']}")
   
   # Reset memory if needed
   model.reset_memory_bank()
   ```

### Getting Help

- Check the [GitHub issues](https://github.com/singularityresearch/ltm-transformer/issues)
- Join our [Discord community](https://discord.gg/ltm-transformer)
- Contact maintainers at support@singularityresearch.org
