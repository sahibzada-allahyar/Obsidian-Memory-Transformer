"""
Inference utilities and engine for LTM models
"""

import os
import json
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Any, Tuple, Iterator
from dataclasses import dataclass, field

from transformers import PreTrainedTokenizer

from .model import TitanPreTrainedModel
from ._ltm import (
    InferenceEngineImpl,
    BatchProcessorImpl,
    CacheManagerImpl,
    QuantizerImpl,
)

@dataclass
class InferenceConfig:
    """
    Configuration for inference engine.
    """
    # Model configuration
    max_sequence_length: int = 2048
    max_batch_size: int = 32
    num_beams: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    
    # Memory configuration
    max_memory_length: int = 16384
    memory_compression_ratio: float = 0.5
    
    # Hardware configuration
    device: str = "cuda"
    dtype: str = "float16"
    use_flash_attention: bool = True
    use_tensor_parallel: bool = False
    tensor_parallel_size: int = 1
    
    # Optimization configuration
    use_cache: bool = True
    use_kv_cache: bool = True
    use_memory_cache: bool = True
    use_quantization: bool = False
    quantization_bits: int = 8
    
    # Performance configuration
    max_concurrent_requests: int = 16
    batch_timeout_ms: int = 100
    stream_output: bool = False

class BatchProcessor:
    """
    Handles batched inference requests.
    """
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.impl = BatchProcessorImpl(config)
    
    def add_request(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> int:
        """Add inference request to batch."""
        return self.impl.add_request(
            input_ids,
            attention_mask,
            max_new_tokens or self.config.max_sequence_length,
            kwargs
        )
    
    def process_batch(self) -> Iterator[Dict[str, Any]]:
        """Process current batch of requests."""
        for result in self.impl.process_batch():
            yield {
                "request_id": result.request_id,
                "output_ids": result.output_ids,
                "scores": result.scores,
                "attention": result.attention,
            }
    
    def is_batch_ready(self) -> bool:
        """Check if batch is ready for processing."""
        return self.impl.is_batch_ready()
    
    def clear(self):
        """Clear current batch."""
        self.impl.clear()

class CacheManager:
    """
    Manages KV and memory caches for efficient inference.
    """
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.impl = CacheManagerImpl(config)
    
    def allocate(self, batch_size: int, seq_length: int):
        """Allocate cache for batch."""
        self.impl.allocate(batch_size, seq_length)
    
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ):
        """Update cache with new KV and memory states."""
        self.impl.update(key, value, memory)
    
    def get(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get cached states for index."""
        return self.impl.get(index)
    
    def clear(self):
        """Clear all caches."""
        self.impl.clear()

class QuantizedEngine:
    """
    Handles quantized inference.
    """
    def __init__(self, model: TitanPreTrainedModel, config: InferenceConfig):
        self.model = model
        self.config = config
        self.impl = QuantizerImpl(config)
        
        # Quantize model
        self.quantized_model = self.impl.quantize(model)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Run forward pass with quantized model."""
        return self.impl.forward(
            self.quantized_model,
            input_ids,
            attention_mask,
            kwargs
        )

class TensorParallelEngine:
    """
    Handles tensor parallel inference.
    """
    def __init__(self, model: TitanPreTrainedModel, config: InferenceConfig):
        if not config.use_tensor_parallel:
            raise ValueError("Tensor parallel inference not enabled in config")
        
        self.model = model
        self.config = config
        
        # Initialize tensor parallel
        self.impl = self.setup_tensor_parallel()
    
    def setup_tensor_parallel(self):
        """Set up tensor parallel execution."""
        import torch.distributed as dist
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        # Shard model across GPUs
        return self.model.parallelize()
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Run forward pass with tensor parallel model."""
        return self.impl.forward(
            input_ids,
            attention_mask,
            kwargs
        )

class InferenceEngine:
    """
    Main inference engine for LTM models.
    """
    def __init__(
        self,
        model: TitanPreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: InferenceConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Set up device and dtype
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        
        # Move model to device and convert to dtype
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        
        # Initialize components
        self.batch_processor = BatchProcessor(config)
        self.cache_manager = CacheManager(config)
        
        # Set up optimized engines if enabled
        self.quantized_engine = (
            QuantizedEngine(self.model, config)
            if config.use_quantization
            else None
        )
        
        self.tensor_parallel_engine = (
            TensorParallelEngine(self.model, config)
            if config.use_tensor_parallel
            else None
        )
        
        # Create C++ implementation
        self.impl = InferenceEngineImpl(
            self.model,
            config,
            self.batch_processor.impl,
            self.cache_manager.impl
        )
    
    def generate(
        self,
        input_ids: Union[torch.LongTensor, List[int]],
        attention_mask: Optional[torch.FloatTensor] = None,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        output_attentions: bool = False,
        **kwargs
    ) -> Union[torch.LongTensor, Dict[str, Any]]:
        """
        Generate text from input prompt.
        """
        # Convert inputs to tensors
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Set generation config
        gen_config = {
            "max_new_tokens": max_new_tokens or self.config.max_sequence_length,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams or self.config.num_beams,
            "temperature": temperature or self.config.temperature,
            "top_k": top_k or self.config.top_k,
            "top_p": top_p or self.config.top_p,
            "repetition_penalty": repetition_penalty or self.config.repetition_penalty,
            "length_penalty": length_penalty or self.config.length_penalty,
            "stop_sequences": stop_sequences,
        }
        
        # Run generation
        outputs = self.impl.generate(
            input_ids,
            attention_mask,
            gen_config,
            return_dict_in_generate,
            output_scores,
            output_attentions
        )
        
        if not return_dict_in_generate:
            return outputs
        
        return {
            "sequences": outputs.sequences,
            "scores": outputs.scores if output_scores else None,
            "attentions": outputs.attentions if output_attentions else None,
        }
    
    def stream_generate(
        self,
        input_ids: Union[torch.LongTensor, List[int]],
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream generation results token by token.
        """
        # Convert inputs to tensors
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        
        # Stream tokens
        for output in self.impl.stream_generate(input_ids, kwargs):
            yield {
                "token_id": output.token_id,
                "token": self.tokenizer.decode([output.token_id]),
                "score": output.score,
                "attention": output.attention,
            }
    
    @torch.no_grad()
    def encode(
        self,
        input_ids: Union[torch.LongTensor, List[int]],
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> torch.FloatTensor:
        """
        Encode input sequence to hidden states.
        """
        # Convert inputs to tensors
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Run encoder
        return self.impl.encode(input_ids, attention_mask, kwargs)
    
    def save_pretrained(self, save_dir: str):
        """Save model and configuration."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # Save config
        config_path = os.path.join(save_dir, "inference_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[InferenceConfig] = None,
        **kwargs
    ) -> "InferenceEngine":
        """Load model and configuration from path."""
        # Load model
        model = TitanPreTrainedModel.from_pretrained(model_path)
        
        # Load tokenizer
        tokenizer = PreTrainedTokenizer.from_pretrained(model_path)
        
        # Load config
        if config is None:
            config_path = os.path.join(model_path, "inference_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config_dict = json.load(f)
                config = InferenceConfig(**config_dict)
            else:
                config = InferenceConfig()
        
        return cls(model, tokenizer, config, **kwargs)
