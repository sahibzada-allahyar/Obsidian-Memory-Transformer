"""
TitanModel implementation and configuration classes
"""

import os
import json
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PretrainedConfig

from ._ltm import (
    TitanModelImpl,
    TitanBlockConfig,
    MemoryBankConfig,
    FlashAttentionConfig,
)

@dataclass
class TitanConfig(PretrainedConfig):
    """
    Configuration class for TitanModel.
    """
    model_type: str = "titan"
    
    # Model architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Memory configuration
    memory_slots: int = 512
    memory_dim: int = 64
    memory_update_rate: float = 0.9
    use_memory_compression: bool = True
    memory_compression_ratio: float = 0.5
    
    # Attention configuration
    use_flash_attention: bool = True
    use_alibi: bool = False
    use_rotary: bool = True
    causal_mask: bool = True
    
    # Training configuration
    gradient_checkpointing: bool = False
    use_cache: bool = True
    
    def __post_init__(self):
        super().__init__()
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Validate configuration
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_attention_heads {self.num_attention_heads}"
            )

class TitanPreTrainedModel(PreTrainedModel):
    """
    Base class for all Titan models.
    """
    config_class = TitanConfig
    base_model_prefix = "titan"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TitanModel):
            module.gradient_checkpointing = value

class TitanModel(TitanPreTrainedModel):
    """
    The base Titan Model transformer outputting raw hidden-states.
    """
    def __init__(self, config: TitanConfig):
        super().__init__(config)
        
        # Create C++ implementation
        self.impl = TitanModelImpl(
            TitanBlockConfig(
                hidden_dim=config.hidden_size,
                ffn_dim=config.intermediate_size,
                num_heads=config.num_attention_heads,
                head_dim=config.head_dim,
                memory_slots=config.memory_slots,
                memory_dim=config.memory_dim,
                memory_update_rate=config.memory_update_rate,
                use_memory_compression=config.use_memory_compression,
                memory_compression_ratio=config.memory_compression_ratio,
                use_flash_attention=config.use_flash_attention,
                use_alibi=config.use_alibi,
                use_rotary=config.use_rotary,
                dropout_prob=config.hidden_dropout_prob,
                use_bias=True,
                use_layer_norm=True,
                fuse_operations=True,
            )
        )
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size
        )
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.post_init()
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self) -> nn.Module:
        return self.word_embeddings
    
    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.word_embeddings = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        # Handle inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        batch_size, seq_length = inputs_embeds.shape[:2]
        
        # Create position IDs if none provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Create token type IDs if none provided
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (batch_size, seq_length),
                dtype=torch.long,
                device=inputs_embeds.device
            )
        
        # Get position and token type embeddings
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Forward pass through model
        outputs = self.impl(
            embeddings,
            attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
        )
        
        return outputs

class TitanForCausalLM(TitanPreTrainedModel):
    """
    Titan Model with a language modeling head on top.
    """
    def __init__(self, config: TitanConfig):
        super().__init__(config)
        self.titan = TitanModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        outputs = self.titan(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = prediction_scores[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': prediction_scores,
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

class TitanForSequenceClassification(TitanPreTrainedModel):
    """
    Titan Model with a sequence classification head on top.
    """
    def __init__(self, config: TitanConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.titan = TitanModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        outputs = self.titan(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

class TitanForTokenClassification(TitanPreTrainedModel):
    """
    Titan Model with a token classification head on top.
    """
    def __init__(self, config: TitanConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.titan = TitanModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        outputs = self.titan(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
