"""
Training utilities and Trainer class for LTM models
"""

import os
import math
import json
import time
import logging
import warnings
from typing import Optional, Dict, List, Union, Any, Tuple, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, get_scheduler

from .model import TitanPreTrainedModel
from ._ltm import (
    DistributedConfig,
    GradientCheckpointer,
    MemoryManager,
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingArguments:
    """
    Arguments for training configuration.
    """
    output_dir: str = field(
        default="./outputs",
        metadata={"help": "Output directory for checkpoints and logs"},
    )
    
    # Training hyperparameters
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Initial learning rate"},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay coefficient"},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "Maximum number of training steps. Overrides num_train_epochs."},
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for learning rate scheduler"},
    )
    
    # Batch size and devices
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training"},
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation"},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before backward pass"},
    )
    
    # Mixed precision training
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use mixed precision training"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={"help": "Mixed precision optimization level"},
    )
    
    # Distributed training
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training"},
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Number of subprocesses for data loading"},
    )
    
    # Memory optimization
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing to save memory"},
    )
    
    # Logging and evaluation
    logging_dir: str = field(
        default=None,
        metadata={"help": "Tensorboard log directory"},
    )
    logging_steps: int = field(
        default=500,
        metadata={"help": "Log every X updates steps"},
    )
    eval_steps: int = field(
        default=None,
        metadata={"help": "Run evaluation every X steps"},
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps"},
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of checkpoints to keep"},
    )

class TrainerCallback:
    """Base class for trainer callbacks."""
    def on_train_begin(self, args: TrainingArguments, state: Dict, control: Dict) -> Dict:
        return control

    def on_train_end(self, args: TrainingArguments, state: Dict, control: Dict) -> Dict:
        return control

    def on_epoch_begin(self, args: TrainingArguments, state: Dict, control: Dict) -> Dict:
        return control

    def on_epoch_end(self, args: TrainingArguments, state: Dict, control: Dict) -> Dict:
        return control

    def on_step_begin(self, args: TrainingArguments, state: Dict, control: Dict) -> Dict:
        return control

    def on_step_end(self, args: TrainingArguments, state: Dict, control: Dict) -> Dict:
        return control

    def on_evaluate(self, args: TrainingArguments, state: Dict, control: Dict) -> Dict:
        return control

    def on_save(self, args: TrainingArguments, state: Dict, control: Dict) -> Dict:
        return control

    def on_log(self, args: TrainingArguments, state: Dict, control: Dict) -> Dict:
        return control

class Trainer:
    """
    Trainer class for LTM models.
    """
    def __init__(
        self,
        model: TitanPreTrainedModel,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        data_collator: Optional[Callable] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.callbacks = callbacks or []
        
        # Initialize distributed training if needed
        if args.local_rank != -1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.args.world_size = dist.get_world_size()
            self.args.n_gpu = 1
        else:
            self.args.world_size = 1
            self.args.n_gpu = torch.cuda.device_count()
        
        # Set up device
        if args.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.args.device = device
            self.model = self.model.to(device)
            if self.args.n_gpu > 1:
                self.model = nn.DataParallel(self.model)
        else:
            torch.cuda.set_device(args.local_rank)
            self.args.device = torch.device("cuda", args.local_rank)
            self.model = self.model.to(self.args.device)
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )
        
        # Set up optimizer and scheduler
        self.optimizer, self.scheduler = self.create_optimizers(optimizers)
        
        # Set up mixed precision training
        self.scaler = GradScaler() if args.fp16 else None
        
        # Set up memory optimization
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Initialize training state
        self.state = {
            "epoch": 0,
            "global_step": 0,
            "max_steps": self.get_max_steps(),
            "best_metric": None,
            "best_model_checkpoint": None,
            "log_history": [],
        }
        
        # Create output directory
        if self.is_world_process_zero():
            os.makedirs(args.output_dir, exist_ok=True)
            if args.logging_dir is not None:
                os.makedirs(args.logging_dir, exist_ok=True)
    
    def get_max_steps(self) -> int:
        """Calculate maximum number of training steps."""
        if self.args.max_steps > 0:
            return self.args.max_steps
        
        num_update_steps_per_epoch = len(self.train_dataloader()) // self.args.gradient_accumulation_steps
        return num_update_steps_per_epoch * self.args.num_train_epochs
    
    def create_optimizers(
        self,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR],
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """Create or get optimizer and scheduler."""
        if optimizers[0] is not None and optimizers[1] is not None:
            return optimizers
        
        # Create optimizer
        decay_parameters = self.get_parameter_names(self.model, [nn.LayerNorm])
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
        )
        
        # Create scheduler
        num_update_steps_per_epoch = len(self.train_dataloader()) // self.args.gradient_accumulation_steps
        num_training_steps = self.get_max_steps()
        
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        return optimizer, scheduler
    
    def get_parameter_names(self, model: nn.Module, skip_modules: List[type]) -> List[str]:
        """Get parameter names for weight decay."""
        result = []
        for name, module in model.named_modules():
            for skip_module in skip_modules:
                if isinstance(module, skip_module):
                    break
            else:
                for param_name, _ in module.named_parameters():
                    result.append(f"{name}.{param_name}" if name else param_name)
        return result
    
    def get_train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (
            DistributedSampler(self.train_dataset)
            if self.args.local_rank != -1
            else RandomSampler(self.train_dataset)
        )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )
    
    def get_eval_dataloader(self) -> DataLoader:
        """Create evaluation dataloader."""
        eval_sampler = SequentialSampler(self.eval_dataset)
        
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=eval_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )
    
    def train(self) -> None:
        """
        Main training loop.
        """
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        
        # Initialize training state
        self.state["max_steps"] = self.get_max_steps()
        self.state["num_train_epochs"] = math.ceil(self.state["max_steps"] / num_update_steps_per_epoch)
        self.state["global_step"] = 0
        
        # Training loop
        for epoch in range(self.state["num_train_epochs"]):
            self.state["epoch"] = epoch
            
            # Run epoch
            self._train_epoch(train_dataloader)
            
            # Save checkpoint
            if self.is_world_process_zero():
                self._save_checkpoint()
    
    def _train_epoch(self, train_dataloader: DataLoader) -> None:
        """Train for one epoch."""
        self.model.train()
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps for gradient accumulation
            if step % self.args.gradient_accumulation_steps != 0:
                continue
            
            # Forward pass
            with autocast(enabled=self.args.fp16):
                outputs = self.model(**self._prepare_inputs(batch))
                loss = outputs["loss"]
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass
            if self.args.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.fp16:
                    self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                
                # Update weights
                if self.args.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.state["global_step"] += 1
                
                # Logging
                if self.state["global_step"] % self.args.logging_steps == 0:
                    self._log_metrics({"loss": loss.item()})
                
                # Evaluation
                if (
                    self.args.eval_steps is not None
                    and self.state["global_step"] % self.args.eval_steps == 0
                ):
                    self.evaluate()
                
                # Callbacks
                for callback in self.callbacks:
                    callback.on_step_end(self.args, self.state, {})
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        eval_dataloader = self.get_eval_dataloader()
        
        # Run evaluation
        self.model.eval()
        losses = []
        
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = self.model(**self._prepare_inputs(batch))
                loss = outputs["loss"]
                losses.append(loss.item())
        
        metrics = {
            "eval_loss": sum(losses) / len(losses),
        }
        
        # Log metrics
        self._log_metrics(metrics)
        
        return metrics
    
    def _prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch inputs."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.args.device)
        return batch
    
    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics."""
        if self.is_world_process_zero():
            logger.info(f"Step {self.state['global_step']}: {metrics}")
            self.state["log_history"].append(metrics)
    
    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(
            self.args.output_dir,
            f"checkpoint-{self.state['global_step']}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save trainer state
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        torch.save(self.state, os.path.join(checkpoint_dir, "trainer_state.json"))
        
        # Remove old checkpoints
        if self.args.save_total_limit is not None:
            self._rotate_checkpoints()
    
    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints if save_total_limit is set."""
        if self.args.save_total_limit <= 0:
            return
        
        # Get all checkpoints
        checkpoints = [
            path
            for path in os.listdir(self.args.output_dir)
            if path.startswith("checkpoint-")
        ]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        
        # Remove old checkpoints
        if len(checkpoints) > self.args.save_total_limit:
            for checkpoint in checkpoints[:-self.args.save_total_limit]:
                checkpoint_path = os.path.join(self.args.output_dir, checkpoint)
                logger.info(f"Removing old checkpoint: {checkpoint_path}")
                os.system(f"rm -rf {checkpoint_path}")
    
    def is_world_process_zero(self) -> bool:
        """Check if this is the main process."""
        if self.args.local_rank == -1:
            return True
        return dist.get_rank() == 0
