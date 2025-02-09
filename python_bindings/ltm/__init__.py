"""
LTM Transformer: Long-term Memory Transformer with Titan-inspired architecture
"""

from ltm.model import (
    TitanModel,
    TitanConfig,
    TitanForCausalLM,
    TitanForSequenceClassification,
    TitanForTokenClassification,
    TitanPreTrainedModel,
)

from ltm.trainer import (
    Trainer,
    TrainingArguments,
    DataCollator,
    DistributedTrainer,
    TrainerCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
    WandBCallback,
)

from ltm.inference import (
    InferenceEngine,
    InferenceConfig,
    BatchProcessor,
    CacheManager,
    QuantizedEngine,
    TensorParallelEngine,
)

# Version information
__version__ = "0.1.0"
__author__ = "Sahibzada Allahyar"
__author_email__ = "allahyar@singularityresearch.org"
__license__ = "Apache License 2.0"
__copyright__ = "Copyright 2025 Singularity Research"
__homepage__ = "https://github.com/singularityresearch/ltm-transformer"
__docs__ = "https://ltm-transformer.readthedocs.io/"

# Module level configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Check for CUDA availability
import torch
CUDA_AVAILABLE = torch.cuda.is_available()
if not CUDA_AVAILABLE:
    logging.warning("CUDA is not available. LTM Transformer will run in CPU-only mode.")

# Import optional dependencies
try:
    import mpi4py
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    logging.info("mpi4py not found. Distributed training features will be limited.")

try:
    import horovod.torch as hvd
    HOROVOD_AVAILABLE = True
except ImportError:
    HOROVOD_AVAILABLE = False
    logging.info("Horovod not found. Some distributed training features will be disabled.")

try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.info("ONNX/ONNXRuntime not found. Quantization features will be limited.")

# Public API
__all__ = [
    # Models
    "TitanModel",
    "TitanConfig",
    "TitanForCausalLM",
    "TitanForSequenceClassification",
    "TitanForTokenClassification",
    "TitanPreTrainedModel",
    
    # Training
    "Trainer",
    "TrainingArguments",
    "DataCollator",
    "DistributedTrainer",
    "TrainerCallback",
    "EarlyStoppingCallback",
    "TensorBoardCallback",
    "WandBCallback",
    
    # Inference
    "InferenceEngine",
    "InferenceConfig",
    "BatchProcessor",
    "CacheManager",
    "QuantizedEngine",
    "TensorParallelEngine",
]

def get_device():
    """Get the default device (CUDA if available, else CPU)."""
    return torch.device("cuda" if CUDA_AVAILABLE else "cpu")

def is_distributed_available():
    """Check if distributed training is available."""
    return MPI_AVAILABLE or HOROVOD_AVAILABLE

def is_quantization_available():
    """Check if quantization features are available."""
    return ONNX_AVAILABLE

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(seed)

def get_config_path():
    """Get the path to the default configuration directory."""
    import os
    return os.path.join(os.path.dirname(__file__), "config")

def cite():
    """Print citation information."""
    print(
        """
If you use LTM Transformer in your research, please cite:

@article{allahyar2025ltm,
    title={LTM Transformer: Long-term Memory Transformer with Titan-inspired Architecture},
    author={Allahyar, Sahibzada},
    journal={arXiv preprint arXiv:2025.xxxxx},
    year={2025}
}
        """
    )
