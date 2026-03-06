from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from src.data_generation.utils_data import TRADING_DAYS


# ------------------------------------------------------------------
# Data configuration
# ------------------------------------------------------------------

@dataclass
class DataParams:
    """
    Configuration parameters describing the dataset used for training.

    This object defines:
      - Which assets are used
      - Time window of historical data
      - Option configuration
      - Data splitting strategy
      - Optional noise injection

    It does NOT contain actual data, only metadata describing how to
    load and prepare the dataset.
    """

    tickers: List[str]
    start_date: str
    end_date: str
    option_type: str

    window: int = 21                     # Rolling volatility window
    n_samples: int = -1                  # Number of training samples (-1 = full dataset)
    maturity: int = TRADING_DAYS / 4     # Option maturity in trading days
    use_percentage: bool = True          # Use percentage-based payoff definition
    noiseScale: float = 0.0              # Multiplicative noise added to training targets
    split_mode: Literal["random", "time"] = "random"
    # random: group-wise random split
    # time: chronological split


# ------------------------------------------------------------------
# Model configuration (base class)
# ------------------------------------------------------------------

@dataclass
class ModelParams:
    """
    Base class for model-specific parameter configurations.

    Allows polymorphic handling of different model types
    (e.g., classic neural networks vs quantum models).
    """
    pass


# ------------------------------------------------------------------
# Classical neural network configuration
# ------------------------------------------------------------------

@dataclass
class ClassicModelParams(ModelParams):
    """
    Configuration for classical (TensorFlow/Keras) models.
    """

    units: int                 # Width of hidden layers
    dropout: float             # Dropout rate
    model_name: str = "classic"


# ------------------------------------------------------------------
# Quantum model configuration
# ------------------------------------------------------------------

@dataclass
class QuantumModelParams(ModelParams):
    """
    Configuration for parameterized quantum circuit models.
    """

    layers: int                # Number of data re-uploading layers
    n_trainable_blocks: int    # Number of variational blocks per layer
    encoding_base: float = 3.0
    model_name: str = "quantum"


# ------------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------------

@dataclass
class TrainingParams:
    """
    Configuration for the optimization process.
    """

    learning_rate: float = 0.001
    epochs: int | None = None
    batch_size: int | None = None
    total_steps: int | None = None
    steps_per_epoch: int | None = None


# ------------------------------------------------------------------
# Complete experiment configuration
# ------------------------------------------------------------------

@dataclass
class Params:
    """
    Full experiment configuration.

    Combines:
      - Random seed
      - Data configuration
      - Model configuration
      - Training configuration

    This object fully describes one experiment run and can be
    serialized for reproducibility.
    """

    seed: int
    data: DataParams
    model_params: ModelParams
    training_params: TrainingParams