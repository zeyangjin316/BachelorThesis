from dataclasses import dataclass
from typing import Any

@dataclass
class CGMDataConfig:
    split_point: float | Any = 0.95
    filter_features: bool = False
    exclude_pandemic: bool = True

@dataclass
class CGMInitConfig:
    dim_latent: int = 50
    n_samples_train: int = 100
    emb_size: int = 2

@dataclass
class CGMFitConfig:
    n_epochs: int = 100
    batch_size: int = 512
    train_freq: int = 30
    train_window_size: int = 50
    learningrate: float | str = 0.001
    verbose: int = 1
    callbacks: Any = None
    validation_split: float = 0.1
    validation_data: Any = None
    sample_weight: Any = None

@dataclass
class CGMSampleConfig:
    n_samples: int = 1000
    verbose: int = 0