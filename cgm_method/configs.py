from dataclasses import dataclass
from typing import Any

@dataclass
class CGMDataConfig:
    split_point: float | Any = 0.99
    standardize: bool = True

@dataclass
class CGMInitConfig:
    dim_latent: int = 50
    n_samples_train: int = 100
    emb_size: int = 2

@dataclass
class CGMFitConfig:
    n_epochs: int = 100
    batch_size: int = 1024
    train_freq: int = 20
    train_window_size: int = 20
    learningrate: float | str = 0.01
    verbose: int = 1
    callbacks: Any = None
    validation_split: float = 0.0
    validation_data: Any = None
    sample_weight: Any = None

@dataclass
class CGMSampleConfig:
    n_samples: int = 1000
    verbose: int = 0