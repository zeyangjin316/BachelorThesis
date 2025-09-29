from dataclasses import dataclass
from typing import Any

@dataclass
class CGMDataConfig:
    """
    Data settings for the CGM pipeline.

    Parameters
    ----------
    split_point : float | Any, default=0.9
        Fraction or explicit date for train/test split.
    filter_features : bool, default=False
        Whether to reduce the set of input features.
    exclude_pandemic : bool, default=True
        Whether to drop pandemic-period data.
    """
    split_point: float | Any = 0.9
    filter_features: bool = False
    exclude_pandemic: bool = True


@dataclass
class CGMInitConfig:
    """
    Model initialization settings for CGM.

    Parameters
    ----------
    dim_latent : int, default=50
        Dimension of the latent space.
    n_samples_train : int, default=100
        Number of samples drawn per training step.
    emb_size : int, default=2
        Embedding size for categorical features (e.g., weekday).
    """
    dim_latent: int = 50
    n_samples_train: int = 100
    emb_size: int = 2


@dataclass
class CGMFitConfig:
    """
    Training settings for CGM.

    Parameters
    ----------
    n_epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=512
        Training batch size.
    train_freq : int, default=30
        Frequency (in days) for retraining in rolling windows.
    train_window_size : int, default=50
        Window size (in days) for rolling training.
    learningrate : float | str, default=0.001
        Learning rate or a learning rate schedule string.
    verbose : int, default=1
        Verbosity level (0 = silent).
    callbacks : Any, optional
        Optional training callbacks (e.g., Keras style).
    validation_split : float, default=0.1
        Fraction of training data used for validation.
    validation_data : Any, optional
        Explicit validation dataset if provided.
    sample_weight : Any, optional
        Sample weights for training.
    """
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
    """
    Sampling settings for CGM.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of forecast samples to generate.
    verbose : int, default=0
        Verbosity level during sampling.
    """
    n_samples: int = 1000
    verbose: int = 0
