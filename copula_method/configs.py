from dataclasses import dataclass
from typing import Any, Tuple

@dataclass
class TSDataConfig:
    split_point: float | Any = 0.99

@dataclass
class TSInitConfig:
    univariate_type: str = "ARMAGARCH"
    copula_type: str = "Gaussian"
    # General size of the rolling window used to calculate correlation matrices.
    rolling_window_size: float = 0.6
    copula_refit_freq: int = 1
    # part of the rolling window used for generating samples for a correlation matrix
    uv_fit_percentage: float = 0.5
    uv_refit_freq: int = 1

@dataclass
class TSFitConfig:
    arma_order: Tuple[int, int] = (1, 1)
    include_mean: bool = True
    arma_maxiter: int = 600  # more iterations for first optimizer
    on_nonconverge: str = "drop_ma" # or "drop_ar" or "warn"
    variance_model: str = "sGARCH"  # or "gjrGARCH", "eGARCH"
    garch_order: Tuple[int, int] = (1, 1)
    dist: str = "norm"
    garch_scale: str = "auto" # or fixed, e.g. 100.0
    garch_target_std: float = 10.0
    suppress_convergence_warnings: bool = True


@dataclass
class TSSampleConfig:
    n_samples: int = 1000
    n_samples_uv: int = 100