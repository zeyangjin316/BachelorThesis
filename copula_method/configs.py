from dataclasses import dataclass
from typing import Any
from datetime import datetime

@dataclass
class TSDataConfig:
    """
    Data settings for the Two-Step copula pipeline.

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
class TSInitConfig:
    """
    Initialization settings for the Two-Step copula pipeline.

    Parameters
    ----------
    univariate_type : str, default="ARMAGARCH"
        Type of univariate model to fit per asset (e.g., "ARMAGARCH").
    copula_type : str, default="gaussian"
        Copula family used for dependence modeling
        ("gaussian", "t", "skewed-t").
    copula_params : dict, default=None
        Optional copula hyperparameters (e.g., degrees of freedom).
    rolling_window_size : int, default=1
        Size of the rolling window for model updates.
    copula_refit_freq : int, default=30
        Frequency (in days) to re-estimate the copula.
    uv_fit_percentage : float, default=0.2
        Fraction of univariate data used for each fit.
    uv_refit_freq : int, default=7
        Frequency (in days) to re-fit univariate models.
    """
    univariate_type: str = "ARMAGARCH"
    copula_type: str = "gaussian"
    copula_params: dict[str, Any] | None = None
    rolling_window_size: int = 1
    copula_refit_freq: int = 30
    uv_fit_percentage: float = 0.2
    uv_refit_freq: int = 7


@dataclass
class TSFitConfig:
    """
    Fitting settings for univariate ARMA–GARCH models.

    Parameters
    ----------
    arma_order : tuple[int, int], default=(1,1)
        AR and MA orders for the ARMA component.
    include_mean : bool, default=True
        Whether to include a mean term in the ARMA model.
    arma_maxiter : int, default=600
        Maximum iterations for ARMA fitting.
    on_nonconverge : str, default="drop_ma"
        Strategy if convergence fails ("drop_ma" or "ignore").
    variance_model : str, default="sGARCH"
        Type of GARCH model ("sGARCH", "eGARCH", etc.).
    garch_order : tuple[int, int], default=(1,1)
        GARCH(p,q) orders.
    dist : str, default="norm"
        Distribution for innovations ("norm", "t", etc.).
    garch_scale : str, default="auto"
        Scaling of variance ("auto" or fixed).
    garch_target_std : float, default=10.0
        Target standard deviation for scaling.
    suppress_convergence_warnings : bool, default=True
        If True, hides warnings for non-converged fits.
    """
    arma_order: tuple[int, int] = (1, 1)
    include_mean: bool = True
    arma_maxiter: int = 600
    on_nonconverge: str = "drop_ma"
    variance_model: str = "sGARCH"
    garch_order: tuple[int, int] = (1, 1)
    dist: str = "norm"
    garch_scale: str = "auto"
    garch_target_std: float = 10.0
    suppress_convergence_warnings: bool = True


@dataclass
class TSSampleConfig:
    """
    Sampling settings for the Two-Step copula pipeline.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of joint samples to generate.
    n_samples_uv : int, default=1000
        Number of samples per univariate model.
    """
    n_samples: int = 1000
    n_samples_uv: int = 1000
