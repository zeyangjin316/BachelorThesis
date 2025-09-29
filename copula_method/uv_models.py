import logging
from typing import Dict, Any, Optional, Tuple, Callable, Type, Union
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Global registry: method name (upper) -> class
_MODEL_REGISTRY: Dict[str, Type["BaseUVModel"]] = {}


def register_uv_model(name: str) -> Callable[[Type["BaseUVModel"]], Type["BaseUVModel"]]:
    """
    Class decorator to register a univariate (UV) model under a method name.

    Parameters
    ----------
    name : str
        Lookup key (case-insensitive). Example: "ARMAGARCH".

    Returns
    -------
    Callable
        Decorator that registers the class in _MODEL_REGISTRY.
    """
    def _dec(cls: Type["BaseUVModel"]) -> Type["BaseUVModel"]:
        _MODEL_REGISTRY[name.upper()] = cls
        return cls
    return _dec


def create_uv_model(method: str, data: pd.DataFrame, model_params: Optional[Any] = None) -> "BaseUVModel":
    """
    Instantiate a registered UV model.

    Parameters
    ----------
    method : str
        Name used at registration time (case-insensitive).
    data : pd.DataFrame
        Long panel for a single window; at least ['date','sym_root', target].
    model_params : Any, optional
        Dataclass-like configuration object passed through as-is.

    Returns
    -------
    BaseUVModel
        Bound instance ready to fit/sample.

    Raises
    ------
    ValueError
        If the method is unknown (not registered).
    """
    cls = _MODEL_REGISTRY.get(method.upper())
    if cls is None:
        raise ValueError(f"Unknown UV method '{method}'. Registered: {list(_MODEL_REGISTRY.keys())}")
    return cls(data=data, model_params=model_params)


class BaseUVModel:
    """
    Minimal interface for univariate models used by the two-step pipeline.

    Subclasses must implement:
      - fit(current_day): train per symbol on self.data (already filtered)
      - sample(symbol, n_samples): draw 1-step-ahead samples for a symbol
    """
    def __init__(self, data: pd.DataFrame, model_params: Optional[Any] = None):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Windowed long data with columns including ['date','sym_root', target].
        model_params : Any, optional
            Config object (dataclass or similar); stored unmodified.
        """
        self.data = data.copy()
        self.model_params = model_params
        self.fitted_models: Dict[str, Dict[str, Any]] = {}

    def fit(self, current_day: Union[pd.Timestamp, str, None] = None) -> None:
        """Train per-symbol models using self.data up to `current_day` (handled by caller)."""
        raise NotImplementedError

    def sample(self, symbol: str, n_samples: int = 1000) -> np.ndarray:
        """Return 1-step-ahead samples for a fitted symbol."""
        raise NotImplementedError
