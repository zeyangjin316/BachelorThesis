import logging
from typing import Dict, Any, Optional, Tuple, Callable, Type, Union
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MODEL_REGISTRY: Dict[str, Type["BaseUVModel"]] = {}

def register_uv_model(name: str) -> Callable[[Type["BaseUVModel"]], Type["BaseUVModel"]]:
    """Decorator to register a UV model class under a method name (case-insensitive)."""
    def _dec(cls: Type["BaseUVModel"]) -> Type["BaseUVModel"]:
        _MODEL_REGISTRY[name.upper()] = cls
        return cls
    return _dec

def create_uv_model(method: str, data: pd.DataFrame, model_params: Optional[Any] = None) -> "BaseUVModel":
    """Instantiate a registered UV model, passing model_params through as-is (dataclass or any object)."""
    cls = _MODEL_REGISTRY.get(method.upper())
    if cls is None:
        raise ValueError(f"Unknown UV method '{method}'. Registered: {list(_MODEL_REGISTRY.keys())}")
    return cls(data=data, model_params=model_params)

class BaseUVModel:
    """All univariate models must implement this simple interface."""
    def __init__(self, data: pd.DataFrame, model_params: Optional[Any] = None):
        self.data = data.copy()
        # Accept dataclass / object; do not coerce to dict
        self.model_params = model_params
        self.fitted_models: Dict[str, Dict[str, Any]] = {}

    def fit(self, current_day: Union[pd.Timestamp, str] = None) -> None:
        """Fit one model per symbol using self.data up to current_day (already filtered by caller)."""
        raise NotImplementedError

    def sample(self, symbol: str, n_samples: int = 1000) -> np.ndarray:
        """Return 1-step-ahead samples for the given symbol."""
        raise NotImplementedError