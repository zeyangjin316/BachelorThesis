from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable

ANNUALIZE = 252  # scaling factor for annualized realized variance

def add_daily_variance_features(
    df: pd.DataFrame,
    return_col: str = "ret_crsp",
    har_windows: Iterable[int] = (5, 21, 63),
    hl_days: Iterable[int] = (1, 5, 21, 63),
    annualize: int = ANNUALIZE,
) -> pd.DataFrame:
    """
    Compute daily realized variance, semivariance, and volatility measures
    from daily returns, and extend them with HAR-style rolling averages
    and exponentially weighted moving averages (EWMA).
    """
    if return_col not in df.columns:
        raise ValueError(f"return_col '{return_col}' missing")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["sym_root", "date"])

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        r = g[return_col].astype(float)

        # Daily realized variance and semivariance
        rv_d = (r ** 2) * annualize
        semivar_d = (r.mask(r >= 0, 0.0) ** 2) * annualize

        g["rv_d"] = rv_d
        g["semivar_d"] = semivar_d

        # Volatility measures
        g["vol_d"] = np.sqrt(np.maximum(rv_d, 0.0))
        g["downvol_d"] = np.sqrt(np.maximum(semivar_d, 0.0))

        # Rolling HAR-style averages
        for w in har_windows:
            g[f"rv_roll{w}"] = rv_d.rolling(w, min_periods=1).mean()
            g[f"semivar_roll{w}"] = semivar_d.rolling(w, min_periods=1).mean()
            g[f"vol_roll{w}"] = np.sqrt(np.maximum(g[f"rv_roll{w}"], 0.0))
            g[f"downvol_roll{w}"] = np.sqrt(np.maximum(g[f"semivar_roll{w}"], 0.0))

        # Exponentially weighted averages
        for h in hl_days:
            g[f"rv_ewm_hl{h}"] = rv_d.ewm(halflife=h, adjust=False).mean()
            g[f"semivar_ewm_hl{h}"] = semivar_d.ewm(halflife=h, adjust=False).mean()
            g[f"vol_ewm_hl{h}"] = np.sqrt(np.maximum(g[f"rv_ewm_hl{h}"], 0.0))
            g[f"downvol_ewm_hl{h}"] = np.sqrt(np.maximum(g[f"semivar_ewm_hl{h}"], 0.0))

        return g

    return out.groupby("sym_root", group_keys=False).apply(_per_symbol)
