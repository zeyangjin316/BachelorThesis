import logging
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from arch.univariate import ZeroMean, GARCH, EGARCH, Normal, StudentsT, GeneralizedError

from copula_method.uv_models import register_uv_model, BaseUVModel

log = logging.getLogger(__name__)

# To install needed packages: python -m pip install -r requirements.txt

# ---------------- helpers ----------------

def _pick_target_column(df: pd.DataFrame, user_col: Optional[str]) -> str:
    if user_col and user_col in df.columns:
        return user_col
    if "ret_crsp" in df.columns:
        return "ret_crsp"
    for cand in ("value", "y", "ret", "return", "price"):
        if cand in df.columns:
            return cand
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("Could not infer target column; add 'ret_crsp' or set target_col on the config.")
    return num_cols[0]

def _norm_poq(order, needs_o: bool) -> Tuple[int, int, int]:
    # Accept (p,q) or (p,o,q); inject o=0 or o=1 as needed
    if isinstance(order, (list, tuple)):
        if len(order) == 3:
            p, o, q = map(int, order)
            return p, o, q
        if len(order) == 2:
            p, q = map(int, order)
            return (p, 0, q) if not needs_o else (p, 1, q)
    raise ValueError("garch_order must be (p,q) or (p,o,q)")

def _build_vol_model(name: str, order):
    nm = (name or "sGARCH").lower()
    if nm in ("sgarch", "garch", "s-garch", "s_garch"):
        p, o, q = _norm_poq(order, needs_o=False)
        return GARCH(p=p, o=o, q=q)
    if nm in ("gjrgarch", "gjr", "tarch", "gjr-garch", "gjr_garch"):
        p, o, q = _norm_poq(order, needs_o=True)
        return GARCH(p=p, o=o, q=q)  # GJR when o>0
    if nm == "egarch":
        p, o, q = _norm_poq(order, needs_o=False)
        return EGARCH(p=p, o=o, q=q)
    p, o, q = _norm_poq(order, needs_o=False)
    return GARCH(p=p, o=o, q=q)

def _build_dist(name: str):
    nm = (name or "norm").lower()
    if nm in ("norm", "normal"):
        return Normal()
    if nm in ("std", "student", "studentst", "student-t", "student_t"):
        return StudentsT()
    if nm in ("ged", "generalizederror", "generalisederror"):
        return GeneralizedError()
    return Normal()

# =========================
#  ArmaGarchModel
# =========================

@register_uv_model("ARMAGARCH")
class ArmaGarchModel(BaseUVModel):
    """
    ARMA(p,q) + (s)GARCH/GJR/EGARCH using a dataclass config (TSFitConfig).
    Expects these attributes on `model_params`:
      arma_order, include_mean, arma_maxiter, on_nonconverge,
      variance_model, garch_order, dist, garch_scale, garch_target_std,
      suppress_convergence_warnings, (optional) target_col.
    """

    def __init__(self, data: pd.DataFrame, model_params: Any):
        super().__init__(data, model_params)

    # ---- internal: fit one series ----
    def _fit_one(self, series: pd.Series) -> Dict[str, Any]:
        ts = (
            pd.to_numeric(series, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .reset_index(drop=True)
        )

        p, q = map(int, self.model_params.arma_order)
        include_mean = bool(getattr(self.model_params, "include_mean", True))
        trend = "c" if include_mean else "n"

        if getattr(self.model_params, "suppress_convergence_warnings", True):
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

        arma = SARIMAX(
            ts,
            order=(p, 0, q),
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        )

        arma_res = None
        for opts in (
            dict(method="lbfgs", maxiter=int(getattr(self.model_params, "arma_maxiter", 600))),
            dict(method="bfgs",  maxiter=400),
            dict(method="powell",maxiter=300),
            dict(method="nm",    maxiter=300),
        ):
            try:
                res = arma.fit(disp=False, **opts)
                arma_res = res
                if getattr(res, "converged", True):
                    break
            except Exception:
                continue

        if arma_res is None:
            fb = getattr(self.model_params, "on_nonconverge", "warn")
            try:
                if fb == "drop_ma" and q > 0:
                    arma_res = SARIMAX(ts, order=(p, 0, 0), trend=trend,
                                       enforce_stationarity=False, enforce_invertibility=False,
                                       concentrate_scale=True).fit(disp=False, method="lbfgs", maxiter=300)
                elif fb == "drop_ar" and p > 0:
                    arma_res = SARIMAX(ts, order=(0, 0, q), trend=trend,
                                       enforce_stationarity=False, enforce_invertibility=False,
                                       concentrate_scale=True).fit(disp=False, method="lbfgs", maxiter=300)
            except Exception:
                pass
            if arma_res is None:
                raise RuntimeError(f"ARMA({p},{q}) could not be estimated for this series.")

        resid = arma_res.resid.astype(float)

        # Residual scaling (keeps ARCH stable; predictive draws mapped back to original scale)
        garch_scale = getattr(self.model_params, "garch_scale", "auto")
        if isinstance(garch_scale, (int, float)):
            scale = float(garch_scale)
        else:
            rstd = float(np.std(resid, ddof=1)) or 1e-12
            target_std = float(getattr(self.model_params, "garch_target_std", 10.0))
            scale = target_std / rstd
        resid_scaled = resid * scale

        vm   = getattr(self.model_params, "variance_model", "sGARCH")
        gpq  = getattr(self.model_params, "garch_order", (1, 1))
        vol  = _build_vol_model(vm, gpq)
        dist = _build_dist(getattr(self.model_params, "dist", "norm"))

        mean_model = ZeroMean(resid_scaled, rescale=False)
        mean_model.volatility   = vol
        mean_model.distribution = dist
        garch_res = mean_model.fit(update_freq=0, disp="off")

        return {
            "arma": arma_res,
            "garch": garch_res,
            "n_obs": int(ts.size),
            "garch_scale": float(scale),
        }

    def fit(self, current_day: Optional[Union[pd.Timestamp, str]]) -> None:
        target_col = _pick_target_column(self.data, getattr(self.model_params, "target_col", None))
        fitted: Dict[str, Dict[str, Any]] = {}

        for sym in self.data["sym_root"].dropna().unique():
            sdf = (
                self.data.loc[self.data["sym_root"] == sym]
                .sort_values("date")
                .dropna(subset=[target_col])
            )
            y = sdf[target_col].astype(float).values
            if y.size >= 2:
                log.info(f"[{sym}] {target_col} std={np.std(y, ddof=1):.6g} n={y.size}")
            try:
                fitted[sym] = self._fit_one(sdf[target_col])
            except Exception as e:
                log.error(f"[ERROR] Failed to fit model for {sym} on day {current_day}: {e}")

        self.fitted_models = fitted
        log.info(f"Fitted ARMA+GARCH for {len(fitted)} symbols at {current_day}")

    def sample(self, symbol: str, n_samples: int = 1000) -> np.ndarray:
        if symbol not in self.fitted_models:
            raise KeyError(f"No fitted model for symbol '{symbol}'")
        m = self.fitted_models[symbol]

        # 1-step mean from ARMA
        mu = float(m["arma"].get_forecast(steps=1).predicted_mean.iloc[-1])

        # Shape-agnostic variance extraction from arch
        fc = m["garch"].forecast(horizon=1)
        v = getattr(fc, "variance", fc)
        var_arr = np.asarray(getattr(v, "values", v))
        var1_scaled = float(var_arr.ravel()[-1])

        scale = float(m.get("garch_scale", 1.0))
        sigma = (var1_scaled ** 0.5) / scale
        return np.random.normal(loc=mu, scale=sigma, size=int(n_samples))