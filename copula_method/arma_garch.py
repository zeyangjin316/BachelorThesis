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


# ---------------- helpers ----------------

def _pick_target_column(df: pd.DataFrame, user_col: Optional[str]) -> str:
    """
    Select the target column from a DataFrame.

    Preference order:
      1) user-provided column if present
      2) 'ret_crsp'
      3) one of ['value','y','ret','return','price']
      4) first numeric column

    Raises
    ------
    ValueError
        If no numeric column can be found.
    """
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
    """
    Normalize a GARCH order to (p, o, q).

    Parameters
    ----------
    order : tuple | list
        Either (p, q) or (p, o, q).
    needs_o : bool
        If True and (p, q) is given, inject o=1; else inject o=0.

    Returns
    -------
    tuple[int, int, int]
        (p, o, q) triple.

    Raises
    ------
    ValueError
        If the input cannot be interpreted.
    """
    if isinstance(order, (list, tuple)):
        if len(order) == 3:
            p, o, q = map(int, order)
            return p, o, q
        if len(order) == 2:
            p, q = map(int, order)
            return (p, 1, q) if needs_o else (p, 0, q)
    raise ValueError("garch_order must be (p,q) or (p,o,q)")


def _build_vol_model(name: str, order):
    """
    Construct a volatility process for ARCH models.

    Supports:
      - sGARCH / GARCH
      - GJR-GARCH (via o>0)
      - EGARCH
    """
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
    """
    Construct a distribution object for the ARCH model.

    Supports: Normal, Student's t, GED.
    """
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
    ARMA(p, q) with (s)GARCH/GJR/EGARCH residuals.

    Expects these attributes on `model_params`:
      - arma_order: tuple[int, int]
      - include_mean: bool
      - arma_maxiter: int
      - on_nonconverge: {"warn","drop_ma","drop_ar"}
      - variance_model: {"sGARCH","GJRGARCH","EGARCH", ...}
      - garch_order: tuple[int, int] or (p,o,q)
      - dist: {"norm","std","ged"}
      - garch_scale: {"auto" | float}
      - garch_target_std: float (used when garch_scale="auto")
      - suppress_convergence_warnings: bool
      - target_col: Optional[str]
    """

    def __init__(self, data: pd.DataFrame, model_params: Any):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Long format with at least ['date','sym_root', target_col].
        model_params : Any
            Dataclass-like object with the fields listed in the class docstring.
        """
        super().__init__(data, model_params)

    # ---- internal: fit one series ----
    def _fit_one(self, series: pd.Series) -> Dict[str, Any]:
        """
        Fit ARMA to the series, then ARCH family to the residuals.

        Returns
        -------
        dict
            Keys: 'arma', 'garch', 'n_obs', 'garch_scale'.
        """
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
                    arma_res = SARIMAX(
                        ts, order=(p, 0, 0), trend=trend,
                        enforce_stationarity=False, enforce_invertibility=False,
                        concentrate_scale=True
                    ).fit(disp=False, method="lbfgs", maxiter=300)
                elif fb == "drop_ar" and p > 0:
                    arma_res = SARIMAX(
                        ts, order=(0, 0, q), trend=trend,
                        enforce_stationarity=False, enforce_invertibility=False,
                        concentrate_scale=True
                    ).fit(disp=False, method="lbfgs", maxiter=300)
            except Exception:
                pass
            if arma_res is None:
                raise RuntimeError(f"ARMA({p},{q}) could not be estimated for this series.")

        resid = arma_res.resid.astype(float)

        # Residual scaling for stable ARCH estimation; used to map forecasts back.
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
        """
        Fit ARMA+GARCH per symbol on the provided data.

        Parameters
        ----------
        current_day : pd.Timestamp | str | None
            Label for logging (e.g., target test day).
        """
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
                log.info("[%s] %s std=%.6g n=%d", sym, target_col, np.std(y, ddof=1), y.size)
            try:
                fitted[sym] = self._fit_one(sdf[target_col])
            except Exception as e:
                log.error("[ERROR] Failed to fit model for %s on day %s: %s", sym, current_day, e)

        self.fitted_models = fitted
        log.info("Fitted ARMA+GARCH for %d symbols at %s", len(fitted), current_day)

    def sample(self, symbol: str, n_samples: int = 1000) -> np.ndarray:
        """
        Draw 1-step predictive samples for a fitted symbol.

        Parameters
        ----------
        symbol : str
            Asset symbol (must have a fitted model).
        n_samples : int, default=1000
            Number of draws.

        Returns
        -------
        np.ndarray
            Samples from N(mu_1, sigma_1^2), where mu_1 is the
            ARMA 1-step mean and sigma_1 from the GARCH forecast
            mapped back by the stored scale.
        """
        if symbol not in self.fitted_models:
            raise KeyError(f"No fitted model for symbol '{symbol}'")
        m = self.fitted_models[symbol]

        # ARMA 1-step mean
        mu = float(m["arma"].get_forecast(steps=1).predicted_mean.iloc[-1])

        # ARCH variance forecast (shape-agnostic extraction)
        fc = m["garch"].forecast(horizon=1)
        v = getattr(fc, "variance", fc)
        var_arr = np.asarray(getattr(v, "values", v))
        var1_scaled = float(var_arr.ravel()[-1])

        # Map back from scaled-residual variance to original scale
        scale = float(m.get("garch_scale", 1.0))
        sigma = (var1_scaled ** 0.5) / scale

        return np.random.normal(loc=mu, scale=sigma, size=int(n_samples))
