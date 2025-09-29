import logging
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.special import erfinv
from scipy.stats import norm, t as student_t

logger = logging.getLogger(__name__)


class CopulaBase(ABC):
    """
    Abstract base for copulas in the two-step framework.

    Subclasses must implement:
      - fit_from_uv_samples(...): calibrate dependence parameters using W2 PITs
      - sample_uniforms(n_samples, ...): draw U ∈ (0,1)^{m×n} from the fitted copula
    """
    def __init__(self, n_dim: int):
        self.n_dim = int(n_dim)

    @abstractmethod
    def fit_from_uv_samples(
        self,
        *,
        full_data: pd.DataFrame,
        uv_samples: Dict[pd.Timestamp, Dict[str, np.ndarray]],
        symbols: List[str],
        day: pd.Timestamp,
        target_col: str = "ret_crsp",
        **kwargs,
    ) -> None:
        """Fit copula parameters from W2 univariate samples (via PITs)."""

    @abstractmethod
    def sample_uniforms(
        self,
        n_samples: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Draw copula uniforms.

        Returns
        -------
        np.ndarray
            Array U with shape (m, n_samples), values in (0, 1).
        """
        ...

def _pits_from_w2(
    *,
    full_data: pd.DataFrame,
    uv_samples: Dict[pd.Timestamp, Dict[str, np.ndarray]],
    symbols: List[str],
    target_col: str,
) -> np.ndarray:
    """
    Compute Gaussianized PITs across W2.

    For each date d in W2 and symbol s:
      u_{d,s} = mean( samples_{d,s} ≤ y_{d,s} ), clipped to (1e-6, 1-1e-6).
    Then z_{d,s} = Φ^{-1}(u_{d,s}) using erfinv.

    Returns
    -------
    np.ndarray
        Z with shape (|W2|, m).
    """
    w2_dates = sorted(uv_samples.keys())
    m = len(symbols)
    if not w2_dates:
        return np.zeros((0, m), dtype=float)

    Z_rows = []
    for d in w2_dates:
        day_df = full_data[full_data["date"] == d]
        u_vec = []
        for sym in symbols:
            try:
                samples = np.asarray(uv_samples[d].get(sym, np.array([])))
                if samples.size == 0:
                    raise ValueError("Empty samples")
                vals = day_df.loc[day_df["sym_root"] == sym, target_col].values
                if vals.size == 0 or not np.isfinite(vals[0]):
                    raise ValueError("Missing realized value")
                true_val = float(vals[0])
                u = float(np.mean(samples <= true_val))
                u = np.clip(u, 1e-6, 1 - 1e-6)
                u_vec.append(u)
            except Exception as e:
                logger.warning("[Copula] PIT failed for %s on %s: %s", sym, d, e)
                u_vec.append(0.5)
        z_row = np.sqrt(2.0) * erfinv(2.0 * np.asarray(u_vec) - 1.0)
        Z_rows.append(z_row)

    return np.vstack(Z_rows)  # (|W2|, m)


def _corr_from_Z(Z: np.ndarray, m: int) -> np.ndarray:
    """
    Build a correlation matrix from Gaussianized PITs with PD/symmetry guards.
    """
    if Z.shape[0] < 2:
        return np.eye(m, dtype=float)
    R = np.corrcoef(Z, rowvar=False)
    if not np.all(np.isfinite(R)):
        logger.warning("[Copula] Non-finite entries in correlation; applying nan_to_num.")
        R = np.nan_to_num(R)
    np.fill_diagonal(R, 1.0)
    return (R + R.T) / 2.0


class GaussianCopula(CopulaBase):
    """Standard Gaussian copula with correlation matrix R."""
    def __init__(self, n_dim: int):
        super().__init__(n_dim)
        self.R: Optional[np.ndarray] = None

    def fit_from_uv_samples(
        self,
        *,
        full_data: pd.DataFrame,
        uv_samples: Dict[pd.Timestamp, Dict[str, np.ndarray]],
        symbols: List[str],
        day: pd.Timestamp,
        target_col: str = "ret_crsp",
        **kwargs,
    ) -> None:
        Z = _pits_from_w2(full_data=full_data, uv_samples=uv_samples, symbols=symbols, target_col=target_col)
        self.R = _corr_from_Z(Z, self.n_dim)

    def sample_uniforms(
        self,
        n_samples: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if self.R is None:
            raise RuntimeError("GaussianCopula not fitted.")
        rng = random_state or np.random.default_rng()
        Z = rng.multivariate_normal(mean=np.zeros(self.n_dim), cov=self.R, size=int(n_samples)).T  # (m, n)
        return norm.cdf(Z)


class StudentTCopula(CopulaBase):
    """
    Symmetric Student-t copula with correlation matrix R and degrees of freedom df.
    (This is not the skewed-t variant.)
    """
    def __init__(self, n_dim: int, df: float = 6.0):
        super().__init__(n_dim)
        if df <= 2:
            logger.warning("df <= 2 gives infinite variance; consider df > 2. Using df=%s", df)
        self.R: Optional[np.ndarray] = None
        self.df: float = float(df)

    def fit_from_uv_samples(
        self,
        *,
        full_data: pd.DataFrame,
        uv_samples: Dict[pd.Timestamp, Dict[str, np.ndarray]],
        symbols: List[str],
        day: pd.Timestamp,
        target_col: str = "ret_crsp",
        **kwargs,
    ) -> None:
        Z = _pits_from_w2(full_data=full_data, uv_samples=uv_samples, symbols=symbols, target_col=target_col)
        self.R = _corr_from_Z(Z, self.n_dim)

    def sample_uniforms(
        self,
        n_samples: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if self.R is None:
            raise RuntimeError("StudentTCopula not fitted.")
        rng = random_state or np.random.default_rng()
        # Normal / chi-square mixture
        Z = rng.multivariate_normal(mean=np.zeros(self.n_dim), cov=self.R, size=int(n_samples)).T
        S = rng.chisquare(df=self.df, size=int(n_samples)) / self.df
        T = Z / np.sqrt(S)[None, :]
        return student_t.cdf(T, df=self.df)


class SkewedTOhPattonCopula(CopulaBase):
    """
    Oh & Patton (2023)-style skewed-t copula (no factors, common ζ shift).

    Parameters
    ----------
    df : float
        Degrees of freedom (> 2 recommended).
    zeta : float
        Skewness parameter (ζ). Negative -> stronger lower-tail dependence.
    shrink_eps : float
        Ridge added to R during Cholesky fallback if needed.
    """
    def __init__(self, n_dim: int, df: float = 6.0, zeta: float = -0.1, shrink_eps: float = 1e-8):
        super().__init__(n_dim)
        self.df = float(df)
        self.zeta = float(zeta)
        self.R: Optional[np.ndarray] = None
        self.shrink_eps = float(shrink_eps)

    def fit_from_uv_samples(
        self,
        *,
        full_data: pd.DataFrame,
        uv_samples: Dict[pd.Timestamp, Dict[str, np.ndarray]],
        symbols: List[str],
        day: pd.Timestamp,
        target_col: str = "ret_crsp",
        **kwargs,
    ) -> None:
        """
        Fit correlation R from W2 PITs; estimate ζ via a small grid to match
        observed lower-vs-upper tail asymmetry in W2 UV samples. df is treated as fixed.
        """
        # 1) Correlation from W2
        Z = _pits_from_w2(full_data=full_data, uv_samples=uv_samples, symbols=symbols, target_col=target_col)
        self.R = _corr_from_Z(Z, self.n_dim)

        # No data -> keep defaults
        if Z.size == 0 or self.R is None:
            self.df = getattr(self, "df", 5.0)
            self.zeta = getattr(self, "zeta", -0.1)
            return

        # 2) Compact target U from W2 UV samples
        max_per_sym = 200
        U_rows = []
        for d in sorted(uv_samples.keys()):
            cols = []
            for sym in symbols:
                arr = np.asarray(uv_samples[d].get(sym, []), dtype=float)
                if arr.size == 0:
                    break
                cols.append(arr[:max_per_sym])
            if len(cols) == len(symbols):
                U_rows.append(np.vstack(cols).T)
        if not U_rows:
            self.df = getattr(self, "df", 5.0)
            self.zeta = getattr(self, "zeta", -0.1)
            return
        U_target = np.vstack(U_rows)

        # 3) Tail-asymmetry feature
        def _tail_asymmetry(U: np.ndarray, q: float = 0.05, rng: Optional[np.random.Generator] = None,
                            max_pairs: int = 4000) -> float:
            rng = rng or np.random.default_rng(0)
            N, m = U.shape
            if m < 2 or N == 0:
                return 0.0
            pairs = set()
            need = min(max_pairs, m * (m - 1) // 2)
            while len(pairs) < need:
                i = int(rng.integers(0, m))
                j = int(rng.integers(0, m - 1))
                if j >= i:
                    j += 1
                if i > j:
                    i, j = j, i
                pairs.add((i, j))
            pairs = list(pairs)

            ll = uu = total = 0
            thrL, thrU = q, 1.0 - q
            for (i, j) in pairs:
                ui = U[:, i]; uj = U[:, j]
                total += ui.size
                ll += np.sum((ui <= thrL) & (uj <= thrL))
                uu += np.sum((ui >= thrU) & (uj >= thrU))
            tauL = (ll / total) / q if total else 0.0
            tauU = (uu / total) / q if total else 0.0
            return float(tauL - tauU)

        rng = np.random.default_rng(123)
        feat_tgt = _tail_asymmetry(U_target, q=0.05, rng=rng)

        # 4) Estimate ζ on a tiny grid (df fixed)
        self.df = getattr(self, "df", 5.0)
        zeta_grid = kwargs.get("zeta_grid", [-0.20, -0.15, -0.10, -0.05, 0.00])

        # helpers
        def _chol_spd(A: np.ndarray) -> np.ndarray:
            try:
                return np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                eps = 1e-10
                for _ in range(6):
                    try:
                        return np.linalg.cholesky(A + eps * np.eye(A.shape[0]))
                    except np.linalg.LinAlgError:
                        eps *= 10
                w, V = np.linalg.eigh(A)
                w = np.clip(w, 1e-8, None)
                return np.linalg.cholesky((V * w) @ V.T)

        def _rank_to_uniform(X: np.ndarray) -> np.ndarray:
            m, n = X.shape
            U = np.empty_like(X, dtype=float)
            for i in range(m):
                order = np.argsort(X[i], kind="mergesort")
                ranks = np.empty(n, dtype=int); ranks[order] = np.arange(n)
                U[i] = (ranks + 0.5) / n
            return U

        L = _chol_spd(self.R)
        n_sim = int(kwargs.get("n_sim", 20000))
        E = L @ rng.standard_normal(size=(self.n_dim, n_sim))

        def _simulate_asym(zeta: float) -> float:
            G = rng.gamma(shape=self.df / 2.0, scale=1.0, size=n_sim)  # Gamma(k,1)
            W = 1.0 / G
            X = (E * np.sqrt(W)[None, :]) + (zeta * W)[None, :]
            U = _rank_to_uniform(X).T
            return _tail_asymmetry(U, q=0.05, rng=rng)

        best = (None, np.inf)
        for z in zeta_grid:
            feat_sim = _simulate_asym(z)
            loss = (feat_sim - feat_tgt) ** 2
            if loss < best[1]:
                best = (z, loss)

        self.zeta = float(best[0]) if best[0] is not None else kwargs.get("zeta_default", -0.1)

    def _chol(self) -> np.ndarray:
        """Cholesky of R with PD guards (ridge / eigen clip fallback)."""
        if self.R is None:
            raise RuntimeError("SkewedTOhPattonCopula not fitted.")
        R = np.array(self.R, dtype=float)
        try:
            return np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            lam = self.shrink_eps
            for _ in range(6):
                try:
                    return np.linalg.cholesky(R + lam * np.eye(self.n_dim))
                except np.linalg.LinAlgError:
                    lam *= 10
            w, V = np.linalg.eigh(R)
            w = np.clip(w, 1e-8, None)
            R_spd = (V * w) @ V.T
            return np.linalg.cholesky(R_spd)

    @staticmethod
    def _rank_to_uniform(X: np.ndarray) -> np.ndarray:
        """Empirical CDF transform per margin -> uniforms."""
        m, n = X.shape
        U = np.empty_like(X, dtype=float)
        for i in range(m):
            order = np.argsort(X[i], kind="mergesort")
            ranks = np.empty(n, dtype=int); ranks[order] = np.arange(n)
            U[i] = (ranks + 0.5) / n
        return U

    def sample_uniforms(
        self,
        n_samples: int,
        random_state: Optional[np.random.Generator] = None,
        df: Optional[float] = None,
        zeta: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draw uniforms from the fitted skewed-t copula.

        Parameters
        ----------
        n_samples : int
            Number of scenarios.
        random_state : np.random.Generator, optional
            RNG to use.
        df : float, optional
            Override degrees of freedom for this call.
        zeta : float, optional
            Override skewness for this call.

        Returns
        -------
        np.ndarray
            U with shape (m, n_samples) in (0,1).
        """
        if self.R is None:
            raise RuntimeError("SkewedTOhPattonCopula not fitted.")
        rng = random_state or np.random.default_rng()
        m, n = self.n_dim, int(n_samples)
        nu = float(df if df is not None else self.df)
        z = float(zeta if zeta is not None else self.zeta)

        L = self._chol()
        E = L @ rng.standard_normal(size=(m, n))

        # W ~ IG(nu/2, nu/2) via W = 1 / Gamma(nu/2, 1)
        G = rng.gamma(shape=nu / 2.0, scale=1.0, size=n)
        W = 1.0 / G

        # X = sqrt(W) * E + z * W (common shift per draw)
        X = (E * np.sqrt(W)[None, :]) + (z * W)[None, :]

        return self._rank_to_uniform(X)


class CopulaFactory:
    """
    Small factory for copula instantiation by name.
    """
    @staticmethod
    def create(
        copula_type: str,
        n_dim: int,
        copula_params: Optional[dict] = None,
    ) -> CopulaBase:
        p = dict(copula_params or {})
        ctype = (copula_type or "Gaussian").lower()
        if ctype in ("gaussian",):
            return GaussianCopula(n_dim)
        if ctype in ("student-t", "studentt", "t", "student"):
            df = float(p.get("df", 6.0))
            return StudentTCopula(n_dim, df=df)
        if ctype in ("skewed-t", "skewedt", "skewt"):
            df = float(p.get("df", 6.0))
            zeta = float(p.get("zeta", -0.1))
            return SkewedTOhPattonCopula(n_dim, df=df, zeta=zeta)
        raise ValueError(
            f"Unsupported copula_type '{copula_type}'. "
            "Choose from 'gaussian', 'student-t', or 'skewed-t'."
        )
