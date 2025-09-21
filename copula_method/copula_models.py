import logging
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.special import erfinv
from scipy.stats import norm, t as student_t

logger = logging.getLogger(__name__)


# ============== Abstract Base ==============

class CopulaBase(ABC):
    """
    Abstract base class for copulas used in the two-step framework.
    Subclasses must implement:
      - fit_from_uv_samples: calibrate dependence parameters from W2 PITs
      - sample_uniforms: draw copula uniforms U \in [0,1]^{m x n}
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
        ...

    @abstractmethod
    def sample_uniforms(
        self,
        n_samples: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Returns
        -------
        U : np.ndarray with shape (m, n_samples), values in (0,1)
        """
        ...


# ============== Utilities ==============

def _pits_from_w2(
    *,
    full_data: pd.DataFrame,
    uv_samples: Dict[pd.Timestamp, Dict[str, np.ndarray]],
    symbols: List[str],
    target_col: str,
) -> np.ndarray:
    """
    Build matrix Z (|W2| x m) by computing PITs from the UV samples, then Gaussianize.
    Returns the Gaussianized matrix Z whose columns are per-symbol.
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
                # numerical guard
                u = np.clip(u, 1e-6, 1 - 1e-6)
                u_vec.append(u)
            except Exception as e:
                logger.warning(f"[Copula] PIT failed for {sym} on {d}: {e}")
                u_vec.append(0.5)
        # Gaussianize PITs: z = Φ^{-1}(u) = sqrt(2) * erfinv(2u - 1)
        z_row = np.sqrt(2.0) * erfinv(2.0 * np.asarray(u_vec) - 1.0)
        Z_rows.append(z_row)

    Z = np.vstack(Z_rows)  # shape (|W2|, m)
    return Z


def _corr_from_Z(Z: np.ndarray, m: int) -> np.ndarray:
    """Safe correlation from Gaussianized PIT matrix."""
    if Z.shape[0] < 2:
        return np.eye(m, dtype=float)
    R = np.corrcoef(Z, rowvar=False)
    if not np.all(np.isfinite(R)):
        logger.warning("[Copula] Non-finite entries in correlation; applying nan_to_num.")
        R = np.nan_to_num(R)
    np.fill_diagonal(R, 1.0)
    R = (R + R.T) / 2.0
    return R


# ============== Concrete Copulas ==============

class GaussianCopula(CopulaBase):
    """
    Standard Gaussian copula with correlation matrix R.
    """
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
        Z = rng.multivariate_normal(mean=np.zeros(self.n_dim), cov=self.R, size=int(n_samples)).T  # (m,n)
        U = norm.cdf(Z)
        return U


class StudentTCopula(CopulaBase):
    """
    Symmetric (unskewed) Student t copula with correlation matrix R and df nu.
    NOTE: This is *not* the skewed-t variant; it provides a baseline t-copula.
    """
    def __init__(self, n_dim: int, df: float = 6.0):
        super().__init__(n_dim)
        if df <= 2:
            logger.warning("df <= 2 gives infinite variance; consider df>2. Using df=%s", df)
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
        # Multivariate t sampling via normal/chi-squared mixture
        # z ~ N(0, R), s ~ Chi2(df)/df, t = z / sqrt(s)
        Z = rng.multivariate_normal(mean=np.zeros(self.n_dim), cov=self.R, size=int(n_samples)).T  # (m,n)
        S = rng.chisquare(df=self.df, size=int(n_samples)) / self.df  # (n,)
        T = Z / np.sqrt(S)[None, :]
        U = student_t.cdf(T, df=self.df)
        return U


class SkewedTOhPattonCopula(CopulaBase):
    """
    Oh & Patton (2023)-style skewed-t copula (no factors, common ζ shift).
    Parameters
    ----------
    df : float
        Degrees of freedom (>2 recommended).
    zeta : float
        Skewness parameter (ζ). Negative -> stronger lower-tail dependence.
    shrink_eps : float
        Small ridge for PD safety if needed.
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
        Fit R from W2 PITs (as usual), then estimate zeta via a tiny 1-D grid
        that matches lower-vs-upper tail asymmetry on W2 UV samples.
        df (self.df) is treated as global/fixed (default to 5.0 if unset).
        """

        # ----- 1) Correlation from W2 (same as Gaussian/t) -----
        Z = _pits_from_w2(
            full_data=full_data,
            uv_samples=uv_samples,
            symbols=symbols,
            target_col=target_col,
        )
        self.R = _corr_from_Z(Z, self.n_dim)

        # Guard: nothing to fit if no W2
        if Z.size == 0 or self.R is None:
            self.df = getattr(self, "df", 5.0)
            self.zeta = getattr(self, "zeta", -0.1)
            return

        # ----- 2) Build a compact "target U" matrix from W2 uv_samples -----
        # Stack a small slice of per-symbol UV forecast samples across W2 dates.
        # Shape will be (N_obs, m). Keep it light for speed.
        max_per_sym = 200  # cap per-date, per-symbol for speed
        U_rows = []
        for d in sorted(uv_samples.keys()):
            cols = []
            for sym in symbols:
                arr = np.asarray(uv_samples[d].get(sym, []), dtype=float)
                if arr.size == 0:
                    break
                cols.append(arr[:max_per_sym])
            if len(cols) == len(symbols):
                # cols: list of (<=max_per_sym,) for each symbol -> stack to (m, n_d) then T
                U_rows.append(np.vstack(cols).T)
        if not U_rows:
            self.df = getattr(self, "df", 5.0)
            self.zeta = getattr(self, "zeta", -0.1)
            return
        U_target = np.vstack(U_rows)  # (N_obs, m)

        # ----- 3) Tail asymmetry feature on target -----
        def _tail_asymmetry(U: np.ndarray, q: float = 0.05, rng: Optional[np.random.Generator] = None,
                            max_pairs: int = 4000) -> float:
            """
            Simple co-exceedance asymmetry: [C(q,q)/q] - [C(1-q,1-q)/q]
            computed over a random subset of pairs for speed.
            U: (N_obs, m)
            """
            rng = rng or np.random.default_rng(0)
            N, m = U.shape
            if m < 2 or N == 0:
                return 0.0
            # sample random pairs (i<j)
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

            ll = uu = 0
            total = 0
            thrL, thrU = q, 1.0 - q
            for (i, j) in pairs:
                ui = U[:, i];
                uj = U[:, j]
                total += ui.size
                ll += np.sum((ui <= thrL) & (uj <= thrL))
                uu += np.sum((ui >= thrU) & (uj >= thrU))
            tauL = (ll / total) / q if total else 0.0
            tauU = (uu / total) / q if total else 0.0
            return float(tauL - tauU)

        rng = np.random.default_rng(123)
        feat_tgt = _tail_asymmetry(U_target, q=0.05, rng=rng)

        # ----- 4) Estimate zeta via tiny 1-D grid (df fixed) -----
        # Keep df global (if not set, default 5). Let zeta capture asymmetry.
        self.df = getattr(self, "df", 5.0)
        zeta_grid = kwargs.get("zeta_grid", [-0.20, -0.15, -0.10, -0.05, 0.00])

        # Utilities: Cholesky (SPD guard) + rank->uniform + fast sampler
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
                # eigen clip fallback
                w, V = np.linalg.eigh(A)
                w = np.clip(w, 1e-8, None)
                return np.linalg.cholesky((V * w) @ V.T)

        def _rank_to_uniform(X: np.ndarray) -> np.ndarray:
            # X: (m, n) -> U: (m, n)
            m, n = X.shape
            U = np.empty_like(X, dtype=float)
            for i in range(m):
                order = np.argsort(X[i], kind="mergesort")
                ranks = np.empty(n, dtype=int);
                ranks[order] = np.arange(n)
                U[i] = (ranks + 0.5) / n
            return U

        L = _chol_spd(self.R)
        n_sim = int(kwargs.get("n_sim", 20000))  # keep small; good enough
        # Cache Gaussian core E across zetas for speed
        E = L @ rng.standard_normal(size=(self.n_dim, n_sim))  # (m, n)

        def _simulate_asym(zeta: float) -> float:
            # W ~ IG(nu/2, nu/2) -> sample as 1 / Gamma(k, 1) with k = nu/2
            G = rng.gamma(shape=self.df / 2.0, scale=1.0, size=n_sim)
            W = 1.0 / G
            X = (E * np.sqrt(W)[None, :]) + (zeta * W)[None, :]
            U = _rank_to_uniform(X).T  # (n, m)
            return _tail_asymmetry(U, q=0.05, rng=rng)

        best = (None, np.inf)
        for z in zeta_grid:
            feat_sim = _simulate_asym(z)
            loss = (feat_sim - feat_tgt) ** 2
            if loss < best[1]:
                best = (z, loss)

        # ----- 5) Store final zeta (fallback if needed) -----
        self.zeta = float(best[0]) if best[0] is not None else kwargs.get("zeta_default", -0.1)

    def _chol(self) -> np.ndarray:
        if self.R is None:
            raise RuntimeError("SkewedTOhPattonCopula not fitted.")
        R = np.array(self.R, dtype=float)
        # PD guard
        try:
            return np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            lam = self.shrink_eps
            for _ in range(6):
                try:
                    return np.linalg.cholesky(R + lam * np.eye(self.n_dim))
                except np.linalg.LinAlgError:
                    lam *= 10
            # eigen clip fallback
            w, V = np.linalg.eigh(R)
            w = np.clip(w, 1e-8, None)
            R_spd = (V * w) @ V.T
            return np.linalg.cholesky(R_spd)

    @staticmethod
    def _rank_to_uniform(X: np.ndarray) -> np.ndarray:
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
        # you can pass per-call overrides if you want:
        df: Optional[float] = None,
        zeta: Optional[float] = None,
    ) -> np.ndarray:
        if self.R is None:
            raise RuntimeError("SkewedTOhPattonCopula not fitted.")
        rng = random_state or np.random.default_rng()
        m, n = self.n_dim, int(n_samples)
        nu = float(df if df is not None else self.df)
        z = float(zeta if zeta is not None else self.zeta)

        # Correlated normal part: L @ N(0,I)
        L = self._chol()
        E = L @ rng.standard_normal(size=(m, n))

        # Mixing variable W ~ IG(nu/2, nu/2) -> draw as 1 / Gamma(nu/2, nu/2)
        # numpy gamma uses shape k and scale θ; here scale = 2/nu for chi-square? We want IG(k, k): W = 1 / G, G ~ Gamma(k, 1)
        G = rng.gamma(shape=nu/2.0, scale=1.0, size=n)  # Gamma(k,1)
        W = 1.0 / G                                    # IG(k,k) up to constant scaling; only ratios matter for copula

        # Apply skewed-t structure: X = sqrt(W)*E + zeta*W (common shift across margins per scenario)
        X = (E * np.sqrt(W)[None, :]) + (z * W)[None, :]

        # Copula uniforms via empirical CDF (rank transform)
        U = self._rank_to_uniform(X)
        return U



# ============== Factory / Public API ==============

class CopulaFactory:
    @staticmethod
    def create(
        copula_type: str,
        n_dim: int,
        copula_params: Optional[dict] = None,
    ) -> CopulaBase:
        p = dict(copula_params or {})
        ctype = (copula_type or "Gaussian").lower()
        if ctype in ("gaussian"):
            return GaussianCopula(n_dim)
        if ctype in ("student-t"):
            df = float(p.get("df", 6.0))
            return StudentTCopula(n_dim, df=df)
        if ctype in ("skewed-t"):
            df = float(p.get("df", 6.0))
            zeta = float(p.get("zeta", -0.1))
            return SkewedTOhPattonCopula(n_dim, df=df, zeta=zeta)
        raise ValueError(f"Unsupported copula_type '{copula_type}'. "
                         "Choose from 'gaussian', 'student-t', or 'skewed-t'.")


"""class CopulaModel:
    def __init__(self, copula_type: str = "Gaussian"):
        if copula_type != "Gaussian":
            raise ValueError(f"Copula type '{copula_type}' not supported by legacy API. Use CopulaFactory instead.")
        self.corr_matrices: Dict[pd.Timestamp, np.ndarray] = {}

    def calc_matrix_for_day(
        self,
        full_data: pd.DataFrame,
        uv_samples: Dict[pd.Timestamp, Dict[str, np.ndarray]],
        symbols: List[str],
        day: pd.Timestamp,
        target_col: str = "ret_crsp",
    ) -> np.ndarray:
        Z = _pits_from_w2(full_data=full_data, uv_samples=uv_samples, symbols=symbols, target_col=target_col)
        R = _corr_from_Z(Z, len(symbols))
        self.corr_matrices[pd.Timestamp(day)] = R
        return R"""
