import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.special import erfinv

logger = logging.getLogger(__name__)

class CopulaModel:
    """
    Gaussian copula: build a correlation matrix R_t for a single forecast day `day`
    from per-day UV samples in the *preceding* dates (W2).
    """

    def __init__(self, copula_type: str = "Gaussian"):
        if copula_type != "Gaussian":
            raise ValueError(f"Copula type '{copula_type}' not supported. Only 'Gaussian'.")
        self.corr_matrices: Dict[pd.Timestamp, np.ndarray] = {}  # optional cache {day: R_t}

    def calc_matrix_for_day(
        self,
        full_data: pd.DataFrame,
        uv_samples: Dict[pd.Timestamp, Dict[str, np.ndarray]],  # keys are W2 dates
        symbols: List[str],
        day: pd.Timestamp,
        target_col: str = "ret_crsp",
    ) -> np.ndarray:
        """
        Parameters
        ----------
        full_data : DataFrame with columns ['date','sym_root', target_col]
        uv_samples : dict mapping each W2 date -> {symbol -> np.ndarray[samples]}
        symbols : list of symbols (column order for the matrix)
        day : forecast day t (used only as key for caching/logging)
        target_col : column name of realized values used for PITs

        Returns
        -------
        R_t : np.ndarray (m x m) correlation matrix for the Gaussian copula on `day`.
        """
        w2_dates = sorted(uv_samples.keys())
        m = len(symbols)

        if not w2_dates:
            logger.warning(f"[Copula] Empty uv_samples for {day}; returning identity.")
            R_t = np.eye(m, dtype=float)
            self.corr_matrices[pd.Timestamp(day)] = R_t
            return R_t

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
                    u = np.clip(u, 1e-6, 1 - 1e-6)  # numerical guard
                    u_vec.append(u)
                except Exception as e:
                    logger.warning(f"[Copula] PIT failed for {sym} on {d}: {e}")
                    u_vec.append(0.5)

            # Gaussianize PITs: z = Φ^{-1}(u) = sqrt(2) * erfinv(2u - 1)
            z_row = np.sqrt(2.0) * erfinv(2.0 * np.asarray(u_vec) - 1.0)
            Z_rows.append(z_row)

        Z = np.vstack(Z_rows)  # shape (|W2|, m)
        if Z.shape[0] < 2:
            R_t = np.eye(m, dtype=float)
        else:
            R_t = np.corrcoef(Z, rowvar=False)
            if not np.all(np.isfinite(R_t)):
                logger.warning(f"[Copula] Non-finite entries in correlation for {day}; applying nan_to_num.")
                R_t = np.nan_to_num(R_t)
            # ensure symmetry and unit diagonal
            np.fill_diagonal(R_t, 1.0)
            R_t = (R_t + R_t.T) / 2.0

        self.corr_matrices[pd.Timestamp(day)] = R_t
        return R_t

