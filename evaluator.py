import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt  # kept if extended later
from typing import Iterable, Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """
    Evaluate multivariate forecast samples against realized returns.

    Expectations
    ------------
    samples : np.ndarray, shape (T, S, N)
        T forecast days, S assets (fixed order), N samples per asset/day.
    test_set : pd.DataFrame
        Long-format test data with at least ['date', 'sym_root', 'ret_crsp'].
    asset_order : list[str] | None
        Order of assets along axis=1 in `samples`. If None, uses sorted unique
        symbols from `test_set`. For robustness you should pass the exact model order.
    """

    def __init__(self, test_set: pd.DataFrame, samples: np.ndarray, asset_order=None):
        """
        Parameters
        ----------
        test_set : pd.DataFrame
            Realized returns, long format.
        samples : np.ndarray
            Forecast samples, shape (T, S, N).
        asset_order : list[str] | None, default=None
            Asset order for the S dimension of `samples`.
        """
        self.test_set = test_set.copy()
        self.samples = np.asarray(samples)
        self.asset_order = asset_order or sorted(self.test_set['sym_root'].unique())
        self.daily_scores: Optional[pd.DataFrame] = None

        # basic checks
        if self.samples.ndim != 3:
            raise ValueError(f"`samples` must be 3D (T,S,N). Got {self.samples.shape}.")
        if not np.isfinite(self.samples).all():
            bad = np.isnan(self.samples).sum()
            inf = np.isinf(self.samples).sum()
            raise ValueError(f"`samples` contains non-finite values (NaNs={bad}, Infs={inf}).")

        symbols_in_test = set(self.test_set['sym_root'].unique())
        missing = [s for s in self.asset_order if s not in symbols_in_test]
        if missing:
            logger.warning(
                "[ForecastEvaluator] asset_order has symbols not present in test_set: %s",
                missing
            )

    # ---------- CRPS (scalar, from samples) ----------
    @staticmethod
    def _crps_from_samples(y: float, s: np.ndarray) -> float:
        """
        CRPS estimator for scalar y with i.i.d. samples s:

            (1/n) * sum |s_i - y|  -  (1/(2 n^2)) * sum_{i,j} |s_i - s_j|

        O(n log n) implementation for the pairwise term.

        Parameters
        ----------
        y : float
            Realized value.
        s : np.ndarray
            1D array of samples.

        Returns
        -------
        float
            CRPS value (np.nan if invalid input).
        """
        s = np.asarray(s, dtype=float)
        n = s.size
        if n == 0 or not np.isfinite(y) or not np.isfinite(s).all():
            return np.nan
        term1 = np.mean(np.abs(s - y))
        s_sorted = np.sort(s)
        idx = np.arange(n, dtype=float)
        coef = (2.0 * idx - n + 1.0)
        # sum_{i,j} |s_i - s_j| == 2 * sum_k (2k - n + 1) * s_(k)
        pair_sum = 2.0 * np.sum(coef * s_sorted)
        term2 = pair_sum / (2.0 * n * n)
        return term1 - term2

    def evaluate(self, p: float = 0.5):
        """
        Compute portfolio-level daily scores (ES, VS, DSS) and per-asset means.

        Parameters
        ----------
        p : float, default=0.5
            Variogram order for VS.

        Returns
        -------
        dict
            Summary row with mean_es, mean_vs, mean_dss and per-asset mean CRPS.
        """
        from scoring_rules_supp import es_sample, vs_sample, dss_sample

        # align dates with samples from the end
        test_dates = np.array(sorted(self.test_set['date'].unique()))
        T, S, N = self.samples.shape

        if len(test_dates) > T:
            test_dates = test_dates[-T:]
            logger.info("[Evaluator] Using last %d of %d test dates to match samples.", T, len(test_dates))
        elif len(test_dates) < T:
            logger.warning(
                "[Evaluator] samples has %d days but test_set has %d. Trimming samples.",
                T, len(test_dates)
            )
            self.samples = self.samples[-len(test_dates):, :, :]
            T = len(test_dates)

        energy_scores, variogram_scores, dss_scores = [], [], []
        daily_records = []

        # per-asset lists (may include NaNs for missing days)
        per_asset = {sym: {"crps": [], "vs": [], "dss": []} for sym in self.asset_order}

        # pre-slice
        test = self.test_set[['date', 'sym_root', 'ret_crsp']]

        progress = tqdm(test_dates, desc="Evaluating Scores")
        for t, date in enumerate(progress):
            day = test[test['date'] == date]

            # realized vector aligned to asset_order
            y_vec = np.full((len(self.asset_order),), np.nan, dtype=float)
            mask = np.zeros((len(self.asset_order),), dtype=bool)
            for i, sym in enumerate(self.asset_order):
                vals = day.loc[day['sym_root'] == sym, 'ret_crsp'].values
                if vals.size and np.isfinite(vals[0]):
                    y_vec[i] = float(vals[0]); mask[i] = True

            y_pred_t = self.samples[t]  # (S, N)

            # portfolio-level (over available assets only)
            if mask.any():
                y_true_sub = y_vec[mask][None, :]       # (1, S_avail)
                y_pred_sub = y_pred_t[mask][None, :, :] # (1, S_avail, N)

                progress.set_description(f"Day {t + 1}/{len(test_dates)} — ES")
                es = es_sample(y_true_sub, y_pred_sub)
                progress.set_description(f"Day {t + 1}/{len(test_dates)} — VS")
                vs = vs_sample(y_true_sub, y_pred_sub, p=p)
                progress.set_description(f"Day {t + 1}/{len(test_dates)} — DSS")
                dss = dss_sample(y_true_sub, y_pred_sub)

                energy_scores.append(es)
                variogram_scores.append(vs)
                dss_scores.append(dss)

                daily_records.append({
                    "date": pd.to_datetime(date),
                    "es": float(es),
                    "vs": float(vs),
                    "dss": float(dss),
                })

            # per-asset metrics
            for i, sym in enumerate(self.asset_order):
                if not mask[i]:
                    per_asset[sym]["crps"].append(np.nan)
                    per_asset[sym]["vs"].append(np.nan)
                    per_asset[sym]["dss"].append(np.nan)
                    continue

                y_i = y_vec[i]
                s_i = y_pred_t[i, :]  # (N,)

                crps_i = self._crps_from_samples(y_i, s_i)

                # VS/DSS in univariate form
                y_i_arr = np.array([[y_i]])
                s_i_arr = s_i.reshape(1, 1, -1)
                vs_i = vs_sample(y_i_arr, s_i_arr, p=p)
                dss_i = dss_sample(y_i_arr, s_i_arr)

                per_asset[sym]["crps"].append(crps_i)
                per_asset[sym]["vs"].append(vs_i)
                per_asset[sym]["dss"].append(dss_i)

        # overall means
        mean_es = float(np.nanmean(energy_scores)) if energy_scores else np.nan
        mean_vs = float(np.nanmean(variogram_scores)) if variogram_scores else np.nan
        mean_dss = float(np.nanmean(dss_scores)) if dss_scores else np.nan

        if daily_records:
            self.daily_scores = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)
        else:
            self.daily_scores = pd.DataFrame(columns=["date", "es", "vs", "dss"])

        logger.info(
            "\n=== Forecast Evaluation Summary ===\n"
            "Mean Energy Score (ES):            %.6f\n"
            "Mean Variogram Score (VS, p=%.2f): %.6f\n"
            "Mean Dawid–Sebastiani Score (DSS): %.6f\n"
            "===================================",
            mean_es, p, mean_vs, mean_dss
        )

        # per-asset mean CRPS
        asset_means = {
            f"{sym}_crps": float(np.nanmean(scores["crps"])) if scores["crps"] else np.nan
            for sym, scores in per_asset.items()
        }

        summary_row = {
            "mean_es": mean_es,
            "mean_vs": mean_vs,
            "mean_dss": mean_dss,
            **asset_means
        }
        return summary_row

    def get_daily_scores(self) -> pd.DataFrame:
        """
        Return per-day portfolio-level scores computed by `evaluate()`.

        Returns
        -------
        pd.DataFrame
            Columns: ['date', 'es', 'vs', 'dss'].
        """
        if self.daily_scores is None:
            raise RuntimeError("No daily scores yet. Call evaluate() first.")
        return self.daily_scores.copy()
