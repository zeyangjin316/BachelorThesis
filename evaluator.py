import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """
    Evaluate multivariate forecast samples against realized returns.

    Expectations
    ------------
    samples: np.ndarray of shape (T, S, N)
        T forecast origins (days), S assets (same order every time), N samples per asset/day.
    test_set: long DataFrame with columns ['date','sym_root','ret_crsp'] at least.
    asset_order: list[str], optional
        The order of assets along axis=1 in `samples`. If None, uses the sorted unique symbols
        in `test_set`, but you should pass the exact order used for the model.
    """

    def __init__(self, test_set: pd.DataFrame, samples: np.ndarray, asset_order=None):
        self.test_set = test_set.copy()
        self.samples = np.asarray(samples)
        self.asset_order = asset_order or sorted(self.test_set['sym_root'].unique())

        # basic sanity
        if self.samples.ndim != 3:
            raise ValueError(f"`samples` must be 3D (T,S,N). Got {self.samples.shape}.")
        if not np.isfinite(self.samples).all():
            bad = np.isnan(self.samples).sum()
            inf = np.isinf(self.samples).sum()
            raise ValueError(f"`samples` contains non-finite values (NaNs={bad}, Infs={inf}).")

        symbols_in_test = set(self.test_set['sym_root'].unique())
        missing = [s for s in self.asset_order if s not in symbols_in_test]
        if missing:
            logger.warning(f"[ForecastEvaluator] asset_order has symbols not present in test_set: {missing}")

    # ---------- CRPS (scalar, from samples) ----------
    @staticmethod
    def _crps_from_samples(y: float, s: np.ndarray) -> float:
        """
        CRPS estimator for scalar y with i.i.d. samples s:
            (1/n) * sum |s_i - y| - (1/(2n^2)) * sum_{i,j} |s_i - s_j|
        O(n log n) implementation for the pairwise term.
        """
        s = np.asarray(s, dtype=float)
        n = s.size
        if n == 0 or not np.isfinite(y) or not np.isfinite(s).all():
            return np.nan
        term1 = np.mean(np.abs(s - y))
        s_sorted = np.sort(s)
        idx = np.arange(n, dtype=float)
        coef = (2.0 * idx - n + 1.0)
        # sum_{i,j} |s_i - s_j|  ==  2 * sum_k (2k - n + 1) * s_(k)
        pair_sum = 2.0 * np.sum(coef * s_sorted)
        term2 = pair_sum / (2.0 * n * n)
        return term1 - term2

    def evaluate(self, p: float = 0.5):
        """
        Returns a dict with:
          - mean_es, mean_vs, mean_dss (portfolio-level means over time)
          - asset_scores: DataFrame with per-asset mean CRPS/VS/DSS
        """
        from scoring_rules_supp import es_sample, vs_sample, dss_sample

        # Align dates with samples length from the END (most common setup)
        test_dates = np.array(sorted(self.test_set['date'].unique()))
        T, S, N = self.samples.shape

        if len(test_dates) > T:
            # use last T dates
            test_dates = test_dates[-T:]
            logger.info(f"[Evaluator] Using last {T} of {len(test_dates)} test dates to match samples.")
        elif len(test_dates) < T:
            # trim samples from the front to match fewer dates
            logger.warning(f"[Evaluator] samples has {T} days but test_set has {len(test_dates)}. "
                           f"Trimming samples to last {len(test_dates)}.")
            self.samples = self.samples[-len(test_dates):, :, :]
            T = len(test_dates)

        # portfolio-level score storage
        energy_scores, variogram_scores, dss_scores = [], [], []

        # per-asset storage (lists of daily metrics; may contain NaNs for missing days)
        per_asset = {sym: {"crps": [], "vs": [], "dss": []} for sym in self.asset_order}

        # pre-slice once for speed
        test = self.test_set[['date', 'sym_root', 'ret_crsp']]

        for t, date in enumerate(tqdm(test_dates, desc="Evaluating Scores")):
            day = test[test['date'] == date]

            # Build per-asset y_true with mask of which assets exist
            y_vec = np.full((len(self.asset_order),), np.nan, dtype=float)
            mask = np.zeros((len(self.asset_order),), dtype=bool)
            for i, sym in enumerate(self.asset_order):
                vals = day.loc[day['sym_root'] == sym, 'ret_crsp'].values
                if vals.size and np.isfinite(vals[0]):
                    y_vec[i] = float(vals[0])
                    mask[i] = True

            y_pred_t = self.samples[t]  # (S, N)

            # --- Portfolio-level over available assets only ---
            if mask.any():
                y_true_sub = y_vec[mask][None, :]          # (1, S_avail)
                y_pred_sub = y_pred_t[mask][None, :, :]    # (1, S_avail, N)

                es = es_sample(y_true_sub, y_pred_sub)
                vs = vs_sample(y_true_sub, y_pred_sub, p=p)
                dss = dss_sample(y_true_sub, y_pred_sub)

                energy_scores.append(es)
                variogram_scores.append(vs)
                dss_scores.append(dss)

            # --- Per-asset metrics (no per-day skip) ---
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

        # --- overall means ---
        mean_es = float(np.nanmean(energy_scores)) if energy_scores else np.nan
        mean_vs = float(np.nanmean(variogram_scores)) if variogram_scores else np.nan
        mean_dss = float(np.nanmean(dss_scores)) if dss_scores else np.nan

        logger.info(
            "\n=== Forecast Evaluation Summary ===\n"
            f"Mean Energy Score (ES):            {mean_es:.6f}\n"
            f"Mean Variogram Score (VS, p={p}):  {mean_vs:.6f}\n"
            f"Mean Dawid–Sebastiani Score (DSS): {mean_dss:.6f}\n"
            "==================================="
        )

        # --- per-asset means ---
        asset_means = {
            f"{sym}_crps": float(np.nanmean(scores["crps"])) if scores["crps"] else np.nan
            for sym, scores in per_asset.items()
        }

        # merge everything into one row
        summary_row = {
            "mean_es": mean_es,
            "mean_vs": mean_vs,
            "mean_dss": mean_dss,
            **asset_means
        }

        return summary_row
