import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ForecastEvaluator:
    def __init__(self, test_set, samples, asset_order=None):
        self.test_set = test_set
        self.samples = samples
        self.asset_order = asset_order or sorted(test_set['sym_root'].unique())

    def evaluate(self, p=0.5):
        from scoring_rules_supp import es_sample, vs_sample, dss_sample

        test_dates = sorted(self.test_set['date'].unique())
        n_days = self.samples.shape[0]

        if len(test_dates) > n_days:
            offset = len(test_dates) - n_days
            test_dates = test_dates[offset:]
            logger.info(
                f"Aligned test_dates to match sample size: using last {n_days} of {len(test_dates) + offset}"
            )
        else:
            offset = 0
            assert len(test_dates) == n_days, "Mismatch between test dates and sample size"

        # Mean scores across all assets
        energy_scores = []
        variogram_scores = []
        dss_scores = []

        # Per-asset storage
        asset_scores = {sym: {"es": [], "vs": [], "dss": []} for sym in self.asset_order}

        for t, date in enumerate(tqdm(test_dates, desc="Evaluating Scores")):
            test_day_data = self.test_set[self.test_set['date'] == date]

            try:
                y_true = np.array([
                    test_day_data[test_day_data['sym_root'] == symbol]['ret_crsp'].values[0]
                    for symbol in self.asset_order
                ]).reshape(1, -1)

                if np.isnan(y_true).any():
                    logger.warning(f"Skipping {date} due to NaN in ground truth returns.")
                    continue

            except IndexError:
                logger.warning(f"Skipping {date} due to missing asset data.")
                continue

            y_pred = self.samples[t][np.newaxis, :, :]  # shape: (1, n_assets, n_samples)

            # --- Mean scores (portfolio level) ---
            es = es_sample(y_true, y_pred)
            vs = vs_sample(y_true, y_pred, p=p)
            dss = dss_sample(y_true, y_pred)

            energy_scores.append(es)
            variogram_scores.append(vs)
            dss_scores.append(dss)

            # --- Per-asset scores ---
            for i, sym in enumerate(self.asset_order):
                y_true_i = y_true[0, i].reshape(1, -1)        # (1,1)
                y_pred_i = y_pred[0, i, :].reshape(1, 1, -1)  # (1,1,n_samples)

                es_i = es_sample(y_true_i, y_pred_i)
                vs_i = vs_sample(y_true_i, y_pred_i, p=p)
                dss_i = dss_sample(y_true_i, y_pred_i)

                asset_scores[sym]["es"].append(es_i)
                asset_scores[sym]["vs"].append(vs_i)
                asset_scores[sym]["dss"].append(dss_i)

        # === Overall mean scores ===
        mean_es = np.mean(energy_scores) if energy_scores else np.nan
        mean_vs = np.mean(variogram_scores) if variogram_scores else np.nan
        mean_dss = np.mean(dss_scores) if dss_scores else np.nan

        logger.info(
            "\n=== Forecast Evaluation Summary ===\n"
            f"Mean Energy Score (ES):           {mean_es:.6f}\n"
            f"Mean Variogram Score (VS, p={p}): {mean_vs:.6f}\n"
            f"Mean Dawid–Sebastiani Score (DSS): {mean_dss:.6f}\n"
            "==================================="
        )

        # === Per-asset DataFrame ===
        df_assets = pd.DataFrame.from_dict(
            {
                sym: {
                    "mean_es": np.mean(scores["es"]) if scores["es"] else np.nan,
                    "mean_vs": np.mean(scores["vs"]) if scores["vs"] else np.nan,
                    "mean_dss": np.mean(scores["dss"]) if scores["dss"] else np.nan,
                }
                for sym, scores in asset_scores.items()
            },
            orient="index"
        ).reset_index().rename(columns={"index": "asset"})

        return {
            "mean_es": mean_es,
            "mean_vs": mean_vs,
            "mean_dss": mean_dss,
            "asset_scores": df_assets,
        }

