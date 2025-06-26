import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ForecastEvaluator:
    def __init__(self, test_set, samples, asset_order=None):
        self.test_set = test_set
        self.samples = samples
        self.asset_order = asset_order or sorted(test_set['sym_root'].unique())

    def evaluate(self, p=0.5):
        from scoring_rules_supp import es_sample, vs_sample

        test_dates = sorted(self.test_set['date'].unique())
        n_days = self.samples.shape[0]

        if len(test_dates) > n_days:
            offset = len(test_dates) - n_days
            test_dates = test_dates[offset:]
            logger.info(f"Aligned test_dates to match sample size: using last {n_days} of {len(test_dates) + offset}")
        else:
            offset = 0
            assert len(test_dates) == n_days, "Mismatch between test dates and sample size"

        energy_scores = []
        variogram_scores = []

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

            es = es_sample(y_true, y_pred)
            vs = vs_sample(y_true, y_pred, p=p)

            energy_scores.append(es)
            variogram_scores.append(vs)

        mean_es = np.mean(energy_scores)
        mean_vs = np.mean(variogram_scores)

        logger.info(f"\nMean Energy Score: {mean_es:.6f}")
        logger.info(f"Mean Variogram Score (p={p}): {mean_vs:.6f}")

        return {
            "es_score": mean_es,
            "vs_score": mean_vs
        }