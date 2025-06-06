import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ForecastEvaluator:
    def __init__(self, test_set):
        self.test_set = test_set

    def evaluate_energy_score(self, samples):
        """
        Evaluate the generated samples with Energy Score and Copula Energy Score.
        """
        from scoring_rules_supp import es_sample

        test_dates = sorted(self.test_set['date'].unique())
        symbols = samples.columns.tolist()

        # Reshape the fixed sample matrix: (1, n_dim, n_samples)
        y_pred = samples.T.values[np.newaxis, :, :]  # shape (1, n_dim, n_samples)

        energy_scores = []

        for date in tqdm(test_dates, desc="Evaluating test days"):
            test_day_data = self.test_set[self.test_set['date'] == date]

            try:
                y_true = np.array([
                    test_day_data[test_day_data['sym_root'] == symbol]['ret_crsp'].values[0]
                    for symbol in symbols
                ]).reshape(1, -1)
            except IndexError:
                logger.warning(f"Skipping {date} due to missing values.")
                continue

            es = es_sample(y_true, y_pred)

            energy_scores.append(es)

        mean_es = np.mean(energy_scores)

        logger.info(f"\nMean Energy Score: {mean_es:.6f}")

        return mean_es