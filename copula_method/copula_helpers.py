import numpy as np
import logging
import pandas as pd
from scipy.special import erfinv
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CopulaTransformer:
    @staticmethod
    def to_gaussian_input(test_set, uv_samples, days):
        logger.info("Calculating Gaussian copula input matrix")
        matrix = []

        for day in tqdm(days, desc="Computing copula inputs"):
            test_data_day = test_set[test_set['date'] == day]
            z_row = {}

            for symbol in test_data_day['sym_root'].unique():
                try:
                    samples = uv_samples[symbol]
                    X_dh = test_data_day[test_data_day['sym_root'] == symbol]['ret_crsp'].values[0]

                    u_dh = np.mean(samples <= X_dh)
                    u_dh = np.clip(u_dh, 1e-6, 1 - 1e-6)

                    # z = norm.ppf(u)
                    Z_dh = np.sqrt(2) * erfinv(2 * u_dh - 1)
                    z_row[symbol] = Z_dh

                except Exception as e:
                    logger.warning(f"Failed to compute Z_d,h for {symbol} on {day}: {e}")
                    continue

            if z_row:
                matrix.append(pd.Series(z_row, name=day))

        df = pd.DataFrame(matrix)
        logger.info(f"Created Gaussian copula input matrix with shape: {df.shape}")
        return df.dropna(axis=1)