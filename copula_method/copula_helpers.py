import numpy as np
import logging
import pandas as pd
from scipy.special import erfinv
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CopulaTransformer:
    @staticmethod
    def to_gaussian_input(test_set, uv_samples, days):
        """
            Compute Gaussian copula input matrix from uv_samples and test set.

            For each test day and symbol:
            - The true observed value X_{d,h} is taken from the test set.
            - The empirical CDF F^{d,h}(X_{d,h}) is estimated using uv_samples[symbol][day].
            - The PIT value u_{d,h} is transformed to Gaussian space via Φ⁻¹(u_{d,h}).
            - The result is one Gaussianized value Z_{d,h} per symbol per day.

            Parameters
            ----------
            test_set : pd.DataFrame
                Test set containing the true observed values.
            uv_samples : dict
                Nested dictionary of samples, typically produced by UnivariateForecaster.generate_uv_samples().

                Structure:

                uv_samples[symbol][day] → np.array of shape (n_samples,)

            days : list of str or pd.Timestamp
                List of test dates to process.

            Returns
            -------
            pd.DataFrame
                Gaussian copula input matrix.

                The DataFrame has:
                    - Rows = test dates.
                    - Columns = symbols.
                    - Values = Z_{d,h} = Φ⁻¹(u_{d,h}), where u_{d,h} is computed from uv_samples.

            Notes
            -----
            - The resulting DataFrame is used as the input for fitting a Gaussian copula.
            - Only ONE Z per symbol per day is computed from the samples.

            """
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