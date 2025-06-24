import numpy as np
import pandas as pd
import logging
from scipy.special import erfinv
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CopulaFitter:
    def __init__(self, copula_type: str = "Gaussian", rolling_window_size: int = 91):
        """
        Parameters
        ----------
        copula_type : str
            Only 'Gaussian' is supported. The copula itself is not fitted directly;
            instead, a correlation matrix is computed to be used as input to a Gaussian copula.
        rolling_window_size : int
            Number of days in the rolling window used to calculate Z-matrix correlation.
        """
        if copula_type != "Gaussian":
            raise ValueError(f"Copula type '{copula_type}' not supported. Only 'Gaussian' is implemented.")
        self.rolling_window_size = rolling_window_size
        self.corr_matrices = {}  # {date: correlation_matrix}

    def _calc_corr_single(self, z_window: pd.DataFrame) -> np.ndarray:
        """
        Compute the empirical correlation matrix for Gaussian copula input.

        Parameters
        ----------
        z_window : pd.DataFrame
            Rows = days, Columns = symbols. Values are Gaussianized PITs.

        Returns
        -------
        np.ndarray
            Correlation matrix to be used as the Gaussian copula parameter.
        """
        return np.corrcoef(z_window.values, rowvar=False)

    def calc_all_matrices(self, full_data: pd.DataFrame, uv_samples: dict, symbols: list[str], test_dates: list[pd.Timestamp]):
        """
        For each day in `test_dates`, compute the Gaussian copula input matrix (correlation of Z-vectors)
        using a rolling window of Gaussianized PITs.

        Parameters
        ----------
        full_data : pd.DataFrame
            Includes columns ['date', 'sym_root', 'ret_crsp'] for real values.
        uv_samples : dict[date][symbol]
            Forecast samples from univariate models.
        symbols : list of str
            List of symbols (copula variables).
        test_dates : list[pd.Timestamp]
            Dates for which the Z-based correlation matrix should be calculated.
        """
        all_dates = sorted(full_data['date'].unique())
        n_symbols = len(symbols)
        z_matrix = np.full((self.rolling_window_size, n_symbols), np.nan)
        z_row_idx = 0

        for current_day in tqdm(all_dates, desc="Building Copula Input Matrices", leave=False):
            day_data = full_data[full_data['date'] == current_day]
            pit_vector = []

            for symbol in symbols:
                try:
                    if current_day not in uv_samples or symbol not in uv_samples[current_day]:
                        raise ValueError("Missing uv_samples entry")

                    samples = np.asarray(uv_samples[current_day][symbol])
                    if samples.size == 0:
                        raise ValueError("Empty samples")

                    vals = day_data[day_data['sym_root'] == symbol]['ret_crsp'].values
                    if len(vals) == 0:
                        raise ValueError("Missing real value")

                    true_val = vals[0]
                    u = np.mean(samples <= true_val)
                    u = np.clip(u, 1e-6, 1 - 1e-6)
                    pit_vector.append(u)
                except Exception as e:
                    logger.warning(f"PIT failed for {symbol} on {current_day}: {e}")
                    pit_vector.append(0.5)

            z_vector = np.sqrt(2) * erfinv(2 * np.array(pit_vector) - 1)
            z_matrix[z_row_idx % self.rolling_window_size] = z_vector
            z_row_idx += 1

            if z_row_idx >= self.rolling_window_size and current_day in test_dates:
                if z_row_idx == self.rolling_window_size:
                    window = z_matrix
                else:
                    head = z_matrix[z_row_idx % self.rolling_window_size:]
                    tail = z_matrix[:z_row_idx % self.rolling_window_size]
                    window = np.vstack([head, tail])

                z_window = pd.DataFrame(window, columns=symbols)
                corr_matrix = self._calc_corr_single(z_window)
                self.corr_matrices[current_day] = corr_matrix
                #logger.info(f"Stored correlation matrix for {current_day}")

    def get_corr_matrix(self, date) -> np.ndarray:
        """
        Retrieve the correlation matrix used as Gaussian copula input for a given date.

        Parameters
        ----------
        date : pd.Timestamp

        Returns
        -------
        np.ndarray or None
        """
        return self.corr_matrices.get(date, None)

