import logging
import numpy as np
import pandas as pd
import joblib
from tqdm.auto import tqdm
from tqdm import tqdm
from copula_method.uv_models import UnivariateModel
from joblib import Parallel, delayed
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

class UnivariateForecaster:
    def __init__(self, data, method, train_set):
        self.train_set_len = len(train_set)
        self.full_data = data
        self.method = method

    def generate_uv_samples(self, test_dates, symbols, n_samples, fixed_window=True, freq=1):
        """
        Generate univariate forecast samples for each symbol and test day.

        For each date:
        - Refit the univariate model every `freq` days (reuse in between).
        - Generate `n_samples` samples per symbol using the fitted model.

        Parameters
        ----------
        test_dates : list of str or pd.Timestamp
            Dates to forecast.
        symbols : list of str
            Asset symbols to model.
        n_samples : int
            Number of samples to generate per symbol per day.
        fixed_window : bool
            Use a rolling window (True) or expanding window (False) for training data.
        freq : int
            Frequency (in days) to refit the univariate model.

        Returns
        -------
        dict[pd.Timestamp, dict[str, np.ndarray]]
            Forecast samples: uv_samples[day][symbol] → np.array of shape (n_samples,)

        Notes
        -----
        Intended for copula input transformation using PIT and Gaussianization.
        """
        uv_samples = {}
        last_model = None

        logger.info("Generating UV samples with model refit every {} days".format(freq))

        for i, date in enumerate(tqdm(test_dates, desc="Generating UV samples")):
            logger.info(f"Processing test day {date} ...")

            # Decide whether to refit
            if i % freq == 0:
                logger.info(f"Fitting univariate model on day {date}")
                data_up_to_date = self.full_data[self.full_data['date'] < date]
                if fixed_window:
                    window_size = self.train_set_len
                    data_up_to_date = data_up_to_date.groupby('sym_root').tail(window_size)

                last_model = UnivariateModel(data_up_to_date, self.method)
                last_model.fit(current_day=date)
            else:
                logger.info(f"Reusing last fitted univariate model for {date}")

            # Parallel sampling per symbol
            def sample_one_symbol(symbol):
                try:
                    samples = last_model.sample(symbol, n_samples=n_samples)
                    return symbol, samples
                except Exception as e:
                    logger.warning(f"Failed sampling {symbol} on {date}: {e}")
                    return symbol, np.array([])

            symbol_samples = Parallel(n_jobs=-1)(
                delayed(sample_one_symbol)(symbol) for symbol in symbols
            )

            uv_samples[date] = {symbol: samples for symbol, samples in symbol_samples}

        logger.info("Finished parallel UV sample generation.")
        logger.info("Generated uv_samples ready for PIT computation.")
        return uv_samples

    def _generate_samples_for_day(self, date, symbols, n_samples, fixed_window):
        """
        Generate n_samples per symbol for a given test date.

        Returns:
        dict { symbol: np.array([samples]) }
        """
        data_up_to_date = self.full_data[self.full_data['date'] < date]

        if fixed_window:
            window_size = self.train_set_len
            data_up_to_date = data_up_to_date.groupby('sym_root').tail(window_size)

        # Fit model on current window
        model = UnivariateModel(data_up_to_date, self.method)
        model.fit(current_day=date)

        samples_for_day = {}

        for symbol in symbols:
            try:
                samples = model.sample(symbol, n_samples=n_samples)
                samples_for_day[symbol] = samples
            except Exception as e:
                logger.warning(f"Failed sampling {symbol} on {date}: {e}")
                samples_for_day[symbol] = np.array([])

        return samples_for_day