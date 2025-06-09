import logging
import numpy as np
import pandas as pd
import multiprocessing
import joblib
from tqdm.auto import tqdm
from tqdm import tqdm
from copula_method.univariate_models import UnivariateModel
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

    def generate_uv_samples(self, test_dates, symbols, n_samples, fixed_window=True):
        """
        Generate multiple samples per symbol per test day, for use in copula input transformation.

        Parameters
        ----------
        test_dates : list of str or pd.Timestamp
            List of test dates for which to generate samples.
        symbols : list of str
            List of symbols to forecast.
        n_samples : int, optional (default=1000)
            Number of samples to generate per symbol per day.
        fixed_window : bool, optional (default=True)
            If True, use a fixed rolling window of size equal to the initial train set length.
            If False, use an expanding window.

        Returns
        -------
        uv_samples : dict
            Nested dictionary with the following structure:

            uv_samples[symbol][day] → np.array of shape (n_samples,)

            Example structure:

            {
                'MSFT': {
                    '2020-01-01': np.array([sample1, sample2, ..., sampleN]),
                    '2020-01-02': np.array([...]),
                    ...
                },
                'AAPL': {
                    '2020-01-01': np.array([...]),
                    '2020-01-02': np.array([...]),
                    ...
                },
                ...
            }

        Notes
        -----
        - This output is intended for use with CopulaTransformer.to_gaussian_input().
        - The PIT (Probability Integral Transform) and Gaussianization are performed using the samples from uv_samples.
        - Only ONE Z per symbol per day is computed and fed into the Gaussian copula.

        """
        uv_samples = {symbol: {} for symbol in symbols}

        num_cores = multiprocessing.cpu_count()
        logger.info(f"Using {num_cores} cores for parallel UV sample generation.")

        logger.info("Starting parallel UV sample generation...")

        def process_one_day(date):
            logger.info(f"Processing test day {date} ...")
            samples = self._generate_samples_for_day(date, symbols, n_samples, fixed_window)
            return date, samples

        # Run parallel loop with tqdm progress bar
        with tqdm_joblib(tqdm(desc="Generating UV samples", total=len(test_dates))) as progress_bar:
            parallel_results = Parallel(n_jobs=-1)(
                delayed(process_one_day)(date) for date in test_dates
            )

        # Collect results
        for date, samples in parallel_results:
            for symbol in symbols:
                uv_samples[symbol][date] = samples.get(symbol, np.array([]))

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