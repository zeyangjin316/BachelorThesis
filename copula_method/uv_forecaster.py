import logging
import numpy as np
import joblib
from uv_models import UnivariateModel
from tqdm.auto import tqdm
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
    def __init__(self, data, method, train_set, uv_train_freq, fixed_window):
        self.full_data = data
        self.method = method
        self.train_set_len = len(train_set)
        self.uv_train_freq = uv_train_freq
        self.fixed_window = fixed_window

    def _fit_model_for_date(self, date):
        """Fit and return a UnivariateModel using data up to (but excluding) `date`."""
        data_up_to_date = self.full_data[self.full_data['date'] < date]
        if self.fixed_window:
            data_up_to_date = data_up_to_date.groupby('sym_root').tail(self.train_set_len)
        model = UnivariateModel(data_up_to_date, self.method)
        model.fit(current_day=date)
        return model

    def _sample_symbols(self, model, symbols, n_samples, date):
        """Sample all symbols in parallel from a fitted model."""
        def sample_one_symbol(symbol):
            try:
                samples = model.sample(symbol, n_samples=n_samples)
                return symbol, samples
            except Exception as e:
                logger.warning(f"Failed sampling {symbol} on {date}: {e}")
                return symbol, np.array([])
        with tqdm_joblib(tqdm(total=len(symbols), desc=f"Sampling {date}")):
            pairs = Parallel(n_jobs=-1)(delayed(sample_one_symbol)(s) for s in symbols)
        return {sym: samp for sym, samp in pairs}

    def _prefit_models(self, test_dates):
        """
        Fit models for all refit dates in parallel and return {refit_date: model}.
        """
        freq = self.uv_train_freq
        refit_dates = [d for i, d in enumerate(test_dates) if i % freq == 0]

        logger.info(f"Parallel fitting {len(refit_dates)} refit points (freq={freq}).")

        # Fit in parallel across refit dates
        with tqdm_joblib(tqdm(total=len(refit_dates), desc="Fitting models (refit dates)")):
            models = Parallel(n_jobs=-1)(
                delayed(self._fit_model_for_date)(d) for d in refit_dates
            )

        return dict(zip(refit_dates, models))

    def generate_uv_samples(self, test_dates, symbols, n_samples):
        """
        Generate univariate forecast samples for each symbol and test day,
        refitting every `self.uv_train_freq` days (parallelized) and reusing in between.
        """
        uv_samples = {}
        freq = self.uv_train_freq

        # 1) Parallel-fit models for refit dates
        models_by_refit_date = self._prefit_models(test_dates)

        # 2) For each test day, pick the latest refit model and sample symbols (parallel)
        last_refit_date = None
        for i, date in enumerate(tqdm(test_dates, desc="Generating UV samples")):
            if (i % freq) == 0:
                last_refit_date = date
                logger.info(f"Using freshly fitted model for refit day {date}")
            else:
                logger.info(f"Reusing model from last refit day {last_refit_date} for {date}")

            model = models_by_refit_date[last_refit_date]
            uv_samples[date] = self._sample_symbols(model, symbols, n_samples, date)

        logger.info("Finished UV sample generation (fit+sample parallelized).")
        return uv_samples