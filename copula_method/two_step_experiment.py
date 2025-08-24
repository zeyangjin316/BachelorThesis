import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from copula_method import CopulaCalibrator
from evaluator import ForecastEvaluator
from data.data_handling import DataHandler
from copula_method import TSDataConfig, TSInitConfig, TSFitConfig, TSSampleConfig

logger = logging.getLogger(__name__)

class TwoStepExperiment:

    def __init__(self,
                 data_config: TSDataConfig = TSDataConfig(),
                 init_config: TSInitConfig = TSInitConfig(),
                 fit_config: TSFitConfig = TSFitConfig(),
                 sample_config: TSSampleConfig = TSSampleConfig()):

        logger.info("Initializing two-step model")
        self.data_config = data_config
        self.init_config = init_config
        self.fit_config = fit_config
        self.sample_config = sample_config

        self.corr_by_day: dict[pd.Timestamp, np.ndarray] = {}
        self.day_marginals: dict[pd.Timestamp, dict[str, np.ndarray]] = {}

        # Collecting and splitting data
        self.data_dict: dict[str, pd.DataFrame] = {}
        self._split_data()

        logger.info("Two-step model initialized")

    def _split_data(self):
        data_handler = DataHandler(self.data_config.split_point)
        self.data_dict = data_handler.get_data(standardize=False, filter_features=True, exclude_pandemic=True)

    def fit(self):
        calibrator = CopulaCalibrator(self.data_dict, self.init_config, self.fit_config, self.sample_config)
        # Parallel across test days
        import os
        n_jobs = max(1, os.cpu_count() - 1) # leave one core free, set n_jobs=-1 for all cores
        self.corr_by_day = calibrator.build_daily_correlations(n_jobs=n_jobs)
        self.day_marginals = calibrator.build_day_marginals(self.sample_config.n_samples, n_jobs=-n_jobs)

    def sample(self) -> np.ndarray:
        """
        Generate daily joint return samples from the copula and marginal forecasts.
        Returns: (n_days, n_symbols, n_samples)
        """
        test_data = self.data_dict['test_set']
        n_samples = int(self.sample_config.n_samples)
        logger.info(f"Sampling {n_samples} multivariate scenarios per day")

        test_dates = sorted(test_data['date'].unique())
        symbols = sorted(test_data['sym_root'].unique())
        n_days, n_symbols = len(test_dates), len(symbols)

        if not self.corr_by_day:
            logger.info("No copula correlations found; running fit() first.")
            self.fit()
        elif not self.day_marginals:
            logger.info("No day-t marginals found; building them now.")
            calibrator = CopulaCalibrator(self.data_dict, self.init_config, self.fit_config, self.sample_config)
            self.day_marginals = calibrator.build_day_marginals(n_samples)

        all_day_samples = np.full((n_days, n_symbols, n_samples), np.nan, dtype=float)

        for day_idx, t in enumerate(tqdm(test_dates, desc="Sampling Copula Forecasts", leave=False)):
            t_ts = pd.Timestamp(t)

            # 1) correlation for day t (dict lookup)
            R_t = self.corr_by_day.get(t_ts)
            if R_t is None:
                logger.warning(f"No copula correlation for {t_ts}; skipping")
                continue

            try:
                # 2) sample Z ~ N(0, R_t) and map to uniforms
                z = np.random.multivariate_normal(mean=np.zeros(n_symbols), cov=R_t, size=n_samples).T
                u = norm.cdf(z)
            except Exception as e:
                logger.warning(f"Failed copula sampling for {t_ts}: {e}")
                continue

            # 3) invert per-symbol marginals for this day t
            per_sym = self.day_marginals.get(t_ts, {})
            for s_idx, sym in enumerate(symbols):
                draws_t = np.asarray(per_sym.get(sym, np.array([])), dtype=float)
                if draws_t.size < 2:
                    logger.warning(f"No day-{t_ts.date()} marginal samples for {sym}; leaving NaNs")
                    continue

                sorted_samples = np.sort(draws_t)
                ngrid = sorted_samples.size
                q = (np.arange(ngrid) + 0.5) / ngrid  # mid-quantiles
                all_day_samples[day_idx, s_idx, :] = np.interp(u[s_idx], q, sorted_samples)

        logger.info("Finished multiday copula sampling.")
        return all_day_samples

    def evaluate(self, samples):
        """
        Evaluate the generated samples with Energy Score and Copula Energy Score.
        """
        logger.info(f"Evaluating {self.init_config.copula_type} copula method")
        evaluator = ForecastEvaluator(self.data_dict.get('test_set'), samples)
        return evaluator.evaluate()

