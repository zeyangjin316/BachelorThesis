import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from copula_method import CopulaEstimator
from evaluator import ForecastEvaluator
from data.data_handling import DataHandler
from copula_method import TSDataConfig, TSInitConfig, TSFitConfig, TSSampleConfig

logger = logging.getLogger(__name__)

class TwoStepExperiment:
    """
    Experiment container for the two-step copula method:
    handles data preparation, model fitting, sampling, and evaluation.
    """

    def __init__(self,
                 data_config: TSDataConfig,
                 init_config: TSInitConfig,
                 fit_config: TSFitConfig,
                 sample_config: TSSampleConfig):
        """
        Initialize experiment with configs and prepare data.

        Parameters
        ----------
        data_config : TSDataConfig
            Data split and preprocessing settings.
        init_config : TSInitConfig
            Initialization settings for copula and marginals.
        fit_config : TSFitConfig
            Univariate fitting configuration (ARMA–GARCH, etc.).
        sample_config : TSSampleConfig
            Sampling configuration (number of samples, per-UV samples).
        """
        logger.info("Initializing two-step model")
        self.data_config = data_config
        self.init_config = init_config
        self.fit_config = fit_config
        self.sample_config = sample_config

        # Per-day copulas and marginals
        self.copulas_by_day: dict[pd.Timestamp, object] = {}
        self.day_marginals: dict[pd.Timestamp, dict[str, np.ndarray]] = {}

        # Data splits
        self.data_dict: dict[str, pd.DataFrame] = {}
        self._split_data()

        logger.info("Two-step model initialized")

    def _split_data(self):
        """
        Split full dataset into train/test using DataHandler.
        """
        data_handler = DataHandler(self.data_config.split_point)
        self.data_dict = data_handler.get_data(
            target_only=True,
            filter_duplicates=self.data_config.filter_features,
            exclude_pandemic=self.data_config.exclude_pandemic
        )

    def fit(self):
        """
        Fit daily copulas and build marginal distributions.
        Stores copulas in self.copulas_by_day and marginals in self.day_marginals.
        """
        calibrator = CopulaEstimator(self.data_dict,
                                     self.init_config,
                                     self.fit_config,
                                     self.sample_config)
        # Leave one CPU core free for parallelization
        n_jobs = max(1, os.cpu_count() - 1)
        self.copulas_by_day = calibrator.build_daily_copulas(n_jobs=n_jobs)
        self.day_marginals = calibrator.build_day_marginals(
            self.sample_config.n_samples,
            n_jobs=-n_jobs
        )

    def sample(self) -> np.ndarray:
        """
        Generate daily joint samples from copula and marginal forecasts.

        Returns
        -------
        np.ndarray
            Array of shape (n_days, n_symbols, n_samples).
        """
        test_data = self.data_dict['test_set']
        n_samples = int(self.sample_config.n_samples)
        logger.info(f"Sampling {n_samples} multivariate scenarios per day")

        test_dates = sorted(test_data['date'].unique())
        symbols = sorted(test_data['sym_root'].unique())
        n_days, n_symbols = len(test_dates), len(symbols)

        # Ensure prerequisites are ready
        if not self.copulas_by_day:
            logger.info("No per-day copulas found; running fit() first.")
            self.fit()
        elif not self.day_marginals:
            logger.info("No day-t marginals found; building them now.")
            calibrator = CopulaEstimator(self.data_dict,
                                         self.init_config,
                                         self.fit_config,
                                         self.sample_config)
            n_jobs = max(1, os.cpu_count() - 1)
            self.day_marginals = calibrator.build_day_marginals(n_samples, n_jobs=n_jobs)

        # Allocate result array
        all_day_samples = np.full((n_days, n_symbols, n_samples), np.nan, dtype=float)

        # Sampling loop per day
        for day_idx, t in enumerate(tqdm(test_dates,
                                         desc="Sampling Copula Forecasts",
                                         leave=False)):
            t_ts = pd.Timestamp(t)

            # 1) Retrieve copula
            copula = self.copulas_by_day.get(t_ts)
            if copula is None:
                logger.warning(f"No fitted copula for {t_ts}; skipping")
                continue

            # 2) Sample uniforms
            try:
                U = copula.sample_uniforms(n_samples)  # shape (m, n)
            except Exception as e:
                logger.warning(f"Failed copula sampling for {t_ts}: {e}")
                continue

            # 3) Invert marginals for each symbol
            per_sym = self.day_marginals.get(t_ts, {})
            for s_idx, sym in enumerate(symbols):
                draws_t = np.asarray(per_sym.get(sym, np.array([])), dtype=float)
                if draws_t.size < 2:
                    logger.warning(f"No day-{t_ts.date()} marginal samples for {sym}; leaving NaNs")
                    continue

                sorted_samples = np.sort(draws_t)
                ngrid = sorted_samples.size
                q = (np.arange(ngrid) + 0.5) / ngrid  # mid-quantiles
                all_day_samples[day_idx, s_idx, :] = np.interp(U[s_idx], q, sorted_samples)

        logger.info("Finished multiday copula sampling.")
        return all_day_samples

    def evaluate(self, samples):
        """
        Evaluate generated samples against the test set.

        Parameters
        ----------
        samples : np.ndarray
            Forecast sample array.

        Returns
        -------
        dict
            Dictionary with evaluation metrics.
        """
        logger.info(f"Evaluating {self.init_config.copula_type} copula method")
        evaluator = ForecastEvaluator(self.data_dict.get('test_set'), samples)
        return evaluator.evaluate()
