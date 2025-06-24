import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Union

from scipy.stats import norm
from tqdm import tqdm

from copula_method.uv_forecaster import UnivariateForecaster
from copula_method.copula_fitting import CopulaFitter
from evaluator import ForecastEvaluator
from data_handling import DataHandler

logger = logging.getLogger(__name__)

class TwoStepModel:
    def __init__(self,
                 split_point: Union[float, datetime] = 0.8,
                 fixed_uv_window: bool = True,
                 uv_train_freq: int = 1,
                 copula_window_size: float = 0.05,
                 univariate_type: str = "ARMAGARCH",
                 copula_type: str ="Gaussian"):

        logger.info("Initializing two-step model")
        self.split_point = split_point
        self.fixed_uv_window = fixed_uv_window
        self.uv_train_freq = uv_train_freq
        self.copula_window_size = copula_window_size
        self.univariate_type = univariate_type
        self.copula_type = copula_type
        self.data_handler = DataHandler(self.split_point)

        # Collecting and splitting data
        self.data_dict = self.data_handler.get_data()
        self.full_data = self.data_dict['full_data']
        self.train_data = self.data_dict['train_set']
        self.test_data = self.data_dict['test_set'] # the first day included in PIT computation

        logger.info("Two-step model initialized")

    def fit(self, n_samples_per_day=100):
        logger.info("Starting fitting two-step model")

        # === Define symbol list ===
        symbols = sorted(self.full_data['sym_root'].unique())

        # === Define copula calibration window ===
        train_dates = sorted(self.train_data['date'].unique())
        copula_window_days = int(len(train_dates) * self.copula_window_size)
        self.copula_start_date = train_dates[-copula_window_days]  # rolling window starts here

        logger.info(f"Copula calibration window starts at {self.copula_start_date} ({copula_window_days} days)")

        # === Define univariate forecast dates ===
        # Must cover the copula window + test set
        uv_forecast_dates = sorted(self.full_data[self.full_data['date'] >= self.copula_start_date]['date'].unique())
        logger.info(
            f"Generating univariate samples for {len(uv_forecast_dates)} days starting from {uv_forecast_dates[0]}")

        # === Generate univariate forecast samples ===
        univariate_forecaster = UnivariateForecaster(
            data=self.full_data,
            method=self.univariate_type,
            train_set=self.train_data
        )
        self.uv_samples = univariate_forecaster.generate_uv_samples(
            test_dates=uv_forecast_dates,
            symbols=symbols,
            n_samples=n_samples_per_day,
            fixed_window=self.fixed_uv_window,
            freq=self.uv_train_freq
        )

        # === Fit rolling copulas using PITs and Z-vectors ===
        test_dates = sorted(self.test_data['date'].unique())
        self.copula_fitter = CopulaFitter(
            copula_type=self.copula_type,
            rolling_window_size=copula_window_days
        )
        copula_data = self.full_data[self.full_data['date'] >= self.copula_start_date]

        self.copula_fitter.calc_all_matrices(
            full_data=copula_data,
            uv_samples=self.uv_samples,
            symbols=symbols,
            test_dates=test_dates
        )

        logger.info("Two-step model fitting complete")

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate daily joint return samples from the copula and marginal forecasts.

        Returns
        -------
        np.ndarray
            Shape (n_days, n_symbols, n_samples)
        """
        logger.info(f"Sampling {n_samples} multivariate scenarios per day")

        test_dates = sorted(self.test_data['date'].unique())
        symbols = sorted(self.test_data['sym_root'].unique())
        n_days = len(test_dates)
        n_symbols = len(symbols)

        all_day_samples = np.full((n_days, n_symbols, n_samples), np.nan)

        with tqdm(test_dates, desc="Sampling Copula Forecasts", leave=False) as pbar:
            for day_idx, current_day in enumerate(pbar):
                pbar.set_description(f"Sampling {current_day.date()}")

                # === Step 1: Get copula correlation matrix ===
                corr_matrix = self.copula_fitter.get_corr_matrix(current_day)
                if corr_matrix is None:
                    logger.warning(f"No copula fitted for {current_day}; skipping")
                    continue

                try:
                    # === Step 2: Sample from Gaussian copula ===
                    mean = np.zeros(n_symbols)
                    z_samples = np.random.multivariate_normal(mean, corr_matrix, size=n_samples).T

                    # === Step 3: Gaussian → Uniform space ===
                    u_samples = norm.cdf(z_samples)
                except Exception as e:
                    logger.warning(f"Failed copula sampling for {current_day}: {e}")
                    continue

                # === Step 4: Invert marginals for each symbol ===
                for s_idx, symbol in enumerate(symbols):
                    try:
                        all_symbol_samples = np.concatenate([
                            self.uv_samples[day][symbol] for day in self.uv_samples if symbol in self.uv_samples[day]
                        ])
                        sorted_samples = np.sort(all_symbol_samples)
                        percentiles = np.linspace(0, 1, len(sorted_samples))
                        all_day_samples[day_idx, s_idx, :] = np.interp(
                            u_samples[s_idx], percentiles, sorted_samples
                        )
                    except Exception as e:
                        logger.warning(f"Failed marginal inversion for {symbol} on {current_day}: {e}")

        logger.info("Finished multiday copula sampling.")
        return all_day_samples

    def evaluate(self, samples):
        """
        Evaluate the generated samples with Energy Score and Copula Energy Score.
        """
        logger.info(f"Evaluating {self.copula_type} copula method")
        evaluator = ForecastEvaluator(self.test_data, samples)
        return evaluator.evaluate()

    def show_data(self):
        for symbol in self.train_data['sym_root'].unique():
            # Get the corresponding train and test data for the symbol
            train_data = self.train_data[self.train_data['sym_root'] == symbol]
            test_data = self.test_data[self.test_data['sym_root'] == symbol]

            # Create a plot for the symbol's training and test data
            plt.figure(figsize=(10, 6))

            # Plot the training data
            plt.plot(train_data['date'], train_data['ret_crsp'], label='Train', color='blue')

            # Plot the test data
            plt.plot(test_data['date'], test_data['ret_crsp'], label='Test', color='red')

            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title(f"'ret_crsp' Split for {symbol}")
            plt.legend()

            # Show the plot
            plt.show()

