import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from datetime import datetime
from typing import Union

from copula_method.uv_forecaster import UnivariateForecaster
from copula_method.copula_fitting import CopulaEstimator
from copula_method.copula_helpers import CopulaTransformer
from evaluator import ForecastEvaluator
from data_handling import DataHandler

logger = logging.getLogger(__name__)

class TwoStepModel:
    def __init__(self,
                 split_point: Union[float, datetime] = 0.8,
                 fixed_window: bool = True,
                 uv_train_freq: int = 1,
                 copula_train_freq: int = 1,
                 univariate_type: str = "ARMAGARCH",
                 copula_type: str ="Gaussian"):

        logger.info("Initializing two-step model")
        self.split_point = split_point
        self.fixed_window = fixed_window
        self.uv_train_freq = uv_train_freq
        self.copula_train_freq = copula_train_freq
        self.univariate_type = univariate_type
        self.copula_type = copula_type
        self.data_handler = DataHandler(self.split_point)

        # Collecting and splitting data
        self.data_dict = self.data_handler.get_data()
        self.full_data = self.data_dict['full_data']
        self.train_data = self.data_dict['train_set']
        self.test_data = self.data_dict['test_set']

        logger.info("Two-step model initialized")


    def fit(self, n_samples_per_day = 100):
        logger.info("Starting fitting two-step model")
        test_dates = sorted(self.test_data['date'].unique())  # All dates from the test set
        symbols = self.full_data['sym_root'].unique()  # All symbols in the full data set

        # Step 1: Generate univariate samples for all days in the test set
        univariate_forecaster = UnivariateForecaster(
            self.full_data,
            self.univariate_type,
            self.train_data
        )
        self.uv_samples = univariate_forecaster.generate_uv_samples(
            test_dates,
            symbols,
            n_samples=n_samples_per_day,
            fixed_window=True,
            freq=self.uv_train_freq
        )

        # Step 2: Fit copula every copula_train_freq days
        self.copulas_by_day = {}
        for i, current_day in enumerate(test_dates):
            if i % self.copula_train_freq == 0:
                logger.info(f"Fitting copula for day {current_day}")
                copula_input_matrix = CopulaTransformer.to_gaussian_input(
                                                            test_set=self.test_data,
                                                            uv_samples=self.uv_samples,
                                                            days=[current_day]
                                                        )
                copula_estimator = CopulaEstimator(self.copula_type)
                copula_estimator.fit(copula_input_matrix)
                self.fitted_copula = copula_estimator.fitted_copula
            else:
                logger.info(f"Reusing copula from previous day for {current_day}")
            self.copulas_by_day[current_day] = self.fitted_copula

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate daily joint return samples from the copula and marginal forecasts.

        Returns:
            np.ndarray of shape (n_days, n_symbols, n_samples)
        """
        from scipy.stats import norm

        logger.info(f"Sampling {n_samples} multivariate scenarios per day")

        test_dates = sorted(self.test_data['date'].unique())
        symbols = sorted(self.test_data['sym_root'].unique())
        n_days = len(test_dates)
        n_symbols = len(symbols)

        all_day_samples = np.full((n_days, n_symbols, n_samples), np.nan)

        for day_idx, current_day in enumerate(test_dates):
            logger.info(f"Sampling for day {current_day}")

            # Get copula
            copula = self.copulas_by_day.get(current_day)
            if copula is None:
                logger.warning(f"No copula fitted for {current_day}; skipping")
                continue

            # Sample from copula in Gaussian space
            if not hasattr(copula, "sample"):
                logger.warning(f"Copula for {current_day} has no sample method; skipping")
                continue

            try:
                z_samples = copula.sample(n_samples)
                z_samples.columns = symbols
                u_samples = pd.DataFrame(norm.cdf(z_samples), columns=symbols)
            except Exception as e:
                logger.warning(f"Failed copula sampling for {current_day}: {e}")
                continue

            for s_idx, symbol in enumerate(symbols):
                try:
                    all_symbol_samples = np.concatenate([
                        self.uv_samples[symbol][day] for day in self.uv_samples[symbol]
                    ])
                    sorted_samples = np.sort(all_symbol_samples)
                    percentiles = np.linspace(0, 1, len(sorted_samples))
                    all_day_samples[day_idx, s_idx, :] = np.interp(
                        u_samples[symbol], percentiles, sorted_samples
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

