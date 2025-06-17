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
    def __init__(self, split_point: Union[float, datetime] = 0.8, univariate_type: str = "ARMAGARCH", copula_type: str ="Gaussian",):

        logger.info("Initializing two-step model")
        self.split_point = split_point
        self.univariate_type = univariate_type
        self.copula_type = copula_type

        # Collecting and splitting data
        self.data_dict = self._get_data()
        self.full_data = self.data_dict['full_data']
        self.train_data = self.data_dict['train_set']
        self.test_data = self.data_dict['test_set']

        logger.info("Two-step model initialized")

    def _get_data(self):
        data_handler = DataHandler(self.split_point)
        full_data = data_handler.get_data(split=False)
        train_set, test_set = data_handler.get_data(split=True)
        """with pd.option_context('display.max_columns', None):
            print(full_data.head())"""
        return {'full_data': full_data, 'train_set': train_set, 'test_set': test_set, 'split_point': self.split_point,}


    def fit(self, n_samples_daily = 100):
        logger.info("Starting fitting two-step model")
        test_dates = sorted(self.test_data['date'].unique()) # All dates from the test set
        symbols = self.full_data['sym_root'].unique()       # All symbols in the full data set

        #Step 1: Fit univariate models
        univariate_forecaster = UnivariateForecaster(self.full_data, self.univariate_type, self.train_data)
        self.uv_samples = univariate_forecaster.generate_uv_samples(test_dates, symbols,n_samples=n_samples_daily, fixed_window=True)

        # Step 2: Create Gaussianized copula inputs
        gaussian_copula_input = CopulaTransformer.to_gaussian_input(self.test_data, self.uv_samples, test_dates)

        # Step 3: Fit copula using the transformed data
        copula_estimator = CopulaEstimator(self.copula_type)
        copula_estimator.fit(gaussian_copula_input)
        self.fitted_copula = copula_estimator.fitted_copula

        logger.info("Finished fitting two-step model")

    def sample(self, n_trajectories: int = 1000):
        """
        Generate joint return samples from the fitted copula and marginal forecast distributions.

        Returns:
            pd.DataFrame: shape = (n_trajectories, n_symbols)
        """
        logger.info(f"Sampling {n_trajectories} multivariate scenarios from copula")

        # Safety check
        if not self.fitted_copula:
            raise ValueError("Copula must be fitted before calling sample().")

        copula_model = self.fitted_copula
        if not hasattr(copula_model, "sample"):
            raise AttributeError("Fitted copula object has no sample method.")

        # Get symbols from copula input columns
        symbols = copula_model.columns if hasattr(copula_model, "columns") else self.test_data[
            'sym_root'].unique().tolist()

        # Step 1: Sample from copula in Gaussian space
        z_samples = copula_model.sample(n_trajectories)
        z_samples.columns = symbols

        # Step 2: Gaussian → Uniform space
        u_samples = pd.DataFrame(norm.cdf(z_samples), columns=symbols)

        # Step 3: Invert marginal distributions using concatenated univariate samples
        final_samples = pd.DataFrame(index=range(n_trajectories), columns=symbols)

        for symbol in symbols:
            logger.info(f"Inverting marginal forecast for {symbol}")

            try:
                # Concatenate all available samples for this symbol
                all_samples = np.concatenate([
                    self.uv_samples[symbol][day] for day in self.uv_samples[symbol]
                ])
                forecast_samples = np.sort(all_samples)
                percentiles = np.linspace(0, 1, len(forecast_samples))

                # Interpolate each u to its quantile value
                final_samples[symbol] = np.interp(u_samples[symbol], percentiles, forecast_samples)

            except Exception as e:
                logger.warning(f"Failed to invert marginal for {symbol}: {e}")
                final_samples[symbol] = np.nan

        logger.info("Sample generation completed successfully")
        return final_samples

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

