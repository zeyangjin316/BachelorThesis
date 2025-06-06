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
from data_handling import Reader, DataHandler

logger = logging.getLogger(__name__)

class TwoStepModel:
    def __init__(self, split_point: Union[float, datetime] = 0.8,
                 univariate_type: str = "ARMAGARCH", copula_type: str ="Gaussian",
                 n_samples: int = 1000):
        logger.info("Initializing two-step model")
        self.univariate_type = univariate_type
        self.copula_type = copula_type
        self.n_samples = n_samples

        # Collecting and splitting data
        self.split_point = split_point
        self.reader = Reader()
        self.data_handler = DataHandler(split_point)
        self.train_set, self.test_set = self.data_handler.get_train_test_data()

        logger.info("Two-step model initialized")


    def fit(self):
        logger.info("Starting fitting two-step model")

        #Step 1: Fit univariate models
        univariate_forecaster = UnivariateForecaster(self.train_set, self.univariate_type)
        self.uv_samples = univariate_forecaster.generate_samples(self.train_set['sym_root'].unique(), self.n_samples)

        # Step 2: Create Gaussianized copula inputs
        days = sorted(self.test_set['date'].unique())
        gaussian_copula_input = CopulaTransformer.to_gaussian_input(self.test_set, self.uv_samples, days)

        # Step 3: Fit copula using the transformed data
        copula_estimator = CopulaEstimator(self.copula_type)
            # print(input_matrix.var())
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

        copula_model = self.fitted_copula  # already the fitted copula object
        if not hasattr(copula_model, "sample"):
            raise AttributeError("Fitted copula object has no sample method.")

        # Get symbols from the copula input (based on what was modeled)
        symbols = copula_model.columns if hasattr(copula_model, "columns") else self.test_set[
            'sym_root'].unique().tolist()

        # Step 1: Sample from the copula in Gaussian space
        z_samples = copula_model.sample(n_trajectories)
        z_samples.columns = symbols

        # Step 2: Gaussian → Uniform space
        u_samples = pd.DataFrame(norm.cdf(z_samples), columns=symbols)

        # Step 3: Invert marginal distributions using univariate model samples
        final_samples = pd.DataFrame(index=range(n_trajectories), columns=symbols)

        for symbol in symbols:
            logger.info(f"Inverting marginal forecast for {symbol}")
            forecast_samples = np.sort(self.uv_samples[symbol])
            percentiles = np.linspace(0, 1, len(forecast_samples))
            # Interpolate each u to its quantile value
            final_samples[symbol] = np.interp(u_samples[symbol], percentiles, forecast_samples)

        logger.info("Sample generation completed successfully")
        return final_samples

    def evaluate(self, samples):
        """
        Evaluate the generated samples with Energy Score and Copula Energy Score.
        """
        evaluator = ForecastEvaluator(self.test_set)
        return evaluator.evaluate_energy_score(samples)

    def show_data(self):
        for symbol in self.train_set['sym_root'].unique():
            # Get the corresponding train and test data for the symbol
            train_data = self.train_set[self.train_set['sym_root'] == symbol]
            test_data = self.test_set[self.test_set['sym_root'] == symbol]

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

