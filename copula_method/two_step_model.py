import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from datetime import datetime
from typing import Union
from scipy.special import erfinv
from copula_method.univariate_models import UnivariateModel
from copula_method.copula_fitting import CopulaEstimator
from reader import Reader

logger = logging.getLogger(__name__)

class TwoStepModel:
    def __init__(self, split_point: Union[float, datetime] = 0.8,
                 univariate_type: str = "ARMAGARCH", copula_type: str ="Gaussian",
                 n_samples: int = 1000):
        logger.info("Initializing two-step model")
        self.reader = Reader()
        self.split_point = split_point
        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        self.univariate_type = univariate_type
        self.copula_type = copula_type
        self.n_samples = n_samples
        self._build()
        logger.info("Two-step model initialized")

    def _build(self) -> None:
        logger.info("Trying to fetch data")
        self.reader.read_data()
        self.reader.merge_all()
        logger.info("Data fetched successfully")

        logger.info("Starting data splitting")
        self.train_set, self.test_set = self.reader.split_data(self.split_point)
        logger.info("Data splitting completed")

    def _fit_univariate_models(self):
        logger.info(f"Fit univariate models of type: {self.univariate_type}")
        # Fit a univariate model for each symbol in the training set
        self.univariate_models = UnivariateModel(
            data=self.train_set,
            method=self.univariate_type
        )
        self.univariate_models.fit()

    def _predict_univariate(self):
        logger.info("Generating samples from univariate models")
        # Create samples for each symbol with the created univariate models
        univariate_samples: dict[str, np.ndarray] = {}
        for symbol in self.train_set['sym_root'].unique():
            try:
                samples = self.univariate_models.sample(symbol, self.n_samples)
                univariate_samples[symbol] = samples
            except Exception as e:
                logger.warning(f"Sample generation failed for {symbol}: {e}")
        logger.info("Finished generating univariate samples")
        return univariate_samples

    def _compute_gaussian_copula_inputs(self, days: list[str], uv_samples: dict[str, np.ndarray]) -> pd.DataFrame:
        logger.info(f"Computing Gaussian copula inputs using empirical sample-based PIT")

        matrix = []

        for day in tqdm(days, desc="Computing copula inputs"):
            test_data_day = self.test_set[self.test_set['date'] == day]
            z_row = {}

            for symbol in test_data_day['sym_root'].unique():
                try:
                    samples = uv_samples[symbol]
                    X_dh = test_data_day[test_data_day['sym_root'] == symbol]['ret_crsp'].values[0]

                    u_dh = np.mean(samples <= X_dh)
                    u_dh = np.clip(u_dh, 1e-6, 1 - 1e-6)

                    #z = norm.ppf(u)
                    Z_dh = np.sqrt(2) * erfinv(2 * u_dh - 1)
                    z_row[symbol] = Z_dh

                except Exception as e:
                    logger.warning(f"Failed to compute Z_d,h for {symbol} on {day}: {e}")
                    continue

            if z_row:
                matrix.append(pd.Series(z_row, name=day))

        df = pd.DataFrame(matrix)
        logger.info(f"Created Gaussian copula input matrix with shape: {df.shape}")
        return df.dropna(axis=1)

    def _fit_copula(self, input_matrix: pd.DataFrame):
        logger.info(f"Fit copula of type: {self.copula_type}")
        copula_estimator = CopulaEstimator(
            method=self.copula_type,
        )
        print(input_matrix.var())
        copula_estimator.fit(input_matrix)
        self.fitted_copula = copula_estimator.fitted_copula
        logger.info(f"Fitted copula of type {self.copula_type} successfully")

    def fit(self):
        logger.info("Starting fitting two-step model")
        # Step 1: Fit univariate models
        self._fit_univariate_models()

        # Step 2: Generate forecast samples
        uv_samples = self._predict_univariate()

        # Step 3: Create Gaussianized copula inputs
        days = sorted(self.test_set['date'].unique())
        gaussian_copula_input = self._compute_gaussian_copula_inputs(days, uv_samples)

        # Step 4: Fit copula using the transformed data
        self._fit_copula(gaussian_copula_input)
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
            forecast_samples = np.sort(self.univariate_models.sample(symbol, self.n_samples))
            percentiles = np.linspace(0, 1, len(forecast_samples))
            # Interpolate each u to its quantile value
            final_samples[symbol] = np.interp(u_samples[symbol], percentiles, forecast_samples)

        logger.info("Sample generation completed successfully")
        return final_samples

    def evaluate_energy_score(self, samples):
        """
        Evaluate the generated samples with Energy Score and Copula Energy Score.
        """
        from scoring_rules_supp import es_sample, ces_sample

        test_dates = sorted(self.test_set['date'].unique())
        symbols = samples.columns.tolist()

        # Reshape the fixed sample matrix: (1, n_dim, n_samples)
        y_pred = samples.T.values[np.newaxis, :, :]  # shape (1, n_dim, n_samples)

        energy_scores = []
        copula_energy_scores = []

        for date in tqdm(test_dates, desc="Evaluating test days"):
            test_day_data = self.test_set[self.test_set['date'] == date]

            try:
                y_true = np.array([
                    test_day_data[test_day_data['sym_root'] == symbol]['ret_crsp'].values[0]
                    for symbol in symbols
                ]).reshape(1, -1)
            except IndexError:
                logger.warning(f"Skipping {date} due to missing values.")
                continue

            es = es_sample(y_true, y_pred)
            ces = ces_sample(y_true, y_pred)

            energy_scores.append(es)
            copula_energy_scores.append(ces)

        mean_es = np.mean(energy_scores)
        mean_ces = np.mean(copula_energy_scores)

        logger.info(f"\nMean Energy Score: {mean_es:.6f}")
        logger.info(f"Mean Copula Energy Score: {mean_ces:.6f}")

        return mean_es, mean_ces

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

