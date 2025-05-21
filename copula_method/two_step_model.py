import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Union
from copula_method.univariate_models import UnivariateModel
from copula_method.copula_fitting import CopulaEstimator
from reader import Reader

logger = logging.getLogger(__name__)

class TwoStepModel:
    def __init__(self, split_point: Union[float, datetime] = 0.8,
                 univariate_type: str = "ARMAGARCH", copula_type: str ="Gaussian"):
        logger.info("Initializing two-step model")
        self.reader = Reader()
        self.split_point = split_point
        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        self.univariate_type = univariate_type
        self.copula_type = copula_type
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


    def _fit_univariate(self):
        logger.info(f"Fit univariate models of type: {self.univariate_type}")
        self.univariate_models = UnivariateModel(
            data_input=self.reader.data,
            split_point=0.8,
            method=self.univariate_type
        )
        return self.univariate_models.fit()


    def _fit_copula(self):
        logger.info(f"Fit copula of type: {self.copula_type}")
        copula_estimator = CopulaEstimator(
            data_input=self.reader.data,
            split_point=0.8,
            method=self.copula_type,
            features=['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open']
        )
        self.fitted_copula, self.copula_marginals = copula_estimator.fit()

    def fit(self):
        logger.info("Starting fitting two-step model")
        self._fit_univariate()
        self._fit_copula()
        logger.info("Finished fitting two-step model")

    def predict(self):
        logger.info("Generating data with two-step model")
        logger.info("Finished generating data with two-step model")

    def evaluate(self):
        pass

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

