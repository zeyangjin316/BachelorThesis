import logging
import pandas as pd
from datetime import datetime
from typing import Union
from copula_method.univariate_models import UnivariateModel
from copula_method.copula_fitting import CopulaEstimator

logger = logging.getLogger(__name__)

class TwoStepModel:
    def __init__(self, file_path: str = "data_for_kit.csv", split_point: Union[float, datetime] = 0.8,
                 univariate_type: str = "ARMAGARCH", copula_type: str ="Gaussian"):
        logger.info("Initializing two-step model")
        self.file_path = file_path
        self.split_point = split_point
        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        self.univariate_type = univariate_type
        self.copula_type = copula_type
        self._build()
        logger.info("Two-step model initialized")

    def _build(self) -> None:
        logger.info("Trying to fetch data")
        self.data = self._get_data()
        logger.info("Data fetched successfully")

        logger.info("Starting data splitting")
        self._split_data()
        logger.info("Data splitting completed")

    def _get_data(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            raise ValueError(f"Error reading data from {self.file_path}: {str(e)}")

    def _split_data(self) -> None:
        """
        Split the data into training and test sets based on split_point, handling each time series individually.
        """
        logger.info("Splitting data with split_point: %s", self.split_point)

        # Initialize lists to store the train and test sets
        train_dfs = []
        test_dfs = []

        # Loop through each unique time series (symbol)
        for symbol in self.data['symbol'].unique():
            symbol_df = self.data[self.data['symbol'] == symbol]

            if isinstance(self.split_point, float):
                # Split based on percentage
                split_idx = int(len(symbol_df) * self.split_point)
                train_dfs.append(symbol_df.iloc[:split_idx])
                test_dfs.append(symbol_df.iloc[split_idx:])

            elif isinstance(self.split_point, datetime):
                # Split based on datetime
                train_dfs.append(symbol_df[symbol_df['date'] <= self.split_point])
                test_dfs.append(symbol_df[symbol_df['date'] > self.split_point])

            else:
                raise ValueError("split_point must be either float or datetime")

        # Concatenate all train and test sets, ensuring temporal order is preserved
        self.train_set = pd.concat(train_dfs).sort_index()
        self.test_set = pd.concat(test_dfs).sort_index()

    def _fit_univariate(self):
        logger.info(f"Fit univariate models of type: {self.univariate_type}")
        self.univariate_models = UnivariateModel(
            data_input=self.data,
            split_point=0.8,
            method=self.univariate_type
        )
        return self.univariate_models.fit()


    def _fit_copula(self):
        logger.info(f"Fit copula of type: {self.copula_type}")
        copula_estimator = CopulaEstimator(
            data_input=self.data,
            split_point=0.8,
            method=self.copula_type,
            features=['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open']
        )
        self.fitted_copula, self.copula_marginals = copula_estimator.run()

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

