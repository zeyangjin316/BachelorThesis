import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Union

from data_handling import DataHandler
from evaluator import ForecastEvaluator
from cgm.data_prep import prepare_cgm_inputs
from cgm.cgm_model import cgm

logger = logging.getLogger(__name__)


class CGMModel:
    def __init__(self, split_point: Union[float, datetime] = 0.8, window_size: int = 7, loss_type: str = "ES"):
        logger.info("Initializing CGM model")
        self.split_point = split_point
        self.window_size = window_size
        self.loss_type = loss_type

        # Collecting and splitting data
        self.data_dict = self._get_data()
        self.full_data = self.data_dict['full_data']
        self.train_data = self.data_dict['train_set']
        self.test_data = self.data_dict['test_set']

        logger.info("CGM model initialized")

    def _get_data(self):
        data_handler = DataHandler(self.split_point)
        full_data = data_handler.get_data(split=False)
        train_set, test_set = data_handler.get_data(split=True)
        """with pd.option_context('display.max_columns', None):
            print(full_data.head())"""
        return {'full_data': full_data, 'train_set': train_set, 'test_set': test_set, 'split_point': self.split_point, }

    def fit(self, n_epochs: int = 100, batch_size: int = 1024):
        logger.info("Starting CGM model training")

        X_past, X_std, X_all, X_weekday, Y = prepare_cgm_inputs(self.train_data)

        dim_out = Y.shape[1]
        dim_in_past = X_past.shape[2]
        dim_in_features = X_all.shape[1]

        self.cgm_model = cgm(
            dim_out=dim_out,
            dim_in_features=dim_in_features,
            dim_in_past=dim_in_past,
            dim_latent=50,
            n_samples_train=100
        )

        self.cgm_model.fit(
            x=[X_past, X_std, X_all, X_weekday],
            y=Y,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=1
        )

        logger.info("Finished training CGM model")

    def sample(self, n_samples: int = 1000):
        logger.info(f"Generating {n_samples} samples from CGM")

        X_past, X_std, X_all, X_weekday, _ = prepare_cgm_inputs(self.test_data)

        samples = self.cgm_model.predict(
            x_test=[X_past, X_std, X_all, X_weekday],
            n_samples=n_samples
        )

        return samples  # shape: (n_days, 10, n_samples)

    def evaluate(self, samples):
        """
        Evaluate the generated samples with Energy Score and Copula Energy Score.
        """
        logger.info(f"Evaluating cgm method")
        evaluator = ForecastEvaluator(self.test_data, samples)
        return evaluator.evaluate()

    def show_data(self):
        for symbol in self.train_data['sym_root'].unique():
            train_data = self.train_data[self.train_data['sym_root'] == symbol]
            test_data = self.test_data[self.test_data['sym_root'] == symbol]

            plt.figure(figsize=(10, 6))
            plt.plot(train_data['date'], train_data['ret_crsp'], label='Train', color='blue')
            plt.plot(test_data['date'], test_data['ret_crsp'], label='Test', color='red')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title(f"'ret_crsp' Split for {symbol}")
            plt.legend()
            plt.show()


