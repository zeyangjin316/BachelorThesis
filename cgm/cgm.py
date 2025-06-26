import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union
from tqdm import tqdm

from config import TARGET_VAR
from data_handling import DataHandler
from evaluator import ForecastEvaluator
from cgm import prepare_cgm_inputs_for_sampling
from cgm import CGMTrainer

logger = logging.getLogger(__name__)


class CGMModel:
    def __init__(self, split_point: Union[float, datetime] = 0.8, train_freq: int = 7, train_window_size = 20,
                 loss_type: str = "ES"):
        logger.info("Initializing CGM model")
        self.split_point = split_point
        self.train_freq = train_freq
        self.train_window_size = train_window_size
        self.loss_type = loss_type
        self.data_handler = DataHandler(self.split_point)

        self.data_dict = self.data_handler.get_data(standardize=True)
        self.full_data = self.data_dict['full_data']
        self.train_data = self.data_dict['train_set']
        self.test_data = self.data_dict['test_set']

        self.trained_models = {}

        logger.info("CGM model initialized")
# shape: (n_assets, n_samples)

    def fit(self, n_epochs: int = 100, batch_size: int = 1024):
        logger.info("Starting training CGM models")

        initial_train_dates = self.train_data['date'].drop_duplicates().sort_values().tolist()

        cgm_trainer = CGMTrainer(
            full_data=self.full_data,
            initial_train_dates=initial_train_dates,
            n_epochs=n_epochs,
            batch_size=batch_size,
            train_freq=self.train_freq,
            train_window_size=self.train_window_size  # only used in prepare_cgm_inputs()
        )

        self.trained_models = cgm_trainer.train_all()

        logger.info("Finished training CGM models")

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """
        Sample CGM forecasts for all test days.
        Returns: np.ndarray of shape (n_days, n_assets=10, n_samples)
        """
        logger.info(f"Sampling {n_samples} CGM forecasts for all {len(self.trained_models)} test days")

        all_samples = []
        for test_day, model in tqdm(self.trained_models.items(), desc="Sampling Days"):
            # Get full rolling history for this test day
            history = self.full_data[self.full_data['date'] <= test_day]
            windowed_data = history.groupby('sym_root').tail(self.train_window_size + 1)

            if windowed_data.empty or windowed_data['date'].nunique() < 2:
                logger.warning(f"Skipping {test_day} due to insufficient data")
                continue

            try:
                X_past, X_std, X_all, X_weekday = prepare_cgm_inputs_for_sampling(windowed_data, self.train_window_size)
            except Exception as e:
                logger.warning(f"Skipping {test_day} due to input error: {e}")
                continue

            # Predict CGM samples (shape: (1, dim_out=10, n_samples))
            raw = model.predict([X_past, X_std, X_all, X_weekday], n_samples=n_samples)

            if raw.shape[0] != 1:
                logger.warning(f"Unexpected batch size in prediction on {test_day}")
                continue

            samples = raw[0, :, :]  # shape: (n_assets=10, n_samples)

            # Optionally inverse transform (in-place)
            self.data_handler.scaler.inverse_transform(TARGET_VAR, samples)

            all_samples.append(samples)

        return np.stack(all_samples) if all_samples else np.empty((0, 0, 0))

    def evaluate(self, samples):
        logger.info(f"Evaluating CGM forecasts")

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


