import logging
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Union

from config import TARGET_VAR
from data_handling import DataHandler
from evaluator import ForecastEvaluator
from cgm import prepare_cgm_inputs
from cgm import CGMTrainer

logger = logging.getLogger(__name__)


class CGMModel:
    def __init__(self, split_point: Union[float, datetime] = 0.8, train_freq: int = 7, loss_type: str = "ES"):
        logger.info("Initializing CGM model")
        self.split_point = split_point
        self.train_freq = train_freq
        self.loss_type = loss_type
        self.data_handler = DataHandler(self.split_point)

        self.data_dict = self.data_handler.get_data(standardize=True)
        self.full_data = self.data_dict['full_data']
        self.train_data = self.data_dict['train_set']
        self.test_data = self.data_dict['test_set']

        self.trained_models = {}

        logger.info("CGM model initialized")

    def fit(self, n_epochs: int = 100, batch_size: int = 1024):
        logger.info("Starting training CGM models")

        cgm_trainer = CGMTrainer(train_data=self.full_data,
                                 n_epochs=n_epochs,
                                 batch_size=batch_size,
                                 train_freq=self.train_freq)

        self.trained_models = cgm_trainer.train_all()

        logger.info("Finished training CGM models")

    def sample(self, test_day: datetime, n_samples: int = 1000):
        logger.info(f"Generating {n_samples} samples for {test_day}")

        model = self.trained_models.get(test_day)
        if model is None:
            raise ValueError(f"No trained model available for test day {test_day}")

        test_data = self.test_data[self.test_data['date'] == test_day]
        X_past, X_std, X_all, X_weekday, _ = prepare_cgm_inputs(test_data)

        samples = model.predict(
            x_test=[X_past, X_std, X_all, X_weekday],
            n_samples=n_samples
        )

        self.data_handler.scaler.inverse_transform(TARGET_VAR, samples)
        return samples

    def evaluate(self, test_day: datetime, samples):
        logger.info(f"Evaluating CGM forecast for {test_day}")

        actuals = self.test_data[self.test_data['date'] == test_day]
        evaluator = ForecastEvaluator(actuals, samples)
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


