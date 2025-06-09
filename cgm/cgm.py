import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Union

from data_handling import DataHandler
from evaluator import ForecastEvaluator

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from cgm_model import cgm

logger = logging.getLogger(__name__)

class CGMModel:
    def __init__(self, split_point: Union[float, datetime] = 0.8, loss_type: str = "ES"):
        logger.info("Initializing CGM model")

        # Store configuration
        self.split_point = split_point
        self.loss_type = loss_type

        # Load data via DataHandler (same as TwoStepModel)
        self.data_handler = DataHandler(split_point)
        self.full_data = self.data_handler.get_data(split=False)
        self.train_set, self.test_set = self.data_handler.get_data(split=True)

        # Placeholder for trained model(s)
        self.cgm_model = None
        self.ensemble_models = []

        logger.info("CGM model initialized")

    def fit(self, n_epochs: int = 100, batch_size: int = 1024, ensemble_size: int = 10):
        logger.info("Starting CGM model training")

        # For each ensemble member:
        for i in range(ensemble_size):
            logger.info(f"Training CGM ensemble member {i+1}/{ensemble_size}")

            # Initialize CGM model (replace with your function or class)
            model = cgm(loss_type=self.loss_type)

            # Compile with optimizer and loss (energy score or custom)
            model.compile(optimizer=Adam(learning_rate=1e-4))

            # Prepare training data (you will implement this part)
            X_train, Y_train = self.prepare_training_data()

            # Fit the model
            model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size)

            # Store trained model
            self.ensemble_models.append(model)

        logger.info("Finished training CGM model")

    def sample(self, n_samples: int = 1000):
        logger.info(f"Generating {n_samples} multivariate samples using CGM")

        # Placeholder: implement sampling from ensemble models
        # Should return: array of (n_samples, path_length=10) samples
        samples = []

        for model in self.ensemble_models:
            # Implement sampling logic per model
            samples_model = self.generate_samples_from_model(model, n_samples)
            samples.append(samples_model)

        # Combine all samples
        combined_samples = np.concatenate(samples, axis=0)

        logger.info("Finished generating samples")
        return combined_samples

    def evaluate(self, samples):
        logger.info("Evaluating CGM samples")

        # Use your existing ForecastEvaluator
        evaluator = ForecastEvaluator(self.test_set)
        return evaluator.evaluate_energy_score(samples)

    def show_data(self):
        for symbol in self.train_set['sym_root'].unique():
            train_data = self.train_set[self.train_set['sym_root'] == symbol]
            test_data = self.test_set[self.test_set['sym_root'] == symbol]

            plt.figure(figsize=(10, 6))
            plt.plot(train_data['date'], train_data['ret_crsp'], label='Train', color='blue')
            plt.plot(test_data['date'], test_data['ret_crsp'], label='Test', color='red')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title(f"'ret_crsp' Split for {symbol}")
            plt.legend()
            plt.show()

    # Placeholder methods to implement:
    def prepare_training_data(self):
        # You will implement this using your CGM input pipeline
        pass

    def generate_samples_from_model(self, model, n_samples):
        # You will implement this using your CGM sampling logic
        pass

