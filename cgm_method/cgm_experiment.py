import logging
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Dict, Any
from tqdm import tqdm

from config import TARGET_VAR
from data.data_handling import DataHandler
from evaluator import ForecastEvaluator
from cgm_method import CGMInputBuilder, CGMDataConfig
from cgm_method import CGMTrainer
from cgm_method import CGMInitConfig, CGMFitConfig, CGMSampleConfig, CGMDataConfig

logger = logging.getLogger(__name__)

class CGMExperiment:
    """
    Experiment container for the CGM method:
    handles data preparation, training, sampling, and evaluation.
    """

    def __init__(self,
                 data_cfg: CGMDataConfig,
                 cgm_init: CGMInitConfig,
                 train_cfg: CGMFitConfig,
                 pred_cfg: CGMSampleConfig):
        """
        Initialize experiment settings and load data.

        Parameters
        ----------
        data_cfg : CGMDataConfig
            Data configuration (split point, filters, exclusions).
        cgm_init : CGMInitConfig
            Model initialization settings.
        train_cfg : CGMFitConfig
            Training configuration (epochs, batch size, etc.).
        pred_cfg : CGMSampleConfig
            Sampling configuration (number of samples, verbosity).
        """
        self.data_cfg = data_cfg
        self.cgm_init = cgm_init
        self.train_cfg = train_cfg
        self.pred_cfg = pred_cfg

        # Prepare data splits
        self.data_handler = DataHandler(self.data_cfg.split_point)
        self.data_dict = self.data_handler.get_data(
            exclude_pandemic=True,
            filter_duplicates=True,
        )
        self.full_data = self.data_dict['full_data']
        self.train_data = self.data_dict['train_set']
        self.test_data = self.data_dict['test_set']

        # Storage for trained models and input builders
        self.trained_models: Dict[Any, Any] = {}
        self.builders: Dict[Any, CGMInputBuilder] = {}

    def fit(self):
        """
        Train CGM models over rolling windows.
        Stores trained models and builders.
        """
        initial_train_dates = self.train_data['date'].drop_duplicates().sort_values().tolist()
        logger.info(f"Initializing models for {len(initial_train_dates)} days")

        trainer = CGMTrainer(
            full_data=self.full_data,
            initial_train_dates=initial_train_dates,
            cgm_init=self.cgm_init,
            fit_cfg=self.train_cfg,
            std_policy=getattr(self, "std_policy", "window"),
        )
        logger.info("Training CGM models")
        self.trained_models, self.builders = trainer.train_all()

    def sample(self) -> np.ndarray:
        """
        Generate forecast samples using trained models.

        Returns
        -------
        np.ndarray
            Forecast samples with shape (T, S, N), or empty array if none.
        """
        n_samples = self.pred_cfg.n_samples
        all_samples = []

        for test_day, model in tqdm(self.trained_models.items(), desc="Sampling Days"):
            # Select rolling window history
            history = self.full_data[self.full_data['date'] <= test_day]
            windowed_data = history.groupby('sym_root').tail(self.train_cfg.train_window_size + 1)
            if windowed_data.empty or windowed_data['date'].nunique() < 2:
                continue

            # Prepare inputs for sampling
            builder = self.builders[test_day]
            X_past, X_std, X_all, X_weekday = builder.prepare_for_sampling(windowed_data)

            # Generate and inverse-transform forecasts
            raw = model.predict(
                [X_past, X_std, X_all, X_weekday],
                n_samples=n_samples,
                verbose=self.pred_cfg.verbose
            )
            samples_scaled = raw[0, :, :]
            samples = builder.scaler.inverse_transform(TARGET_VAR, samples_scaled)

            all_samples.append(samples)

        return np.stack(all_samples) if all_samples else np.empty((0, 0, 0))

    def evaluate(self, samples):
        """
        Evaluate forecast samples against the test set.

        Parameters
        ----------
        samples : np.ndarray
            Forecast samples.

        Returns
        -------
        dict
            Evaluation metrics (ES, VS, DSS, CRPS).
        """
        return ForecastEvaluator(self.test_data, samples).evaluate()

