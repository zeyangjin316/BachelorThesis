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
from cgm_method import CGMInitConfig, CGMFitConfig, CGMSampleConfig

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    split_point: Union[float, datetime] = 0.8
    standardize: bool = True

class CGMExperiment:
    def __init__(self,
                 data_cfg: DataConfig,
                 cgm_init: CGMInitConfig,
                 train_cfg: CGMFitConfig,
                 pred_cfg: CGMSampleConfig):
        """Experiment container for data, training, sampling, and evaluation."""
        self.data_cfg = data_cfg
        self.cgm_init = cgm_init
        self.train_cfg = train_cfg
        self.pred_cfg = pred_cfg

        self.data_handler = DataHandler(self.data_cfg.split_point)
        self.data_dict = self.data_handler.get_data(
            exclude_pandemic=True,
            filter_duplicates=True,
            save_df=False,
            save_png=False,  # enable PNG
            png_path="results/full_data_preview.png",  # custom path
            png_head_n=50,  # show first 50 rows
            png_dpi=200,
        )
        self.full_data = self.data_dict['full_data']
        self.train_data = self.data_dict['train_set']
        self.test_data = self.data_dict['test_set']

        self.trained_models: Dict[Any, Any] = {}
        self.builders: Dict[Any, CGMInputBuilder] = {}

    def fit(self):
        """Training orchestration over rolling windows."""
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
        """Sampling using trained models and stored builders."""
        n_samples = self.pred_cfg.n_samples
        all_samples = []

        for test_day, model in tqdm(self.trained_models.items(), desc="Sampling Days"):
            history = self.full_data[self.full_data['date'] <= test_day]
            windowed_data = history.groupby('sym_root').tail(self.train_cfg.train_window_size + 1)
            if windowed_data.empty or windowed_data['date'].nunique() < 2:
                continue

            builder = self.builders[test_day]
            X_past, X_std, X_all, X_weekday = builder.prepare_for_sampling(windowed_data)

            raw = model.predict(
                [X_past, X_std, X_all, X_weekday],
                n_samples=n_samples,
                verbose=self.pred_cfg.verbose
            )
            samples_scaled = raw[0, :, :]
            samples = builder.scaler.inverse_transform(TARGET_VAR, samples_scaled)

            all_samples.append(samples)
            #print("Scaled forecast min/max:", raw.min(), raw.max())
            #print("Inverse forecast min/max:", samples.min(), samples.max())
            #print("Realized min/max:", self.test_data[TARGET_VAR].min(), self.test_data[TARGET_VAR].max())

        return np.stack(all_samples) if all_samples else np.empty((0, 0, 0))

    def evaluate(self, samples):
        """Evaluation of forecast samples."""
        return ForecastEvaluator(self.test_data, samples).evaluate()

    def show_data(self):
        """Visualization of train/test split for each symbol."""
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
