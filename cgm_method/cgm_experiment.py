import logging
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Optional, Dict, Any
from tqdm import tqdm

from config import TARGET_VAR
from data_handling import DataHandler
from evaluator import ForecastEvaluator
from cgm_method import prepare_cgm_inputs_for_sampling, CGMDataConfig
from cgm_method import CGMTrainer
from cgm_method import CGMInitConfig, CGMFitConfig, CGMSampleConfig

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    split_point: Union[float, datetime] = 0.8
    standardize: bool = True

class CGMExperiment:
    def __init__(self,
                 data_cfg: DataConfig = CGMDataConfig(),
                 cgm_init: CGMInitConfig = CGMInitConfig(),
                 train_cfg: CGMFitConfig = CGMFitConfig(),
                 pred_cfg: CGMSampleConfig = CGMSampleConfig()):
        self.data_cfg = data_cfg
        self.cgm_init = cgm_init
        self.train_cfg = train_cfg
        self.pred_cfg = pred_cfg

        self.data_handler = DataHandler(self.data_cfg.split_point)
        self.data_dict = self.data_handler.get_data(standardize=self.data_cfg.standardize)
        self.full_data = self.data_dict['full_data']
        self.train_data = self.data_dict['train_set']
        self.test_data = self.data_dict['test_set']
        self.trained_models: Dict[Any, Any] = {}

    def fit(self):
        initial_train_dates = self.train_data['date'].drop_duplicates().sort_values().tolist()
        trainer = CGMTrainer(
            full_data=self.full_data,
            initial_train_dates=initial_train_dates,
            cgm_init=self.cgm_init,
            fit_cfg=self.train_cfg,
        )
        self.trained_models = trainer.train_all()

    def sample(self) -> np.ndarray:
        n_samples = self.pred_cfg.n_samples
        all_samples = []
        for test_day, model in tqdm(self.trained_models.items(), desc="Sampling Days"):
            history = self.full_data[self.full_data['date'] <= test_day]
            windowed_data = history.groupby('sym_root').tail(self.train_cfg.train_window_size + 1)
            if windowed_data.empty or windowed_data['date'].nunique() < 2:
                continue
            X_past, X_std, X_all, X_weekday = prepare_cgm_inputs_for_sampling(
                windowed_data, self.train_cfg.train_window_size
            )
            raw = model.predict([X_past, X_std, X_all, X_weekday],
                                n_samples=n_samples,
                                verbose=self.pred_cfg.verbose)
            samples = raw[0, :, :]
            self.data_handler.scaler.inverse_transform(TARGET_VAR, samples)
            all_samples.append(samples)
        return np.stack(all_samples) if all_samples else np.empty((0, 0, 0))

    def evaluate(self, samples):
        return ForecastEvaluator(self.test_data, samples).evaluate()

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


