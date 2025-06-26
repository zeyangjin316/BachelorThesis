import pandas as pd
from datetime import datetime
from cgm import prepare_cgm_inputs, cgm
import logging

logger = logging.getLogger(__name__)

class CGMTrainer:
    def __init__(self,
                 full_data: pd.DataFrame,
                 initial_train_dates: list[datetime],
                 n_epochs: int = 100,
                 batch_size: int = 1024,
                 train_freq: int = 1,
                 train_window_size: int = 20):
        """
        Parameters:
            full_data: Complete dataset including both train and test dates.
            initial_train_dates: List of unique dates defining the initial training window.
            n_epochs: Training epochs for CGM.
            batch_size: Batch size for training.
            train_freq: Retrain frequency (every X test days).
            train_window_size: Passed to `prepare_cgm_inputs()` only (not related to date range).
        """
        self.full_data = full_data
        self.initial_train_dates = initial_train_dates
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.window_size = train_window_size
        self.rolling_days = len(initial_train_dates)

    def _train_single_on(self, data: pd.DataFrame) -> cgm:
        X_past, X_std, X_all, X_weekday, Y = prepare_cgm_inputs(data, self.window_size)

        dim_out = Y.shape[1]
        dim_in_past = X_past.shape[2]
        dim_in_features = X_all.shape[1]

        model = cgm(
            dim_out=dim_out,
            dim_in_features=dim_in_features,
            dim_in_past=dim_in_past,
            dim_latent=50,
            n_samples_train=100
        )

        model.fit(
            x=[X_past, X_std, X_all, X_weekday],
            y=Y,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=1
        )

        return model

    def train_all(self) -> dict[datetime, cgm]:
        trained_models = {}
        all_dates = self.full_data['date'].drop_duplicates().sort_values().reset_index(drop=True)
        total_steps = len(all_dates) - self.rolling_days

        last_model = None
        for i in range(self.rolling_days, len(all_dates)):
            test_day = all_dates[i]
            start_day = all_dates[i - self.rolling_days]
            end_day = all_dates[i]  # exclusive

            rolling_data = self.full_data[
                (self.full_data['date'] >= start_day) &
                (self.full_data['date'] < end_day)
                ]

            if rolling_data.empty:
                logger.warning(f"No rolling data for test day {test_day}")
                continue

            if (i - self.rolling_days) % self.train_freq == 0:
                days_left = total_steps - (i - self.rolling_days)
                logger.info(f"Training CGM model for {test_day} ({days_left} days left)")
                last_model = self._train_single_on(rolling_data)
            else:
                logger.debug(f"Reusing model for {test_day}")

            trained_models[test_day] = last_model

        return trained_models