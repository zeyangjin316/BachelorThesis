import pandas as pd

from datetime import datetime
from cgm import prepare_cgm_inputs
from cgm import cgm


class CGMTrainer:
    def __init__(self, train_data: pd.DataFrame,
                 n_epochs: int = 100, batch_size: int = 1024,
                 train_freq: int = 1):
        self.train_data = train_data
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.window_size = train_freq  # size of rolling window
        self.train_freq = train_freq   # frequency of retraining

    def _train_single_on(self, data: pd.DataFrame) -> cgm:
        X_past, X_std, X_all, X_weekday, Y = prepare_cgm_inputs(data)

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

        all_dates = self.train_data['date'].drop_duplicates().sort_values().reset_index(drop=True)

        for i in range(self.window_size, len(all_dates)):
            if (i - self.window_size) % self.train_freq != 0:
                continue

            test_day = all_dates[i]
            start_day = all_dates[i - self.window_size]

            rolling_data = self.train_data[
                (self.train_data['date'] >= start_day) &
                (self.train_data['date'] < test_day)
            ]

            if rolling_data.empty:
                continue

            model = self._train_single_on(rolling_data)
            trained_models[test_day] = model

        return trained_models