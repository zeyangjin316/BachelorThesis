import pandas as pd
from datetime import datetime
from cgm_method import prepare_cgm_inputs
from cgm_method import cgm, CGMInitConfig, CGMFitConfig
import logging

logger = logging.getLogger(__name__)

class CGMTrainer:
    def __init__(self,
                 full_data: pd.DataFrame,
                 initial_train_dates: list[datetime],
                 cgm_init: CGMInitConfig = CGMInitConfig(),
                 fit_cfg: CGMFitConfig = CGMFitConfig):
        self.full_data = full_data
        self.initial_train_dates = initial_train_dates
        self.cgm_init = cgm_init
        self.cfg = fit_cfg
        self.rolling_days = len(initial_train_dates)

    def _train_single_on(self, data: pd.DataFrame) -> cgm:
        X_past, X_std, X_all, X_weekday, Y = prepare_cgm_inputs(data, self.cfg.train_window_size)
        dim_out = Y.shape[1]
        dim_in_past = X_past.shape[2]
        dim_in_features = X_all.shape[1]

        model = cgm(
            dim_out=dim_out,
            dim_in_features=dim_in_features,
            dim_in_past=dim_in_past,
            dim_latent=self.cgm_init.dim_latent,
            n_samples_train=self.cgm_init.n_samples_train,
            emb_size=self.cgm_init.emb_size,
        )

        model.fit(
            x=[X_past, X_std, X_all, X_weekday],
            y=Y,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.n_epochs,
            verbose=self.cfg.verbose,
            callbacks=self.cfg.callbacks,
            validation_split=self.cfg.validation_split,
            validation_data=self.cfg.validation_data,
            sample_weight=self.cfg.sample_weight,
            learningrate=self.cfg.learningrate,
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
            end_day = all_dates[i]
            rolling_data = self.full_data[(self.full_data['date'] >= start_day) &
                                          (self.full_data['date'] < end_day)]
            if rolling_data.empty:
                logger.warning(f"No rolling data for test day {test_day}")
                continue
            if (i - self.rolling_days) % self.cfg.train_freq == 0:
                days_left = total_steps - (i - self.rolling_days)
                logger.info(f"Training CGM model for {test_day} ({days_left} days left)")
                last_model = self._train_single_on(rolling_data)
            trained_models[test_day] = last_model
        return trained_models