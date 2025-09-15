from typing import Any, Tuple, Dict
import pandas as pd
from datetime import datetime
from cgm_method import CGMInputBuilder
from cgm_method import cgm, CGMInitConfig, CGMFitConfig
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CGMTrainer:
    def __init__(
        self,
        full_data: pd.DataFrame,
        initial_train_dates: list[datetime],
        cgm_init: CGMInitConfig,
        fit_cfg: CGMFitConfig,
        *,
        std_policy: str = "window",
    ):
        """
        full_data : DataFrame with all available data
        initial_train_dates : list of training dates
        cgm_init : CGM initialization config
        fit_cfg : CGM fit config
        std_policy : {"window","full"}
        """
        assert std_policy in {"window", "full"}

        self.full_data = full_data
        self.initial_train_dates = initial_train_dates
        self.cgm_init = cgm_init
        self.cfg = fit_cfg
        self.std_policy = std_policy
        self.rolling_days = len(initial_train_dates)

        self.trained_models: Dict[Any, cgm] = {}
        self.builders: Dict[Any, CGMInputBuilder] = {}

    def _train_single_on(self, data: pd.DataFrame) -> Tuple[cgm, CGMInputBuilder]:
        """Train a single CGM model on a rolling window of data."""
        builder = CGMInputBuilder(
            window_size=self.cfg.train_window_size,
            std_policy=self.std_policy,
        )
        X_past, X_std, X_all, X_weekday, Y = builder.fit_prepare(data)

        for name, arr in {"X_past": X_past, "X_std": X_std, "X_all": X_all, "Y": Y}.items():
            if not np.isfinite(arr).all():
                bad = np.isnan(arr).sum() + np.isinf(arr).sum()
                print(f"{name} has {bad} NaN/Inf values")

        dim_out, dim_in_features, dim_in_past = builder.model_dims()
        model = cgm(
            dim_out=dim_out,
            dim_in_features=dim_in_features,
            dim_in_past=dim_in_past,
            dim_latent=self.cgm_init.dim_latent,
            n_samples_train=self.cgm_init.n_samples_train,
            emb_size=self.cgm_init.emb_size,
            past_len=self.cfg.train_window_size,
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
        return model, builder

    def train_all(self) -> Tuple[Dict[Any, cgm], Dict[Any, CGMInputBuilder]]:
        """Train rolling CGM models for all test days and return models and builders."""
        trained_models: Dict[Any, cgm] = {}
        builders: Dict[Any, CGMInputBuilder] = {}

        all_dates = (
            self.full_data["date"].drop_duplicates().sort_values().reset_index(drop=True)
        )
        total_steps = len(all_dates) - self.rolling_days

        last_model = None
        last_builder = None

        for i in range(self.rolling_days, len(all_dates)):
            test_day = all_dates[i]
            start_day = all_dates[i - self.rolling_days]
            end_day = all_dates[i]

            rolling_data = self.full_data[
                (self.full_data["date"] >= start_day) & (self.full_data["date"] < end_day)
            ]
            if rolling_data.empty:
                logger.warning(f"No rolling data for test day {test_day}")
                continue

            if (i - self.rolling_days) % self.cfg.train_freq == 0:
                days_left = total_steps - (i - self.rolling_days)
                logger.info(f"Training CGM model for {test_day} ({days_left} days left)")
                last_model, last_builder = self._train_single_on(rolling_data)

            trained_models[test_day] = last_model
            builders[test_day] = last_builder

        self.trained_models = trained_models
        self.builders = builders

        logger.info(
            f"Trained {len(trained_models)} models for {total_steps} days: {list(trained_models.keys())}"
        )
        return trained_models, builders
