from typing import Any, Tuple, Dict
import pandas as pd
from datetime import datetime
from cgm_method import CGMInputBuilder
from cgm_method import cgm, CGMInitConfig, CGMFitConfig
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CGMTrainer:
    """
    Rolling-window trainer for the Conditional Generative Model (CGM).

    Trains (and reuses) CGM models across test days according to a refit
    frequency. For each test date, stores the most recent trained model
    and its corresponding input builder.

    Parameters
    ----------
    full_data : pd.DataFrame
        Full merged dataset containing at least ['date', 'sym_root', target].
    initial_train_dates : list[datetime]
        Ordered list of dates defining the initial training window length.
        Its length determines the rolling window size.
    cgm_init : CGMInitConfig
        Initialization/config for the CGM architecture.
    fit_cfg : CGMFitConfig
        Training configuration (epochs, batch size, window, refit freq, etc.).
    std_policy : {"window","full"}, default="window"
        Standardization policy used by CGMInputBuilder.
    """

    def __init__(
        self,
        full_data: pd.DataFrame,
        initial_train_dates: list[datetime],
        cgm_init: CGMInitConfig,
        fit_cfg: CGMFitConfig,
        *,
        std_policy: str = "window",
    ):
        if std_policy not in {"window", "full"}:
            raise ValueError("std_policy must be 'window' or 'full'.")

        self.full_data = full_data
        self.initial_train_dates = initial_train_dates
        self.cgm_init = cgm_init
        self.cfg = fit_cfg
        self.std_policy = std_policy
        self.rolling_days = len(initial_train_dates)

        self.trained_models: Dict[Any, cgm] = {}
        self.builders: Dict[Any, CGMInputBuilder] = {}

    def _train_single_on(self, data: pd.DataFrame) -> Tuple[cgm, CGMInputBuilder]:
        """
        Train a single CGM on a given rolling-window slice.

        Parameters
        ----------
        data : pd.DataFrame
            Windowed training data (per-symbol panel up to the test day).

        Returns
        -------
        model : cgm
            Fitted CGM model instance.
        builder : CGMInputBuilder
            Builder used to create inputs/scalers for this window.
        """
        builder = CGMInputBuilder(
            window_size=self.cfg.train_window_size,
            std_policy=self.std_policy,
        )
        X_past, X_std, X_all, X_weekday, Y = builder.fit_prepare(data)

        # quick NaN/Inf check to catch upstream issues early
        for name, arr in {"X_past": X_past, "X_std": X_std, "X_all": X_all, "Y": Y}.items():
            if not np.isfinite(arr).all():
                bad = np.isnan(arr).sum() + np.isinf(arr).sum()
                logger.warning("%s has %d NaN/Inf values", name, int(bad))

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
        """
        Train models across all test days using a rolling window and refit schedule.

        For each test day, either refits a new model (per `train_freq`) or reuses
        the last trained model. Returns dictionaries keyed by test date.

        Returns
        -------
        trained_models : dict[datetime, cgm]
            Latest model applicable for each test day.
        builders : dict[datetime, CGMInputBuilder]
            Matching input builders to prepare sampling inputs later.
        """
        trained_models: Dict[Any, cgm] = {}
        builders: Dict[Any, CGMInputBuilder] = {}

        all_dates = self.full_data["date"].drop_duplicates().sort_values().reset_index(drop=True)
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
                logger.warning("No rolling data for test day %s", test_day)
                continue

            # refit at the configured frequency; otherwise reuse last model
            if (i - self.rolling_days) % self.cfg.train_freq == 0:
                days_left = total_steps - (i - self.rolling_days)
                logger.info("Training CGM model for %s (%d days left)", test_day, max(days_left, 0))
                last_model, last_builder = self._train_single_on(rolling_data)

            trained_models[test_day] = last_model
            builders[test_day] = last_builder

        self.trained_models = trained_models
        self.builders = builders

        logger.info(
            "Trained %d models for %d days.", len(trained_models), max(total_steps, 0)
        )
        return trained_models, builders
