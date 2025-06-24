import pandas as pd
import logging as log
import numpy as np
import os
from tqdm import tqdm
from rpy2.robjects import pandas2ri, r
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

rpy2_logger.setLevel(log.ERROR)

class UnivariateModel():
    def __init__(self, data: pd.DataFrame, method: str = "ARMAGARCH"):
        self.data = data
        self.method = method
        self.fitted_models = {}

    def _fit_arma_garch(self, time_series):
        pandas2ri.activate()

        # Convert to R format
        r_timeseries = pandas2ri.py2rpy(time_series)

        # Fit model

        current_dir = os.path.dirname(os.path.abspath(__file__))
        r_script_path = os.path.join(current_dir, "arma_garch.R")
        r.source(r_script_path)
        fitted_model = r['fit_arma_garch'](r_timeseries)

        # Return only R model objects
        return {
            'arma_model': fitted_model.rx2('arma_model'),
            'garch_model': fitted_model.rx2('garch_model')
        }

    def _fit_lasso(self, symbol_data):
        """Fit LASSO model to symbol data"""
        pandas2ri.activate()

        # Prepare data
        y = symbol_data['ret_crsp']
        exclude = ['ret_crsp', 'sym_root', 'date', 'permno']  # exclude target + identifiers

        # Dynamically select features
        features = [col for col in symbol_data.columns if col not in exclude]
        x = symbol_data[features]
        # Convert to R format
        r_y = pandas2ri.py2rpy(y)
        r_x = pandas2ri.py2rpy(x)

        # Fit model
        r.source('lasso.R')
        fitted_model = r['fit_lasso'](r_y, r_x)

        # Return only model object and features
        return {
            'model': fitted_model.rx2('model'),
            'features_used': features
        }

    def _sample_arma_garch(self, symbol, n_samples):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        r_script_path = os.path.join(current_dir, "arma_garch.R")
        r.source(r_script_path)
        sample_func = r['forecast_arma_garch_samples']
        model = self.fitted_models[symbol]
        r_result = sample_func(model["arma_model"], model["garch_model"], n_samples)
        return np.array(r_result)

    def _sample_lasso(self, symbol, n_samples):
        # Assume homoskedastic residuals for LASSO sampling
        model = self.fitted_models[symbol]
        y_pred = model['y_hat']
        residual_std = model['residual_std']
        return np.random.normal(loc=y_pred, scale=residual_std, size=n_samples)

    def fit(self, current_day=None):
        """
        Train univariate models for all symbols in the data.

        For each symbol in self.data['sym_root']:
        - A univariate model of the chosen type (ARMAGARCH or LASSO) is fitted to the data for that symbol.
        - The fitted models are stored in self.fitted_models:
            self.fitted_models[symbol] → dict with fitted model components.

        Parameters
        ----------
        current_day : str or pd.Timestamp, optional
            If provided, will log which test day this model is being fitted for (for rolling window scenarios).

        Returns
        -------
        None
        """
        #if current_day is not None:
        #    tqdm.write(f"[INFO] Fitting univariate models for test day {current_day}")
        #else:
        #    tqdm.write(f"[INFO] Fitting univariate models for all symbols")

        def train_model(symbol):
            symbol_data = self.data[self.data['sym_root'] == symbol]

            try:
                if self.method == "ARMAGARCH":
                    log.info(f" Completed training {self.method} model for symbol {symbol} on day {current_day}")
                    model = self._fit_arma_garch(symbol_data['ret_crsp'])
                elif self.method == "LASSO":
                    model = self._fit_lasso(symbol_data)
                else:
                    raise ValueError(f"Unsupported method: {self.method}")
                return model
            except Exception as e:
                log.error(f"[ERROR] Failed to fit model for {symbol}: {e}")
                return None

        symbols = self.data['sym_root'].unique()

        fitted_models = {}
        for symbol in symbols:
            model = train_model(symbol)
            if model is not None:
                fitted_models[symbol] = model

        self.fitted_models = fitted_models




    def sample(self, symbol: str, n_samples: int = 1000):
        if self.method == "ARMAGARCH":
            return self._sample_arma_garch(symbol, n_samples)
        elif self.method == "LASSO":
            return self._sample_lasso(symbol, n_samples)
        else:
            raise NotImplementedError(f"Sampling not implemented for {self.method}")