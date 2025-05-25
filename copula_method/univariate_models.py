import datetime
import pandas as pd
import logging as log
from typing import Union
from rpy2.robjects import pandas2ri, r
from reader import Reader

class UnivariateModel():
    def __init__(self, data: pd.DataFrame, split_point: Union[float, datetime.datetime] = 0.8,
                 method: str = "ARMAGARCH"):
        log.info(f"Initializing {method} model")
        self.data = data
        self.split_point = split_point
        self.method = method
        self.fitted_models = {}
        log.info(f"{method} model initialized")

    def _fit_arma_garch(self, time_series):
        """Fit ARMA-GARCH model to time series"""
        pandas2ri.activate()

        # Convert to R format
        r_timeseries = pandas2ri.py2rpy(time_series)

        # Fit model
        r.source('arma_garch.R')
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

    def fit(self):
        """Train univariate models for all symbols"""
        log.info(f"Training {self.method} models for all symbols")

        def train_model(symbol):
            log.debug(f"Fitting model for symbol: {symbol}")
            symbol_data = self.data[self.data['sym_root'] == symbol]

            try:
                if self.method == "ARMAGARCH":
                    model = self._fit_arma_garch(symbol_data['ret_crsp'])
                elif self.method == "LASSO":
                    model = self._fit_lasso(symbol_data)
                else:
                    raise ValueError(f"Unsupported method: {self.method}")

                log.debug(f"Successfully fitted model for {symbol}")
                return model
            except Exception as e:
                log.error(f"Failed to fit model for {symbol}: {str(e)}")
                return None

        # Train models using dictionary comprehension
        symbols = self.data['sym_root'].unique()
        self.fitted_models = {
            symbol: model for symbol in symbols if (model := train_model(symbol)) is not None
        }

        log.info(f"Completed training for {len(self.fitted_models)} symbols")
        return self.fitted_models

    def sample_from_model(self, symbol: str, n_samples: int = 1000):
        from rpy2.robjects import r
        r.source("arma_garch.R")
        r_sample = r["forecast_arma_garch_samples"]

        model = self.fitted_models.get(symbol)
        if not model:
            raise ValueError(f"No fitted model for symbol {symbol}")

        arma_model = model["arma_model"]
        garch_model = model["garch_model"]
        samples = r_sample(arma_model, garch_model, n_samples)

        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        return pandas2ri.rpy2py(samples)