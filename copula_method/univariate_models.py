import datetime
import pandas as pd
import logging as log
from typing import Union
from model import CustomModel
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr

class UnivariateModel(CustomModel):
    def __init__(self, data_input: Union[str, pd.DataFrame], split_point: Union[float, datetime.datetime] = 0.8,
                 method: str = "ARMAGARCH"):
        log.info(f"Initializing {method} model")
        super().__init__(data_input, split_point)
        self.method = method
        self.fitted_models = {}
        log.info(f"{method} model initialized")

    def train(self):
        """Train univariate models for all symbols"""
        log.info(f"Training {self.method} models for all symbols")
        
        # Get unique symbols
        symbols = self.data['sym_root'].unique()
        
        # Train model for each symbol
        for symbol in symbols:
            log.debug(f"Fitting model for symbol: {symbol}")
            symbol_data = self.data[self.data['sym_root'] == symbol]
            
            try:
                if self.method == "ARMAGARCH":
                    model = self._fit_arma_garch(symbol_data['ret_crsp'])
                elif self.method == "LASSO":
                    model = self._fit_lasso(symbol_data)
                else:
                    raise ValueError(f"Unsupported method: {self.method}")
                
                self.fitted_models[symbol] = model
                log.debug(f"Successfully fitted model for {symbol}")
            except Exception as e:
                log.error(f"Failed to fit model for {symbol}: {str(e)}")
        
        log.info(f"Completed training for {len(self.fitted_models)} symbols")
        return self.fitted_models

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
        features = ['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open']
        X = symbol_data[features]
        
        # Convert to R format
        r_y = pandas2ri.py2rpy(y)
        r_X = pandas2ri.py2rpy(X)
        
        # Fit model
        r.source('lasso.R')
        fitted_model = r['fit_lasso'](r_y, r_X)
        
        # Return only model object and features
        return {
            'model': fitted_model.rx2('model'),
            'features_used': features
        }

    def test(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass