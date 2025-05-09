from typing import Union
import datetime
import pandas as pd
from model import CustomModel
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr

class UnivariateModel(CustomModel):
    def __init__(self, data_input: Union[str, pd.DataFrame], split_point: Union[float, datetime.datetime] = 0.8,
                 method: str = "ARMAGARCH"):
        super().__init__(data_input, split_point)
        self.method = method
        self.fitted_models = {}

    def run(self):
        """
        Execute the complete modeling pipeline and return a summary
        
        Returns:
            dict: A dictionary containing:
                - Training summary
                - Test predictions
                - Evaluation metrics
                - Model parameters
        """
        # Split the data
        self._split()
        
        # Train models
        self.train()
        
        # Get test predictions
        predictions = self.test()
        
        # Calculate metrics and create summary
        summary = {
            'model_type': self.method,
            'data_summary': {
                'train_size': len(self.train_set),
                'test_size': len(self.test_set),
                'n_symbols': len(self.data['sym_root'].unique()),
                'date_range': {
                    'start': self.data['date'].min(),
                    'end': self.data['date'].max()
                }
            },
            'model_results': {}
        }
        
        # Add model-specific summaries
        for symbol in self.fitted_models:
            symbol_preds = predictions[symbol]
            
            if self.method == "ARMAGARCH":
                metrics = self.evaluate(symbol_preds['actual_values'], 
                                     symbol_preds['mean_forecast'])
                
                summary['model_results'][symbol] = {
                    'evaluation_metrics': metrics,
                    'model_parameters': {
                        'arma_coefficients': self.fitted_models[symbol]['arma_coefficients'],
                        'garch_parameters': self.fitted_models[symbol]['garch_parameters']
                    },
                    'predictions': {
                        'mean_forecast': symbol_preds['mean_forecast'].tolist(),
                        'volatility_forecast': symbol_preds['volatility_forecast'].tolist()
                    }
                }
                
            elif self.method == "LASSO":
                metrics = self.evaluate(symbol_preds['actual_values'], 
                                     symbol_preds['predictions'])
                
                summary['model_results'][symbol] = {
                    'evaluation_metrics': metrics,
                    'model_parameters': {
                        'coefficients': self.fitted_models[symbol]['coefficients'].tolist(),
                        'lambda_min': self.fitted_models[symbol]['lambda_min'],
                        'features_used': self.fitted_models[symbol]['features_used']
                    },
                    'predictions': symbol_preds['predictions'].tolist()
                }
        
        # Add aggregate metrics across all symbols
        all_metrics = pd.DataFrame([res['evaluation_metrics'] 
                                  for res in summary['model_results'].values()])
        
        summary['aggregate_metrics'] = {
            'mean_rmse': all_metrics['rmse'].mean(),
            'mean_r2': all_metrics['r2'].mean(),
            'best_performing_symbol': max(summary['model_results'].items(), 
                                        key=lambda x: x[1]['evaluation_metrics']['r2'])[0],
            'worst_performing_symbol': min(summary['model_results'].items(), 
                                         key=lambda x: x[1]['evaluation_metrics']['r2'])[0]
        }
        
        return summary

    def train(self):
        symbols = self.data['sym_root'].unique()
        
        for symbol in symbols:
            symbol_data = self.data[self.data['sym_root'] == symbol]
            
            if self.method == "ARMAGARCH":
                fitted_model = self._fit_arma_garch(symbol_data['ret_crsp'])
            elif self.method == "LASSO":
                fitted_model = self._fit_lasso(symbol_data)
            else:
                raise ValueError(f"Unsupported method: {self.method}")
        
            self.fitted_models[symbol] = fitted_model

    def test(self):
        """
        Test the fitted models on test data
        """
        if not self._split:
            raise ValueError("Data must be split before testing")
        
        predictions = {}
        for symbol in self.fitted_models:
            symbol_test_data = self.test_set[self.test_set['sym_root'] == symbol]
            predictions[symbol] = self.predict(symbol, symbol_test_data)
        
        return predictions

    def predict(self, symbol, new_data=None):
        """
        Make predictions using the fitted model for a given symbol
        
        Args:
            symbol (str): Symbol to make predictions for
            new_data (pd.DataFrame): New data to make predictions on. If None, uses test set
        
        Returns:
            dict: Dictionary containing predictions and other relevant outputs
        """
        if symbol not in self.fitted_models:
            raise ValueError(f"No fitted model found for symbol {symbol}")
        
        if new_data is None:
            new_data = self.test_set[self.test_set['sym_root'] == symbol]
        
        if self.method == "ARMAGARCH":
            predictions = self._predict_arma_garch(symbol, new_data['ret_crsp'])
        elif self.method == "LASSO":
            predictions = self._predict_lasso(symbol, new_data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        return predictions

    def evaluate(self, true_values, predicted_values):
        """
        Evaluate model performance using various metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def _fit_arma_garch(self, time_series):
        """
        Fits ARMA-GARCH model to a time series using the R script.
        
        Args:
            time_series (pd.Series): Time series data for a single symbol
            
        Returns:
            dict: Dictionary containing model components and parameters
        """
        # Initialize R interface
        pandas2ri.activate()
        forecast = importr('forecast')
        rugarch = importr('rugarch')
        
        # Convert to R format and fit model
        r_timeseries = pandas2ri.py2rpy(time_series)
        r.source('arma_garch.R')
        fitted_model = r['fit_arma_garch'](r_timeseries)
        
        # Extract components
        arma_model = fitted_model.rx2('arma_model')
        garch_model = fitted_model.rx2('garch_model')
        
        # Get model parameters and convert them safely
        try:
            arma_coef = r.coef(arma_model)
            arma_coef_py = [float(x) for x in arma_coef]
        except:
            arma_coef_py = []
        
        try:
            garch_params = r.coef(garch_model)
            garch_params_py = [float(x) for x in garch_params]
        except:
            garch_params_py = []
        
        try:
            arma_fitted = r.fitted(arma_model)
            arma_fitted_py = list(pandas2ri.rpy2py(arma_fitted))
        except:
            arma_fitted_py = []
        
        try:
            garch_fitted = r.fitted(garch_model)
            garch_fitted_py = list(pandas2ri.rpy2py(garch_fitted))
        except:
            garch_fitted_py = []
        
        # Return organized results
        return {
            'arma_model': arma_model,
            'garch_model': garch_model,
            'arma_coefficients': arma_coef_py,
            'garch_parameters': garch_params_py,
            'fitted_values': {
                'arma': arma_fitted_py,
                'garch': garch_fitted_py
            }
        }

    def _fit_lasso(self, symbol_data):
        """
        Fits LASSO model to a symbol's data using the R script.
        
        Args:
            symbol_data (pd.DataFrame): Data for a single symbol with features
        
        Returns:
            dict: Dictionary containing model components and parameters
        """
        # Initialize R interface
        from rpy2.robjects import pandas2ri, r
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        
        # Import glmnet package
        glmnet = importr('glmnet')
        
        # Prepare the target variable and features
        y = symbol_data['ret_crsp']
        
        # Select features for LASSO (adjust these based on your needs)
        features = ['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open']
        X = symbol_data[features]
        
        # Convert to R format
        r_y = pandas2ri.py2rpy(y)
        r_X = pandas2ri.py2rpy(X)
        
        # Source and run the R script
        r.source('lasso.R')
        fitted_model = r['fit_lasso'](r_y, r_X)
        
        # Extract components
        model = fitted_model.rx2('model')
        cv_fit = fitted_model.rx2('cv_fit')
        coefficients = fitted_model.rx2('coefficients')
        fitted_values = fitted_model.rx2('fitted_values')
        lambda_min = fitted_model.rx2('lambda_min')
        
        # Return organized results
        return {
            'model': model,
            'cv_fit': cv_fit,
            'coefficients': pandas2ri.rpy2py(coefficients),
            'fitted_values': pandas2ri.rpy2py(fitted_values),
            'lambda_min': pandas2ri.rpy2py(lambda_min)[0],
            'features_used': features
        }

    def _predict_arma_garch(self, symbol, new_data):
        """
        Make predictions using fitted ARMA-GARCH model
        """
        from rpy2.robjects import pandas2ri, r
        from rpy2.robjects.vectors import FloatVector
        pandas2ri.activate()
        
        fitted_model = self.fitted_models[symbol]
        arma_model = fitted_model['arma_model']
        garch_model = fitted_model['garch_model']
        
        # Convert data to R format
        r_new_data = pandas2ri.py2rpy(new_data)
        
        # Make predictions
        arma_pred = r.forecast(arma_model, h=len(new_data))
        # Use FloatVector to convert the mean predictions
        arma_mean = pandas2ri.rpy2py(FloatVector(arma_pred.rx2('mean')))
        
        # Get GARCH predictions (volatility forecasts)
        garch_pred = r.ugarchforecast(garch_model, n_ahead=len(new_data))
        garch_sigma = pandas2ri.rpy2py(r.sigma(garch_pred))
        
        return {
            'mean_forecast': arma_mean,
            'volatility_forecast': garch_sigma,
            'actual_values': new_data.values
        }

    def _predict_lasso(self, symbol, new_data):
        """
        Make predictions using fitted LASSO model
        """
        from rpy2.robjects import pandas2ri, r
        pandas2ri.activate()
        
        fitted_model = self.fitted_models[symbol]
        model = fitted_model['model']
        features = fitted_model['features_used']
        
        # Prepare features
        X_new = new_data[features]
        
        # Convert to R matrix
        r_X_new = pandas2ri.py2rpy(X_new)
        r.assign("X_new", r_X_new)
        
        # Make predictions
        predictions = r.predict(model, r_X_new)
        predictions = pandas2ri.rpy2py(predictions)
        
        return {
            'predictions': predictions,
            'actual_values': new_data['ret_crsp'].values,
            'features_used': features
        }