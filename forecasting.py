from abc import ABC, abstractmethod
import pandas as pd

class Forecast(ABC):
    """
    Abstract base class for forecasting models.
    """
    def __init__(self, data:pd.DataFrame):
        self.data = data

    @abstractmethod
    def train(self):
        """
        Train the forecasting model.
        
        Args:
            data: Training data
        """
        pass
    
    @abstractmethod
    def test(self):
        """
        Test the forecasting model.
        
        Args:
            data: Test data
        """
        pass
    
    @abstractmethod
    def predict(self):
        """
        Make predictions using the trained model.
        
        Args:
            horizon: Number of time steps to forecast
            
        Returns:
            Forecasted values
        """
        pass
    
    @abstractmethod
    def evaluate(self, true_values, predicted_values):
        """
        Evaluate model performance.
        
        Args:
            true_values: Actual values
            predicted_values: Predicted values
            
        Returns:
            Performance metrics
        """
        pass