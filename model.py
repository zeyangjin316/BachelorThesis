import datetime
from abc import ABC, abstractmethod
import pandas as pd

class CustomModel(ABC):
    """
    Abstract base class for forecasting models.
    """
    def __init__(self, data:pd.DataFrame, split_point: float|datetime =0.8, file_path: str = 'data_for_kit.csv'):
        self.file_path = file_path
        self.data = self._get_data()
        self.split_point = split_point

        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()

    def _get_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.file_path)  # Read the CSV file
        data['date'] = pd.to_datetime(data['date'])  # Convert date column to datetime

        missing_values = data.isnull().sum()
        if missing_values.sum() == 0:
            print("\nNo missing values found in the dataset!")
        else:
            print(f"\nTotal number of missing values: {missing_values.sum()}")
            # palceholder for missing value handling

        return data

    @abstractmethod
    def split(self):
        """
        Split the dataset into training and testing sets.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train the forecasting model.
        """
        pass
    
    @abstractmethod
    def test(self):
        """
        Test the forecasting model.
        """
        pass
    
    @abstractmethod
    def predict(self):
        """
        Make predictions using the trained model.

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