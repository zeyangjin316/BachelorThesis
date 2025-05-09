import datetime
from abc import ABC, abstractmethod
import pandas as pd

class CustomModel(ABC):
    """
    Abstract base class for forecasting models.
    """
    def __init__(self, file_path: str, data: pd.DataFrame = pd.DataFrame(), split_point: float|datetime = 0.8):
        self.file_path = file_path
        if data.empty is False:
            self.data = data
        else:
            self.data = self._get_data()

        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()

        self.split_point = split_point
        self._split()

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

    def _split(self):
        """
        Split the dataset into training and testing sets.
        """
        if isinstance(self.split_point, (float, int)):
            if not 0 < self.split_point < 1:
                raise ValueError("Percentage split must be between 0 and 1")
            split_idx = int(len(self.data) * self.split_point)
            train_split = self.data.iloc[:split_idx]
            test_split = self.data.iloc[split_idx:]
        else:
            # Try to convert to datetime if it's not already
            split_date = pd.to_datetime(self.split_point)
            train_split = self.data[self.data['date'] <= split_date]
            test_split = self.data[self.data['date'] > split_date]

        if len(train_split) == 0 or len(test_split) == 0:
            raise ValueError("Split resulted in empty training or testing set")

        self.train_set = train_split
        self.test_set = test_split
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