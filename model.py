import datetime
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Union

logger = logging.getLogger(__name__)

class CustomModel(ABC):
    """
    Abstract base class for forecasting models.
    """
    def __init__(self, data_input: Union[pd.DataFrame, str], split_point: Union[float, datetime.datetime] = 0.8, split: bool = True):
        logger.info("Trying to fetch data")
        if isinstance(data_input, str):
            logger.debug("File location given: %s", data_input)# Fixed condition
            self.file_path = data_input
            self.data = self._get_data()
        elif isinstance(data_input, pd.DataFrame):
            logger.debug("Data frame given")
            self.data = data_input
        else:
            raise ValueError("Invalid input type. Must be a string or a pandas DataFrame.")
        logger.info("Data fetched successfully")
        #logger.debug("Data shape: %s", self.data.info())

        logger.info("Initializing train and test sets")
        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        self.split = split
        self.split_point = split_point

    def _get_data(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            raise ValueError(f"Error reading data from {self.file_path}: {str(e)}")

    def _split(self):
        """
        Split the data into training and test sets based on split_point
        """
        if isinstance(self.split_point, float):
            logger.info("Splitting data with training set percentage: %s", self.split_point)
            split_idx = int(len(self.data) * self.split_point)
            self.train_set = self.data.iloc[:split_idx]
            self.test_set = self.data.iloc[split_idx:]
        elif isinstance(self.split_point, datetime.datetime):
            logger.info("Splitting data with training set date: %s", {self.split_point})
            self.train_set = self.data[self.data['date'] <= self.split_point]
            self.test_set = self.data[self.data['date'] > self.split_point]
        else:
            raise ValueError("split_point must be either float or datetime")

    @abstractmethod
    def train(self):
        """Train the model"""
        pass

    @abstractmethod
    def test(self):
        """Test the model"""
        pass

    @abstractmethod
    def predict(self):
        """Make predictions"""
        pass

    @abstractmethod
    def evaluate(self, true_values, predicted_values):
        """Evaluate model performance"""
        pass