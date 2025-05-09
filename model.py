import datetime
from abc import ABC, abstractmethod
import pandas as pd
from typing import Union

class CustomModel(ABC):
    """
    Abstract base class for forecasting models.
    """
    def __init__(self, data_input: Union[pd.DataFrame, str], split_point: Union[float, datetime.datetime] = 0.8, split: bool = True):
        if isinstance(data_input, str):  # Fixed condition
            self.file_path = data_input
            self.data = self._get_data()
        elif isinstance(data_input, pd.DataFrame):
            self.data = data_input
        else:
            raise ValueError("Invalid input type. Must be a string or a pandas DataFrame.")

        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()

        self.split = split
        self.split_point = split_point