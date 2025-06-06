import pandas as pd
import logging
from datetime import datetime
from typing import Union

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, split_point):
        self.reader = Reader()
        self.split_point = split_point

    def get_data(self, split=True):
        self.reader.read_data()
        self.reader.merge_all()
        if split:
            return self.reader.split_data(self.split_point)
        else:
            return self.reader.data




class Reader:
    def __init__(self, base_path="data_for_kit.csv", ltv_path="LTV_History.csv", vix_path="VIX_History.csv"):
        self.base_path = base_path
        self.ltv_path = ltv_path
        self.vix_path = vix_path

        self.base_data = None
        self.ltv_data = None
        self.vix_data = None
        self.data = None  # Final merged data

    def _get_data(self, file_path: str) -> pd.DataFrame:
        """Helper to read a CSV file safely."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading data from {file_path}: {str(e)}")

    def _merge(self, external_df: pd.DataFrame, column_prefix: str) -> None:
        """Merge external_df into base_data on 'date', prefixing columns."""
        logger.info(f"Merging {column_prefix} data into base_data")

        # Rename columns
        external_df = external_df.rename(columns={
            col: f"{column_prefix}_{col.lower()}" for col in external_df.columns if col.upper() != 'DATE'
        })
        external_df = external_df.rename(columns={'DATE': 'date'})

        # Ensure datetime format
        external_df['date'] = pd.to_datetime(external_df['date'])
        self.base_data['date'] = pd.to_datetime(self.base_data['date'])

        # Merge
        self.base_data = pd.merge(self.base_data, external_df, on='date', how='left')

    def read_data(self) -> None:
        """Read base, LTV, and VIX data from CSV files."""
        logger.info("Reading CSV files")
        self.base_data = self._get_data(self.base_path)
        self.ltv_data = self._get_data(self.ltv_path)
        self.vix_data = self._get_data(self.vix_path)

    def merge_all(self) -> None:
        """Merge LTV and VIX data into base_data, store result in self.data."""
        logger.info("Merging external data into base_data")
        self._merge(self.ltv_data, column_prefix="ltv")
        self._merge(self.vix_data, column_prefix="vix")
        self.data = self.base_data.copy()

    def split_data(self, split_point: Union[float, datetime] = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and test sets based on split_point, handling each time series individually.
        """
        logger.info("Splitting data with split_point: %s", split_point)

        def split_symbol_data(symbol_df):
            if isinstance(split_point, float):
                split_idx = int(len(symbol_df) * split_point)
                return symbol_df.iloc[:split_idx], symbol_df.iloc[split_idx:]
            elif isinstance(split_point, datetime):
                train_df = symbol_df[symbol_df['date'] <= split_point]
                test_df = symbol_df[symbol_df['date'] > split_point]
                return train_df, test_df
            else:
                raise ValueError("split_point must be either float or datetime")

        split_dfs = self.data.groupby('sym_root').apply(lambda group: split_symbol_data(group))
        train_dfs = [train for train, _ in split_dfs]
        test_dfs = [test for _, test in split_dfs]

        train_set = pd.concat(train_dfs).sort_index()
        test_set = pd.concat(test_dfs).sort_index()

        return train_set, test_set