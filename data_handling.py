from datetime import datetime
from data_scaling import SmartScaler
from config import BASE_PATH, LTV_PATH, VIX_PATH

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, split_point):
        self.reader = Reader()
        self.scaler = None
        self.split_point = split_point

    def _split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and test sets based on split_point, handling each time series individually.
        """
        logger.info("Splitting data with split_point: %s", self.split_point)

        def split_symbol_data(symbol_df):
            if isinstance(self.split_point, float):
                split_idx = int(len(symbol_df) * self.split_point)
                return symbol_df.iloc[:split_idx], symbol_df.iloc[split_idx:]
            elif isinstance(self.split_point, datetime):
                train_df = symbol_df[symbol_df['date'] <= self.split_point]
                test_df = symbol_df[symbol_df['date'] > self.split_point]
                return train_df, test_df
            else:
                raise ValueError("split_point must be either float or datetime")

        split_dfs = data.groupby('sym_root').apply(lambda group: split_symbol_data(group))
        train_dfs = [train for train, _ in split_dfs]
        test_dfs = [test for _, test in split_dfs]

        train_set = pd.concat(train_dfs).sort_index()
        test_set = pd.concat(test_dfs).sort_index()

        return train_set, test_set

    def get_data(self, standardize: bool = False):
        self.reader.read_data()
        self.reader.merge_all()
        full_data = self.reader.data

        if standardize:
            self.scaler = SmartScaler(full_data)
            full_data = self.scaler.transform()

        train_set, test_set = self._split_data(full_data)
        return {'full_data': full_data, 'train_set': train_set, 'test_set': test_set}


class Reader:

    def __init__(self, base_path: str = BASE_PATH, ltv_path: str = LTV_PATH, vix_path: str = VIX_PATH,):
        self.base_path = base_path
        self.ltv_path = ltv_path
        self.vix_path = vix_path

        self.base_data = None
        self.ltv_data = None
        self.vix_data = None
        self.data = None  # Final merged data

    def _read_from(self, file_path: str) -> pd.DataFrame:
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
        self.base_data = self._read_from(self.base_path)
        self.ltv_data = self._read_from(self.ltv_path)
        self.vix_data = self._read_from(self.vix_path)

    def merge_all(self) -> None:
        """Merge LTV and VIX data into base_data, store result in self.data."""
        logger.info("Merging external data into base_data")
        self._merge(self.ltv_data, column_prefix="ltv")
        self._merge(self.vix_data, column_prefix="vix")
        self.data = self.base_data.copy()