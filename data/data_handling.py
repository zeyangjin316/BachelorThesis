from datetime import datetime
from config import BASE_PATH, LTV_PATH, VIX_PATH

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, split_point):
        self.reader = Reader()
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

    def _cut_pandemic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Private function that removes data from 2020 onwards (pandemic period) from the dataframe.

        Args:
            df: DataFrame with a 'date' column

        Returns:
            DataFrame filtered to exclude data from 2020 onwards
        """
        logger.info("Filtering out pandemic data (2020 onwards)")

        # Ensure date column is datetime
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Filter to keep only data before 2020
        pre_pandemic_df = df[df['date'] < '2020-01-01']

        logger.info(f"Original data shape: {df.shape}, Pre-pandemic data shape: {pre_pandemic_df.shape}")

        return pre_pandemic_df

    def _feature_filter(
            self,
            train_df: pd.DataFrame,
            target_col: str = "ret_crsp",
            exclude_cols: list[str] = None,
    ):
        """
        Remove duplicate features using TRAIN ONLY to avoid leakage. Returns a report dict.
        """
        if exclude_cols is None:
            exclude_cols = ["date", "sym_root", "permno"]
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        cand = [c for c in numeric_cols if c not in set(exclude_cols + [target_col])]

        X = train_df[cand].copy()
        dropped_duplicates = []

        # Remove duplicate columns (exact duplicate columns)
        if len(X.columns) > 1:
            keep_T = X.T.drop_duplicates(keep="first")
            keep_cols = keep_T.T.columns.tolist()
            dup_cols = [c for c in X.columns if c not in keep_cols]
            if dup_cols:
                dropped_duplicates += dup_cols
                X = X[keep_cols]

        logger.info("Dropped duplicate columns: %s", dropped_duplicates)

        kept = list(X.columns)
        return {
            "kept": kept,
            "dropped_duplicates": dropped_duplicates,
        }

    def get_data(
            self,
            standardize: bool = False,
            save_df: bool = False,
            df_path: str | None = None,
            df_format: str = "csv",  # "csv" or "parquet"
            save_png: bool = False,
            png_path: str | None = None,
            png_head_n: int = 100,
            png_dpi: int = 200,

            # ---- feature filter options (new) ----
            filter_features: bool = False,
            target_col: str = "ret_crsp",
            exclude_cols: list[str] | None = None,  # e.g. ["date","sym_root","permno"]
            min_unique_frac: float = 0.001,
            corr_threshold: float = 0.95,
            corr_method: str = "spearman",
            vif_threshold: float | None = None,  # set to 10.0 to enable
            
            # ---- pandemic data filter option ----
            exclude_pandemic: bool = False,  # removes data from 2020 onwards
    ):
        """
        Load/merge data (optionally standardize), split into train/test, and optionally:
          - save_df: write the full merged dataframe (full_data) to CSV/Parquet
          - save_png: save a PNG picture of ONLY head(png_head_n) of full_data
          - filter_features: remove near-constant/duplicate/highly-correlated/(optional VIF) features
                             The decision is made on TRAIN ONLY (leak-safe) and then applied to all.
          - exclude_pandemic: remove data from 2020 onwards to focus on pre-pandemic patterns
        """
        # ----- load & merge -----
        self.reader.read_data()
        self.reader.merge_all()
        full_data = self.reader.data

        # ----- optional pandemic data filtering (before standardization/splitting) -----
        if exclude_pandemic:
            full_data = self._cut_pandemic_data(full_data)

        # ----- split FIRST (so filter is decided on training only) -----
        train_set, test_set = self._split_data(full_data)

        # ----- feature filter (optional) -----
        feature_filter_report = None
        if filter_features:
            feature_filter_report = self._feature_filter(
                train_df=train_set,
                target_col=target_col,
                exclude_cols=exclude_cols,
            )
            keep_feats = feature_filter_report["kept"]
            id_cols = exclude_cols or ["date", "sym_root", "permno"]

            # build final column order: ids + target + kept features
            final_cols = []
            for c in id_cols + [target_col] + keep_feats:
                if c in full_data.columns and c not in final_cols:
                    final_cols.append(c)

            # apply to all splits
            full_data = full_data.loc[:, [c for c in final_cols if c in full_data.columns]]
            train_set = train_set.loc[:, [c for c in final_cols if c in train_set.columns]]
            test_set = test_set.loc[:, [c for c in final_cols if c in test_set.columns]]

            logging.getLogger(__name__).info(
                "Feature filter kept %d features; dropped duplicates: %d",
                len(keep_feats),
                len(feature_filter_report["dropped_duplicates"]),
            )

        # ===== optional: save dataframe file (FULL, after filtering if any) =====
        df_save_path = None
        if save_df:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            if df_path is None:
                os.makedirs("results", exist_ok=True)
                ext = "csv" if df_format.lower() == "csv" else "parquet"
                df_path = os.path.join("results", f"final_data_{stamp}.{ext}")

            os.makedirs(os.path.dirname(df_path) or ".", exist_ok=True)
            fmt = df_format.lower()
            if fmt == "csv":
                full_data.to_csv(df_path, index=False)
            elif fmt == "parquet":
                try:
                    full_data.to_parquet(df_path, index=False)
                except (ImportError, ModuleNotFoundError):
                    stem, _ = os.path.splitext(df_path)
                    df_path = f"{stem}.csv"
                    full_data.to_csv(df_path, index=False)
            else:
                raise ValueError("df_format must be 'csv' or 'parquet'")
            df_save_path = df_path
            # NEW: print info about saved dataframe
            print(
                f"[DataHandler] Saved dataframe to {df_save_path}\n"
                f"  full_data shape: {full_data.shape} (rows x cols)\n"
                f"  train_set shape: {train_set.shape}\n"
                f"  test_set  shape: {test_set.shape}"
            )

        # ===== optional: save PNG picture (HEAD ONLY, after filtering if any) =====
        from data.data_visualizer import _save_dataframe_png
        png_save_path = None
        if save_png:
            png_save_path = _save_dataframe_png(
                df=full_data,
                png_path=png_path,
                head_n=png_head_n,
                dpi=png_dpi,
            )

        return {
            'full_data': full_data,
            'train_set': train_set,
            'test_set': test_set,
            'df_save_path': df_save_path,
            'png_save_path': png_save_path,
            'feature_filter_report': feature_filter_report,
        }


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