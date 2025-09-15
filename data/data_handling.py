from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Tuple
import os
import logging
import pandas as pd

from config import BASE_PATH, LTV_PATH, VIX_PATH, INTRADAY_PATH
from data.reader import Reader
from data.features import add_daily_variance_features, ANNUALIZE as FE_ANNUALIZE

logger = logging.getLogger(__name__)

class DataHandler:
    """
    Load merged dataset, add daily variance features, and split into train/test.
    Intraday RV is pre-computed by Reader and already present as RV5m_d/Vol5m_d.
    """
    def __init__(self, split_point: float | datetime):
        if not (isinstance(split_point, float) or isinstance(split_point, datetime)):
            raise ValueError("split_point must be float in (0,1) or datetime")
        if isinstance(split_point, float) and not (0.0 < split_point < 1.0):
            raise ValueError("float split_point must lie in (0,1)")

        self.split_point = split_point
        self.reader = Reader(BASE_PATH, LTV_PATH, VIX_PATH, INTRADAY_PATH)

    def _split_by_symbol(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split per symbol by percentage or datetime."""
        df = df.sort_values(["sym_root", "date"])

        def _split_group(g: pd.DataFrame):
            if isinstance(self.split_point, float):
                idx = int(len(g) * self.split_point)
                return g.iloc[:idx], g.iloc[idx:]
            else:
                return g[g["date"] <= self.split_point], g[g["date"] > self.split_point]

        parts = df.groupby("sym_root").apply(_split_group)
        train = pd.concat([a for a, _ in parts], axis=0).sort_values(["sym_root", "date"])
        test  = pd.concat([b for _, b in parts], axis=0).sort_values(["sym_root", "date"])
        return train, test

    def _feature_filter(self, train_df: pd.DataFrame, target_col: str, exclude_cols: Optional[List[str]]) -> dict:
        """Remove exact-duplicate numeric columns using the training set only."""
        if exclude_cols is None:
            exclude_cols = ["date", "sym_root", "permno"]
        num_cols = train_df.select_dtypes(include="number").columns.tolist()
        cand = [c for c in num_cols if c not in set(exclude_cols + [target_col])]
        X = train_df[cand].copy()
        dropped = []
        if len(X.columns) > 1:
            keep_T = X.T.drop_duplicates(keep="first")
            keep_cols = keep_T.T.columns.tolist()
            dropped = [c for c in X.columns if c not in keep_cols]
        kept = [c for c in X.columns if c not in dropped]
        return {"kept": kept, "dropped_duplicates": dropped}

    def get_data(
        self,
        # feature params
        return_col: str = "ret_crsp",
        har_windows: Tuple[int, ...] = (5, 21, 63),
        hl_days: Tuple[int, ...] = (1, 5, 21, 63),

        # data hygiene
        exclude_pandemic: bool = False,

        # optional feature pruning
        filter_features: bool = False,
        target_col: str = "ret_crsp",
        exclude_cols: Optional[List[str]] = None,

        # saving
        save_df: bool = False,
        df_path: Optional[str] = None,
        df_format: str = "csv",

        # PNG preview (restored)
        save_png: bool = False,
        png_path: Optional[str] = None,
        png_head_n: int = 100,
        png_dpi: int = 200,
    ) -> dict:
        """
        Load, engineer features, split, and optionally save CSV/Parquet and a PNG preview of head(full_data).
        """
        # Load and merge sources
        self.reader.read_all()
        df = self.reader.data.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Optional cut
        if exclude_pandemic:
            df = df[df["date"] < "2020-01-01"]

        # Daily variance/volatility features
        df = add_daily_variance_features(
            df=df,
            return_col=return_col,
            har_windows=har_windows,
            hl_days=hl_days,
            annualize=FE_ANNUALIZE,
        )

        # --- remove artificial zeros ---
        var_cols = [c for c in df.columns if any(tag in c.lower() for tag in ["rv", "vol", "semivar", "downvol"])]
        df = df.loc[~(df[var_cols] == 0).any(axis=1)]

        # Split
        train_set, test_set = self._split_by_symbol(df)

        # Optional duplicate feature filter
        feature_filter_report = None
        if filter_features:
            feature_filter_report = self._feature_filter(train_set, target_col, exclude_cols)
            keep_feats = feature_filter_report["kept"]
            id_cols = exclude_cols or ["date", "sym_root", "permno"]
            col_order = [c for c in id_cols + [target_col] + keep_feats if c in df.columns]
            df = df[col_order]
            train_set = train_set[[c for c in col_order if c in train_set.columns]]
            test_set  = test_set[[c for c in col_order if c in test_set.columns]]

        # Save CSV/Parquet
        df_save_path = None
        if save_df:
            stamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
            if df_path is None:
                os.makedirs("results", exist_ok=True)
                ext = "csv" if df_format.lower() == "csv" else "parquet"
                df_path = os.path.join("results", f"final_data_{stamp}.{ext}")
            os.makedirs(os.path.dirname(df_path) or ".", exist_ok=True)
            if df_format.lower() == "csv":
                df.to_csv(df_path, index=False)
            else:
                try:
                    df.to_parquet(df_path, index=False)
                except Exception:
                    stem, _ = os.path.splitext(df_path)
                    df_path = f"{stem}.csv"
                    df.to_csv(df_path, index=False)
            df_save_path = df_path

        # Save PNG preview
        png_save_path = None
        if save_png:
            # Local import to avoid hard dependency if visualizer is absent
            from data.data_visualizer import _save_dataframe_png
            png_save_path = _save_dataframe_png(
                df=df,
                png_path=png_path,
                head_n=png_head_n,
                dpi=png_dpi,
            )

        return {
            "full_data": df,
            "train_set": train_set,
            "test_set": test_set,
            "feature_filter_report": feature_filter_report,
            "df_save_path": df_save_path,
            "png_save_path": png_save_path,
        }
