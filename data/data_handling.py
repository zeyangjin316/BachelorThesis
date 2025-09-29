from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Tuple
import os
import logging
import pandas as pd

from config import BASE_PATH, LTV_PATH, VIX_PATH, INTRADAY_PATH
from data.data_reader import Reader

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Orchestrates loading the merged dataset, computing realized measures via `Reader`,
    and splitting into train/test sets. Intraday RV/RS measures are computed in `Reader`.
    """

    def __init__(self, split_point: float | datetime):
        """
        Parameters
        ----------
        split_point : float | datetime
            Train/test split definition. If float, uses per-symbol fraction in (0,1).
            If datetime, uses a calendar date threshold.
        """
        if not (isinstance(split_point, float) or isinstance(split_point, datetime)):
            raise ValueError("split_point must be float in (0,1) or datetime")
        if isinstance(split_point, float) and not (0.0 < split_point < 1.0):
            raise ValueError("float split_point must lie in (0,1)")

        self.split_point = split_point
        self.reader = Reader(BASE_PATH, LTV_PATH, VIX_PATH, INTRADAY_PATH)

    def _split_by_symbol(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a balanced panel into train/test per symbol by fraction or date.

        Returns
        -------
        (train, test) : tuple[pd.DataFrame, pd.DataFrame]
            DataFrames containing disjoint train and test rows.
        """
        df = df.sort_values(["sym_root", "date"])

        if isinstance(self.split_point, float):
            group_sizes = df.groupby("sym_root")["date"].transform("size")
            frac_idx = df.groupby("sym_root").cumcount() / group_sizes
            train = df[frac_idx < self.split_point]
            test  = df[frac_idx >= self.split_point]
        else:
            train = df[df["date"] <= self.split_point]
            test  = df[df["date"] > self.split_point]

        if not train.empty:
            logger.info(
                "Train set: %s -> %s (%d rows)",
                train["date"].min().date(),
                train["date"].max().date(),
                len(train),
            )
        if not test.empty:
            logger.info(
                "Test set: %s -> %s (%d rows)",
                test["date"].min().date(),
                test["date"].max().date(),
                len(test),
            )
        return train, test

    def _feature_filter(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        exclude_cols: Optional[List[str]]
    ) -> dict:
        """
        Remove exact-duplicate numeric columns using only the training set.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training subset used to identify collinear duplicates.
        target_col : str
            Name of the target column to keep.
        exclude_cols : list[str] | None
            Identifier columns to always keep (e.g., date, sym_root, permno).

        Returns
        -------
        dict
            {'kept': [...], 'dropped_duplicates': [...]}
        """
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

    def _validate_balanced_panel(self, df: pd.DataFrame) -> None:
        """
        Validate that each symbol has identical row counts and the same date index.

        Raises
        ------
        ValueError
            If counts differ or date indices are misaligned across symbols.
        """
        counts = df.groupby("sym_root").size().sort_values()
        if counts.nunique() != 1:
            smallest, largest = counts.iloc[0], counts.iloc[-1]
            head_tail = pd.concat([counts.head(5), counts.tail(5)]).drop_duplicates()
            raise ValueError(
                "[BalancedPanel] Imbalanced counts.\n"
                f"min={smallest}, max={largest}\n"
                f"sample counts:\n{head_tail.to_string()}"
            )

        first_sym = counts.index[0]
        ref_dates = (
            df.loc[df["sym_root"] == first_sym, "date"]
            .sort_values()
            .astype("datetime64[ns]")
            .tolist()
        )
        ref_set = set(ref_dates)

        bad_syms, details = [], []
        for sym, g in df.groupby("sym_root"):
            dates = g["date"].sort_values().astype("datetime64[ns]").tolist()
            if dates != ref_dates:
                bad_syms.append(sym)
                sset = set(dates)
                miss = sorted(ref_set - sset)[:5]
                extra = sorted(sset - ref_set)[:5]
                details.append(
                    f"{sym}: missing {len(ref_set - sset)} (e.g. {miss}), "
                    f"extra {len(sset - ref_set)} (e.g. {extra})"
                )
        if bad_syms:
            sample = "\n".join(details[:10])
            raise ValueError(
                "[BalancedPanel] Date misalignment across symbols.\n"
                f"{sample}"
            )

        logger.info(
            "Balanced panel confirmed: %d symbols x %d dates",
            df["sym_root"].nunique(),
            len(ref_dates),
        )

    def get_data(
        self,
        exclude_pandemic: bool = False,
        target_only: bool = False,
        filter_duplicates: bool = False,
        target_col: str = "ret_crsp",
        exclude_cols: Optional[List[str]] = None,
        save_df: bool = False,
        df_path: Optional[str] = None,
        df_format: str = "csv",
        save_png: bool = False,
        png_path: Optional[str] = None,
        png_head_n: int = 100,
        png_dpi: int = 200,
    ) -> dict:
        """
        Load the merged dataset, compute realized measures, enforce a balanced panel,
        split into train/test, and optionally persist a CSV/Parquet and a PNG preview.

        Parameters
        ----------
        exclude_pandemic : bool, default=False
            If True, drop all dates >= 2020-01-01.
        target_only : bool, default=False
            If True, restrict to target and minimal identifiers.
        filter_duplicates : bool, default=False
            If True, drop exact-duplicate numeric features (train-only decision).
        target_col : str, default="ret_crsp"
            Name of the target column.
        exclude_cols : list[str] | None, default=None
            Identifier columns to keep when filtering features.
        save_df : bool, default=False
            If True, save the final merged dataset to disk.
        df_path : str | None, default=None
            Output path; if None, a timestamped path under `results/` is used.
        df_format : str, default="csv"
            "csv" or "parquet".
        save_png : bool, default=False
            If True, save a small PNG preview of the DataFrame head.
        png_path : str | None, default=None
            Output path for the PNG; if None, a timestamped path is used.
        png_head_n : int, default=100
            Number of rows to show in the PNG preview.
        png_dpi : int, default=200
            DPI for the PNG preview.

        Returns
        -------
        dict
            {
              "full_data": pd.DataFrame,
              "train_set": pd.DataFrame,
              "test_set": pd.DataFrame,
              "feature_filter_report": dict | None,
              "df_save_path": str | None,
              "png_save_path": str | None,
            }
        """
        # 1) load + merge, build realized measures in Reader
        self.reader.read_all(build_weekly_monthly=True, target_only=target_only)
        df = self.reader.data.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 2) optional cut
        if exclude_pandemic:
            df = df[df["date"] < "2020-01-01"]

        # 3) drop entire days containing any NaNs to keep a balanced panel
        id_cols = ["date", "sym_root", "permno"]
        non_id_cols = [c for c in df.columns if c not in id_cols]
        if non_id_cols:
            nan_any_row = df[non_id_cols].isna().any(axis=1)
            if nan_any_row.any():
                bad_dates = pd.to_datetime(df.loc[nan_any_row, "date"]).unique()
                n_bad = len(bad_dates)
                df = df[~df["date"].isin(bad_dates)].copy()
                # small preview for logs
                try:
                    preview = ", ".join(str(d)[:10] for d in sorted(pd.to_datetime(bad_dates))[:5])
                except Exception:
                    preview = "n/a"
                logger.info(
                    "Dropped %d dates across all symbols due to NaNs in any column. Sample: %s",
                    n_bad, preview
                )

        # 4) validate balanced panel (after NaN-day drop)
        self._validate_balanced_panel(df)

        # 5) split
        train_set, test_set = self._split_by_symbol(df)

        # 6) optional duplicate-feature filter (train-only decision)
        feature_filter_report = None
        if filter_duplicates:
            feature_filter_report = self._feature_filter(train_set, target_col, exclude_cols)
            keep_feats = feature_filter_report["kept"]
            id_keep = exclude_cols or ["date", "sym_root", "permno"]
            col_order = [c for c in id_keep + [target_col] + keep_feats if c in df.columns]
            if col_order:
                df = df[col_order]
                train_set = train_set[[c for c in col_order if c in train_set.columns]]
                test_set  = test_set[[c for c in col_order if c in test_set.columns]]

        # 7) save dataset (optional)
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

        # 8) save PNG preview (optional)
        png_save_path = None
        if save_png:
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
