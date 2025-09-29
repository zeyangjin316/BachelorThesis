from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANNUALIZE = 252


@dataclass
class ReaderPaths:
    base_path: str
    ltv_path: Optional[str] = None
    vix_path: Optional[str] = None
    intraday_path: Optional[str] = None


class Reader:
    """
    Reads and merges:
      - Base panel (data_for_kit.csv)
      - LTV history (ltv_open, ltv_high, ltv_low, ltv_close, ...)
      - VIX history (vix_open, vix_high, vix_low, vix_close, ...)
      - Intraday 5m returns (returns_5m.csv) → 15m aggregation → realized measures
    Keys: (date, sym_root, permno)
    """

    def __init__(self, base_path: str, ltv_path: str, vix_path: str, intraday_path: str):
        self.paths = ReaderPaths(base_path, ltv_path, vix_path, intraday_path)
        self.data: Optional[pd.DataFrame] = None

    # ----------------------
    # change the signature to add target_only (default False to keep old behavior)
    def read_all(self, build_weekly_monthly: bool = True, target_only: bool = False) -> None:
        def _check_keys(df: pd.DataFrame, step: str):
            missing = [k for k in ["date", "sym_root", "permno"] if k not in df.columns]
            if missing:
                logger.error(f"[Reader] Keys missing after {step}: {missing}")
            else:
                logger.info(f"[Reader] Keys OK after {step}")

        # --- Base ---
        logger.info("Reading base data...")
        base = pd.read_csv(self.paths.base_path)
        base = self._standardize_keys(base)
        base["date"] = pd.to_datetime(base["date"])
        base = base.sort_values(["sym_root", "date"]).reset_index(drop=True)
        _check_keys(base, "loading base")

        # >>>>>> NEW: target-only fast path >>>>>>
        if target_only:
            if "ret_crsp" not in base.columns:
                raise ValueError("target_only=True requires 'ret_crsp' in the base data.")
            df = base[["date", "sym_root", "permno", "ret_crsp"]].copy()
            # keep original behavior otherwise (no merges, no weekly/monthly)
            self.data = df.sort_values(["sym_root", "date"]).reset_index(drop=True)
            logger.info("Reader: target_only=True -> skipped LTV/VIX/Intraday merges; kept ret_crsp only.")
            return
        # <<<<<< end target-only fast path <<<<<<

        # --- LTV ---
        if self.paths.ltv_path:
            logger.info("Reading LTV...")
            ltv = pd.read_csv(self.paths.ltv_path)
            ltv = ltv.rename(columns={c: "date" for c in ltv.columns if c.lower() == "date"})
            ltv["date"] = pd.to_datetime(ltv["date"], errors="coerce")
            ltv = ltv.rename(columns={c: f"ltv_{c.lower()}" for c in ltv.columns if c != "date"})
            base = base.merge(ltv, on="date", how="left")
            _check_keys(base, "merging LTV")

        # --- VIX ---
        if self.paths.vix_path:
            logger.info("Reading VIX...")
            vix = pd.read_csv(self.paths.vix_path)
            vix = vix.rename(columns={c: "date" for c in vix.columns if c.lower() == "date"})
            vix["date"] = pd.to_datetime(vix["date"], errors="coerce")
            vix = vix.rename(columns={c: f"vix_{c.lower()}" for c in vix.columns if c != "date"})
            base = base.merge(vix, on="date", how="left")
            _check_keys(base, "merging VIX")

        # --- Intraday ---
        if self.paths.intraday_path:
            logger.info("Reading intraday returns...")
            intr = pd.read_csv(self.paths.intraday_path)
            intr = self._standardize_keys(intr)

            intr["date"] = pd.to_datetime(intr["date"])
            intr["sym_root"] = intr["sym_root"].astype(base["sym_root"].dtype)
            intr["permno"] = intr["permno"].astype(base["permno"].dtype)

            if len(intr) != len(base):
                raise ValueError(f"Intraday rows ({len(intr)}) != base rows ({len(base)})")

            base = self._attach_realized_measures(base, intr)
            _check_keys(base, "merging intraday")

        # --- Weekly/monthly averages ---
        if build_weekly_monthly:
            base = self._attach_weekly_monthly(base)
            _check_keys(base, "attaching weekly/monthly")

        # --- Final check ---
        missing = [k for k in ["date", "sym_root", "permno"] if k not in base.columns]
        if missing:
            raise ValueError(f"[Reader] Final dataframe is missing keys: {missing}")
        for k in ["date", "sym_root", "permno"]:
            if base[k].isna().any():
                raise ValueError(f"[Reader] Column {k} contains NaNs after merging")

        self.data = base

    # ----------------------
    # Key standardization
    # ----------------------
    def _standardize_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for c in df.columns:
            if c.lower() == "sym_root":
                rename_map[c] = "sym_root"
            if c.lower() == "date":
                rename_map[c] = "date"
            if c.lower() == "permno":
                rename_map[c] = "permno"
        if rename_map:
            df = df.rename(columns=rename_map)

        required = {"date", "sym_root", "permno"}
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' after standardization")

        return df

    # ----------------------
    # Intraday realized measures
    # ----------------------
    def _attach_realized_measures(self, base: pd.DataFrame, intr: pd.DataFrame) -> pd.DataFrame:
        # identify intraday return columns (V1, V2, …)
        ret5_cols = [c for c in intr.columns if re.match(r"^V\d+$", c)]
        ret5_cols = sorted(ret5_cols, key=lambda x: int(re.sub(r"\D", "", x)))
        intr[ret5_cols] = intr[ret5_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # --- align key dtypes with base ---
        intr["date"] = pd.to_datetime(intr["date"])
        intr["sym_root"] = intr["sym_root"].astype(base["sym_root"].dtype)
        intr["permno"] = intr["permno"].astype(base["permno"].dtype)

        # --- aggregate 5m → 15m ---
        ret15_cols, intr = self._aggregate_5m_to_15m(intr, ret5_cols)

        R15 = intr[ret15_cols]
        rv_day = R15.pow(2).sum(axis=1).to_numpy()
        rs_neg_day = R15.clip(upper=0.0).pow(2).sum(axis=1).to_numpy()

        measures = intr[["date", "sym_root", "permno"]].copy()
        measures["RVd_ann"] = rv_day * ANNUALIZE
        measures["RSd_neg_ann"] = rs_neg_day * ANNUALIZE

        # --- merge safely on keys ---
        df = base.merge(measures, on=["date", "sym_root", "permno"], how="left", validate="1:1")
        return df

    @staticmethod
    def _aggregate_5m_to_15m(intr: pd.DataFrame, ret5_cols: List[str]) -> Tuple[List[str], pd.DataFrame]:
        n5 = len(ret5_cols)
        n_groups = n5 // 3
        data = {}
        ret15_cols = []

        for g in range(n_groups):
            c1, c2, c3 = ret5_cols[3*g], ret5_cols[3*g+1], ret5_cols[3*g+2]
            out_col = f"ret_15m_{g+1:02d}"
            data[out_col] = (1.0 + intr[c1]) * (1.0 + intr[c2]) * (1.0 + intr[c3]) - 1.0
            ret15_cols.append(out_col)

        intr = pd.concat([intr, pd.DataFrame(data, index=intr.index)], axis=1)
        return ret15_cols, intr

    # ----------------------
    # Weekly/monthly averages
    # ----------------------
    def _attach_weekly_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["sym_root", "date"]).copy()

        df["RVw_ann"] = (
            df.groupby("sym_root", group_keys=False)["RVd_ann"]
            .transform(lambda x: x.rolling(window=5, min_periods=5).mean())
        )
        df["RVm_ann"] = (
            df.groupby("sym_root", group_keys=False)["RVd_ann"]
            .transform(lambda x: x.rolling(window=21, min_periods=21).mean())
        )
        df["RSw_neg_ann"] = (
            df.groupby("sym_root", group_keys=False)["RSd_neg_ann"]
            .transform(lambda x: x.rolling(window=5, min_periods=5).mean())
        )
        df["RSm_neg_ann"] = (
            df.groupby("sym_root", group_keys=False)["RSd_neg_ann"]
            .transform(lambda x: x.rolling(window=21, min_periods=21).mean())
        )

        return df

    @staticmethod
    def _avg_daily(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["RVw_ann"] = g["RVd_ann"].rolling(window=5, min_periods=5).mean()
        g["RVm_ann"] = g["RVd_ann"].rolling(window=21, min_periods=21).mean()
        g["RSw_neg_ann"] = g["RSd_neg_ann"].rolling(window=5, min_periods=5).mean()
        g["RSm_neg_ann"] = g["RSd_neg_ann"].rolling(window=21, min_periods=21).mean()
        return g
