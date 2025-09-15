from __future__ import annotations
import pandas as pd
import logging
import numpy as np
import re

logger = logging.getLogger(__name__)

class Reader:
    def __init__(self, base_path: str, ltv_path: str, vix_path: str, intraday_path: str):
        self.base_path = base_path
        self.ltv_path = ltv_path
        self.vix_path = vix_path
        self.intraday_path = intraday_path

        self.base_data = None
        self.ltv_data = None
        self.vix_data = None
        self.intraday_data = None
        self.data = None  # merged daily panel

    def _read_csv(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Error reading {path}: {e}")
        # normalize 'DATE' -> 'date' if present
        for c in df.columns:
            if c.upper() == "DATE":
                df = df.rename(columns={c: "date"})
                break
        return df

    def _merge_on_date(self, ext: pd.DataFrame, prefix: str) -> None:
        # prefix all non-date columns and left-join on date
        ext = ext.rename(columns={c: f"{prefix}_{c.lower()}" for c in ext.columns if c != "date"})
        self.base_data["date"] = pd.to_datetime(self.base_data["date"])
        ext["date"] = pd.to_datetime(ext["date"])
        self.base_data = pd.merge(self.base_data, ext, on="date", how="left")

    def read_all(self) -> None:
        logger.info("Reading daily + intraday CSVs")
        self.base_data = self._read_csv(self.base_path)
        self.ltv_data = self._read_csv(self.ltv_path)
        self.vix_data = self._read_csv(self.vix_path)
        self.intraday_data = self._read_csv(self.intraday_path)

        # ensure datetime where available
        for df in [self.base_data, self.ltv_data, self.vix_data]:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        # merge exogenous
        self._merge_on_date(self.ltv_data, "ltv")
        self._merge_on_date(self.vix_data, "vix")

        # compute RV5m_d and Vol5m_d from intraday wide file (V1..V78)
        v_cols = [c for c in self.intraday_data.columns if re.fullmatch(r"V\d{1,3}", str(c))]
        if not v_cols:
            logger.warning("Intraday file has no V1..V78 columns. Skipping intraday features.")
            self.data = self.base_data.copy()
            return

        v_cols = sorted(v_cols, key=lambda s: int(str(s)[1:]))
        if len(self.base_data) != len(self.intraday_data):
            raise ValueError(
                f"Row mismatch: base={len(self.base_data)} vs intraday={len(self.intraday_data)}. "
                "Files must align by row order."
            )

        vals = self.intraday_data[v_cols].astype(float)
        rv5m = (vals.pow(2).sum(axis=1) * 252.0).to_numpy()
        vol5m = np.sqrt(np.maximum(rv5m, 0.0))

        base = self.base_data.reset_index(drop=True).copy()
        base["RV5m_d"] = rv5m
        base["Vol5m_d"] = vol5m

        self.data = base
