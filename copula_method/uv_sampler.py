import logging
from typing import Dict, List, Iterable, Optional
import numpy as np
import pandas as pd

from .uv_models import create_uv_model
# ensure model classes register themselves (e.g., ARMAGARCH)
from . import arma_garch


logger = logging.getLogger(__name__)


class UnivariateSampler:
    """
    Per-window sampler.

    Expected usage:
      - self.full_data should already be your W1 (UV-fit) history
      - test_dates should be W2 dates (the copula-calibration days)
      - set uv_train_freq=None or len(W2) to fit once and sample all W2 days
    """

    def __init__(self,
                 data: pd.DataFrame,
                 method: str,
                 model_params):
        self.full_data = data
        self.method = method
        self.model_params = model_params

        # Currently only fixed rolling window
        self.rolling_window_size = int(pd.Series(self.full_data["date"]).nunique())

    def _fit_and_sample_block(
            self,
            start_date: pd.Timestamp,
            block_dates: Iterable[pd.Timestamp],
            symbols: List[str],
            n_samples: int,
            fixed_window: bool,
        ) -> Dict[pd.Timestamp, Dict[str, np.ndarray]]:
        """
        Fit once at refit_date using history < refit_date (within self.full_data),
        then sample each day in block_dates with the same fitted model.
        """
        # 1) training slice (already limited to W1 by caller)
        data_up_to_date = self.full_data[self.full_data["date"] < start_date]

        if fixed_window:
            data_up_to_date = (
                data_up_to_date.sort_values(["sym_root", "date"])
                .groupby("sym_root", group_keys=False)
                .tail(self.rolling_window_size)
            )

        if data_up_to_date.empty:
            logger.warning(f"[UV] Training slice empty before {start_date}; using all provided data.")
            data_up_to_date = self.full_data.copy()

        # 2) fit once
        model = create_uv_model(self.method, data_up_to_date, self.model_params)
        model.fit(current_day=start_date)

        # 3) sample rest of block
        out_for_block: Dict[pd.Timestamp, Dict[str, np.ndarray]] = {}
        n = int(n_samples)
        for day in block_dates:
            per_symbol: Dict[str, np.ndarray] = {}
            for sym in symbols:
                try:
                    per_symbol[sym] = model.sample(sym, n_samples=n)
                except Exception as e:
                    logger.warning(f"[UV] Failed sampling {sym} on {day}: {e}")
                    per_symbol[sym] = np.array([])
            out_for_block[pd.Timestamp(day)] = per_symbol

        return out_for_block

    def generate_uv_samples(
            self,
            sample_dates: Iterable[pd.Timestamp],
            symbols: List[str],
            n_samples: int,
            uv_train_freq: Optional[int] = None,
            fixed_window: bool = True,
        ) -> Dict[pd.Timestamp, Dict[str, np.ndarray]]:
        """
          - Partition test_dates into blocks according to uv_train_freq (if provided),
            otherwise one block covering all dates.
          - For each block, fit once at its first date and sample for all dates in that block.
        """
        dates = list(pd.to_datetime(list(sample_dates)))
        if not dates:
            return {}

        # make blocks (sequentially processed)
        if uv_train_freq is None or int(uv_train_freq) >= len(dates):
            blocks = [(dates[0], dates)]
        else:
            freq = int(uv_train_freq)
            blocks = []
            for start in range(0, len(dates), freq):
                refit_date = dates[start]
                block_dates = dates[start:start + freq]
                blocks.append((refit_date, block_dates))

        logger.info(f"UV fit+sample over {len(blocks)} block(s).")

        uv_samples: Dict[pd.Timestamp, Dict[str, np.ndarray]] = {}
        for refit_date, block_dates in blocks:
            block_dict = self._fit_and_sample_block(
                start_date=refit_date,
                block_dates=block_dates,
                symbols=symbols,
                n_samples=n_samples,
                fixed_window=fixed_window,
            )
            uv_samples.update(block_dict)

        logger.info("Finished UV sample generation.")
        return uv_samples