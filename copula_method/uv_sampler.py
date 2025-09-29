import logging
from typing import Dict, List, Iterable, Optional
import numpy as np
import pandas as pd

from .uv_models import create_uv_model
from . import arma_garch

logger = logging.getLogger(__name__)


class UnivariateSampler:
    """
    Wrapper to fit per-symbol univariate models and generate predictive samples.

    Typical usage in the Two-Step copula pipeline:
      - Provide W1 history (`data`) to initialize.
      - Call `generate_uv_samples(...)` on W2 dates (copula calibration days).
      - Controls:
          * `uv_train_freq`: how often to refit (None or ≥len(W2) = fit once).
          * `fixed_window`: whether to keep only last k days for training.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 method: str,
                 model_params):
        self.full_data = data
        self.method = method
        self.model_params = model_params

        # number of unique training dates (used when fixed_window=True)
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
        Fit once at `start_date` using history < start_date, then sample all block_dates.
        """
        # 1) training slice
        data_up_to_date = self.full_data[self.full_data["date"] < start_date]

        if fixed_window:
            data_up_to_date = (
                data_up_to_date.sort_values(["sym_root", "date"])
                .groupby("sym_root", group_keys=False)
                .tail(self.rolling_window_size)
            )

        if data_up_to_date.empty:
            logger.warning(f"[UV] Training slice empty before {start_date}; using full_data instead.")
            data_up_to_date = self.full_data.copy()

        # 2) fit UV model
        model = create_uv_model(self.method, data_up_to_date, self.model_params)
        model.fit(current_day=start_date)

        # 3) sample forecasts for each block date
        out_for_block: Dict[pd.Timestamp, Dict[str, np.ndarray]] = {}
        n = int(n_samples)
        for day in block_dates:
            per_symbol: Dict[str, np.ndarray] = {}
            for sym in symbols:
                try:
                    per_symbol[sym] = model.sample(sym, n_samples=n)
                except Exception as e:
                    logger.warning(f"[UV] Sampling failed for {sym} on {day}: {e}")
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
        Generate predictive samples for a set of dates.

        Parameters
        ----------
        sample_dates : Iterable[pd.Timestamp]
            Target dates to generate samples for (e.g. W2).
        symbols : list of str
            Symbols to forecast.
        n_samples : int
            Number of samples per symbol/date.
        uv_train_freq : int, optional
            Refit frequency (in days). If None or ≥len(dates), fit once and reuse.
        fixed_window : bool, default=True
            Whether to restrict training history to last k days.

        Returns
        -------
        dict
            Mapping {date -> {symbol -> np.ndarray(samples)}}.
        """
        dates = list(pd.to_datetime(list(sample_dates)))
        if not dates:
            return {}

        # Partition into refit blocks
        if uv_train_freq is None or int(uv_train_freq) >= len(dates):
            blocks = [(dates[0], dates)]
        else:
            freq = int(uv_train_freq)
            blocks = []
            for start in range(0, len(dates), freq):
                refit_date = dates[start]
                block_dates = dates[start:start + freq]
                blocks.append((refit_date, block_dates))

        logger.info(f"[UV] Generating samples over {len(blocks)} block(s).")

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

        logger.info("[UV] Finished sample generation.")
        return uv_samples
