import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .uv_sampler import UnivariateSampler
from .copula_models import CopulaFactory, CopulaBase
from copula_method import TSInitConfig, TSFitConfig, TSSampleConfig

from contextlib import contextmanager
from joblib import Parallel, delayed
import joblib
from tqdm.auto import tqdm

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Route joblib progress events into a given tqdm progress bar.

    Usage
    -----
    with tqdm_joblib(tqdm(total=...)) as pbar:
        Parallel(...)(...)
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


logger = logging.getLogger(__name__)


class CopulaEstimator:
    """
    Per-day copula calibration from univariate marginals.

    For each target test day t:
      1) Build a lookback window of k days ending at t-1.
      2) Split into W1 (first l days) and W2 (last k-l days).
      3) Fit univariate models on W1; generate UV samples on W2.
      4) Fit the selected copula on the UV samples from W2.
      5) Optionally build day-t marginals (expanding or fixed window).

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]
        Output of DataHandler.get_data(): keys include 'full_data', 'train_set', 'test_set'.
    init_config : TSInitConfig
        Initialization configuration (univariate model type, copula type, rolling window size, etc.).
    fit_config : TSFitConfig
        Univariate fit configuration (ARMA/GARCH parameters, distribution, etc.).
    sample_config : TSSampleConfig
        Sampling configuration (n_samples for UV/multivariate draws).

    Notes
    -----
    - Window sizes:
        k = round(len(train_dates) * rolling_window_size)
        l = round(k * uv_fit_percentage)
      with 1 <= l < k.
    """

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        init_config: TSInitConfig,
        fit_config: TSFitConfig,
        sample_config: TSSampleConfig
    ):
        self.data_dict = data_dict
        self.init_config = init_config
        self.fit_config = fit_config
        self.sample_config = sample_config

        # Unpack data
        self.full_data:  pd.DataFrame = data_dict["full_data"]
        self.train_data: pd.DataFrame = data_dict["train_set"]
        self.test_data:  pd.DataFrame = data_dict["test_set"]

        # Canonical lists
        self.symbols: List[str] = sorted(self.full_data["sym_root"].unique().tolist())
        self.fit_dates:  List[pd.Timestamp] = sorted(self.train_data["date"].unique().tolist())
        self.test_dates: List[pd.Timestamp] = sorted(self.test_data["date"].unique().tolist())
        self.all_dates:  List[pd.Timestamp] = sorted(self.full_data["date"].unique().tolist())
        self.date_index = {d: i for i, d in enumerate(self.all_dates)}

        # Window sizes
        k = int(len(self.fit_dates) * float(self.init_config.rolling_window_size))
        l = int(k * float(self.init_config.uv_fit_percentage))

        if not (0 < self.init_config.rolling_window_size <= 1):
            raise ValueError("rolling_window_size must be in (0, 1].")
        if not (0 < self.init_config.uv_fit_percentage < 1):
            raise ValueError("uv_fit_percentage must be in (0, 1).")
        if k < 2:
            raise ValueError("rolling window (k) too small.")
        if not (1 <= l < k):
            raise ValueError("uv_fit_percentage implies l not in [1, k-1].")

        self.k_days = k               # total lookback
        self.l_days = l               # UV fit window
        self.copula_days = k - l      # copula calibration window

        logger.info(
            "k=%d (lookback), l=%d (UV fit), k-l=%d (copula calibration)",
            self.k_days, self.l_days, self.copula_days
        )

    # ---------------- helpers ----------------

    def _window_dates_for(self, t: pd.Timestamp) -> tuple[list, list]:
        """
        Compute (W1_dates, W2_dates) for target day t.

        Returns
        -------
        (list, list)
            W1 = dates [t-k, ..., t-k+l-1], W2 = dates [t-k+l, ..., t-1].
        """
        t_idx = self.date_index[t]
        if t_idx < self.k_days:
            raise ValueError(f"Not enough history before {t} for k={self.k_days}.")
        window = self.all_dates[t_idx - self.k_days: t_idx]
        w1 = window[: self.l_days]
        w2 = window[self.l_days:]
        return w1, w2

    def _copula_for_day(self, t: pd.Timestamp) -> tuple[pd.Timestamp, CopulaBase] | None:
        """
        Fit a copula object for a single target day t.

        Returns
        -------
        tuple | None
            (t, fitted_copula) or None if insufficient data.
        """
        try:
            w1_dates, w2_dates = self._window_dates_for(t)
        except ValueError as e:
            logger.warning("[Skip %s] %s", t, e)
            return None
        if not w2_dates:
            logger.warning("[Skip %s] empty W2.", t)
            return None

        # Fit UV on W1
        w1_df = self.full_data[self.full_data["date"].isin(w1_dates)].copy()
        sampler = UnivariateSampler(
            data=w1_df,
            method=self.init_config.univariate_type,
            model_params=self.fit_config,
        )

        # One refit at W2[0], sample across W2
        uv_samples = sampler.generate_uv_samples(
            sample_dates=w2_dates,
            symbols=self.symbols,
            n_samples=self.sample_config.n_samples,
            uv_train_freq=len(w2_dates),
            fixed_window=True,
        )

        # Fit copula on UV samples
        copula_params = getattr(self.init_config, "copula_params", {}) or {}
        copula = CopulaFactory.create(
            self.init_config.copula_type,
            n_dim=len(self.symbols),
            copula_params=copula_params
        )
        copula.fit_from_uv_samples(
            full_data=self.full_data,
            uv_samples=uv_samples,
            symbols=self.symbols,
            day=t,
            target_col="ret_crsp",
        )
        return (t, copula)

    def _marginals_for_day(self, t: pd.Timestamp, n: int) -> tuple[pd.Timestamp, dict[str, np.ndarray]] | None:
        """
        Build day-t univariate marginal samples for all symbols.

        Returns
        -------
        tuple | None
            (t, {symbol -> np.ndarray}) or None if no history is available.
        """
        t = pd.Timestamp(t)
        hist_df = self.full_data[self.full_data["date"] < t].copy()
        if hist_df.empty:
            logger.warning("[Skip %s] no history before target.", t)
            return None

        sampler = UnivariateSampler(
            data=hist_df,
            method=self.init_config.univariate_type,
            model_params=self.fit_config,
        )
        uv_t = sampler.generate_uv_samples(
            sample_dates=[t],
            symbols=self.symbols,
            n_samples=int(n),
            uv_train_freq=1,
            fixed_window=False,  # expanding history < t
        )
        return (t, uv_t.get(t, {}))

    # ---------------- public API ----------------

    def build_daily_copulas(self, n_jobs: int = 1) -> Dict[pd.Timestamp, CopulaBase]:
        """
        Fit a copula object for every test day.

        Parameters
        ----------
        n_jobs : int, default=1
            Parallel workers; set -1 to use all cores.

        Returns
        -------
        dict[pd.Timestamp, CopulaBase]
            Mapping day -> fitted copula.
        """
        targets = [pd.Timestamp(t) for t in self.test_dates]
        if n_jobs == 1:
            out: Dict[pd.Timestamp, CopulaBase] = {}
            for t in tqdm(targets, desc="Fitting copulas", leave=False):
                res = self._copula_for_day(t)
                if res is not None:
                    tt, C = res
                    out[tt] = C
            return out

        with tqdm_joblib(tqdm(total=len(targets), desc="Fitting copulas", leave=False)):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._copula_for_day)(t) for t in targets
            )
        return {tt: C for (tt, C) in results if C is not None}

    def build_day_marginals(self, n_samples: int, n_jobs: int = 1) -> Dict[pd.Timestamp, Dict[str, np.ndarray]]:
        """
        Build day-t marginals for all test days.

        Parameters
        ----------
        n_samples : int
            Number of univariate samples per symbol for day t.
        n_jobs : int, default=1
            Parallel workers; set -1 to use all cores.

        Returns
        -------
        dict[pd.Timestamp, dict[str, np.ndarray]]
            Mapping day -> {symbol -> samples}.
        """
        n = int(n_samples)
        targets = [pd.Timestamp(t) for t in self.test_dates]
        if n_jobs == 1:
            out: Dict[pd.Timestamp, Dict[str, np.ndarray]] = {}
            for t in tqdm(targets, desc="Building day-t marginals", leave=False):
                res = self._marginals_for_day(t, n)
                if res is not None:
                    tt, d = res
                    out[tt] = d
            return out

        with tqdm_joblib(tqdm(total=len(targets), desc="Building day-t marginals", leave=False)):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._marginals_for_day)(t, n) for t in targets
            )
        return {tt: d for (tt, d) in results if d is not None}
