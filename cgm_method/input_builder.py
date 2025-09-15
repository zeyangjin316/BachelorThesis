import numpy as np
import pandas as pd
from typing import List, Tuple
from data.data_scaling import SmartScaler

class CGMInputBuilder:
    """
    Builds CGM tensors with automatic scaling and a consistent X_std policy.

    Parameters
    ----------
    window_size : int
        Past window length (CGM.past_len).
    std_policy : {'full','window'}
        'full'   -> X_std from full (scaled) merged training table; reused at sampling.
        'window' -> X_std from the scaled past window; same in training & sampling.
    macro_prefixes : list[str]
        Column prefixes used to collect macro features (e.g. ['vix','ltv']).
    """

    def __init__(self, window_size: int = 20, std_policy: str = "window"):
        self.window_size = window_size
        assert std_policy in {"full", "window"}
        self.std_policy = std_policy

        # explicit list of date-level (macro) prefixes
        self.macro_prefixes: List[str] = ["vix", "ltv"]

        # filled automatically from training data
        self.stock_features: List[str] = []
        self.target_col: str = "ret_crsp"

        # meta attributes
        self.scaler = None
        self.std_vector_full = None
        self.expected_stocks = None
        self.dim_in_past = None
        self.dim_out = None
        self.dim_in_features = None

    # ---------------- public API ----------------

    def fit_prepare(self, train_data: pd.DataFrame):
        """Fit scaler on train_data, transform, and build training tensors."""
        self.scaler = SmartScaler(train_data)
        scaled = self.scaler.transform(train_data)

        X_past, X_std_full, X_all, X_weekday, Y, meta = self._prepare_training_core(scaled)

        if self.std_policy == "window":
            X_std = X_past.std(axis=1).astype(np.float32)
            self.std_vector_full = None
        else:  # 'full'
            X_std = X_std_full
            self.std_vector_full = X_std[0].copy() if X_std.shape[0] else None

        self.expected_stocks = meta["expected_stocks"]
        self.dim_in_past = meta["dim_in_past"]
        self.dim_out = meta["dim_out"]
        self.dim_in_features = meta["dim_in_features"]

        tensors = {"X_past": X_past, "X_std": X_std, "X_all": X_all, "X_weekday": X_weekday, "Y": Y}
        self._check_tensors(tensors, context="fit_prepare")

        return X_past, X_std, X_all, X_weekday, Y

    def prepare_for_sampling(self, data: pd.DataFrame):
        """Use fitted scaler to transform `data` and prepare inputs for predict()."""
        if self.scaler is None:
            raise RuntimeError("Call fit_prepare(...) first.")

        scaled = self.scaler.transform(data)
        X_past, X_std_window, X_all, X_weekday = self._prepare_sampling_core(scaled)

        if self.std_policy == "window":
            X_std = X_std_window
        else:  # 'full'
            if self.std_vector_full is None:
                raise RuntimeError("Missing std_vector_full; fit was not run with std_policy='full'.")
            X_std = np.asarray(self.std_vector_full, dtype=np.float32).reshape(1, -1)

        tensors = {"X_past": X_past, "X_std": X_std, "X_all": X_all, "X_weekday": X_weekday}
        self._check_tensors(tensors, context="prepare_for_sampling")

        return X_past, X_std, X_all, X_weekday

    def model_dims(self) -> Tuple[int, int, int]:
        """Returns (dim_out, dim_in_features, dim_in_past)."""
        if None in (self.dim_out, self.dim_in_features, self.dim_in_past):
            raise RuntimeError("Call fit_prepare(...) first.")
        return self.dim_out, self.dim_in_features, self.dim_in_past

    # ---------------- internal helpers ----------------

    def _collect_macros(self, df: pd.DataFrame) -> List[str]:
        cols: List[str] = []
        for p in self.macro_prefixes:
            cols.extend([c for c in df.columns if c.startswith(p)])
        seen, uniq = set(), []
        for c in cols:
            if c not in seen:
                uniq.append(c); seen.add(c)
        return uniq

    def _collect_stock_features(self, df: pd.DataFrame) -> List[str]:
        """Everything numeric except ids, target, and macros."""
        id_like = {"date", "sym_root", "permno"}
        exclude = id_like | {self.target_col}

        def is_macro(c: str) -> bool:
            return any(c.startswith(p) for p in self.macro_prefixes)

        numeric_cols = df.select_dtypes(include="number").columns
        candidates = [c for c in numeric_cols if c not in exclude and not is_macro(c)]
        return [c for c in df.columns if c in candidates]

    def _pivot_stock_features(self, df: pd.DataFrame, expected_stocks: List[str], pivot_features: List[str]) -> pd.DataFrame:
        df_pivot = df.pivot(index="date", columns="sym_root", values=pivot_features)
        df_pivot.columns = [f"{stock}_{feat}" for feat, stock in df_pivot.columns]

        expected_columns = [f"{s}_{f}" for s in expected_stocks for f in pivot_features]
        missing = set(expected_columns) - set(df_pivot.columns)
        if missing:
            missing_list = "\n  - " + "\n  - ".join(sorted(missing))
            raise ValueError(f"Missing stock-feature columns after pivot:{missing_list}")
        return df_pivot

    def _merge_macro(self, df_base: pd.DataFrame, df: pd.DataFrame, macro_features: List[str]) -> pd.DataFrame:
        if macro_features:
            df_macro = df.drop_duplicates(subset="date")[["date"] + macro_features].copy()
        else:
            df_macro = df[["date"]].drop_duplicates().copy()
        df_macro["date"] = pd.to_datetime(df_macro["date"])
        df_base = df_base.reset_index()
        df_merged = df_base.merge(df_macro, on="date", how="inner").sort_values("date").reset_index(drop=True)
        return df_merged

    def _prepare_training_core(self, train_data: pd.DataFrame):
        expected_stocks = sorted(train_data["sym_root"].unique())
        self.stock_features = self._collect_stock_features(train_data)
        macro_features = self._collect_macros(train_data)

        # Pivot features + target so Y can be built
        pivot_features = self.stock_features + [self.target_col]
        df_pivot = self._pivot_stock_features(train_data, expected_stocks, pivot_features)
        df_merged = self._merge_macro(df_pivot, train_data, macro_features)

        if len(df_merged) <= self.window_size + 1:
            raise ValueError(f"Not enough data: have {len(df_merged)}, need ≥ {self.window_size + 2}")

        stock_cols = [f"{s}_{f}" for s in expected_stocks for f in self.stock_features]
        ret_cols   = [f"{s}_{self.target_col}" for s in expected_stocks]

        X_past, X_std_full, X_all, X_weekday, Y = [], [], [], [], []
        full_std_vec = df_merged[stock_cols].std().values.astype(np.float32)

        for i in range(self.window_size, len(df_merged) - 1):
            past_window = df_merged.iloc[i - self.window_size:i][stock_cols].values
            today = df_merged.iloc[i]
            tomorrow = df_merged.iloc[i + 1]

            X_past.append(past_window)
            X_std_full.append(full_std_vec)
            X_all.append(today[macro_features].values if macro_features else [])
            X_weekday.append([pd.to_datetime(today["date"]).weekday()])
            Y.append(tomorrow[ret_cols].values.reshape(-1, 1))

        X_past = np.array(X_past, dtype=np.float32)
        X_std_full = np.array(X_std_full, dtype=np.float32)
        X_all = np.array(X_all, dtype=np.float32) if macro_features else np.zeros((len(X_past), 0), dtype=np.float32)
        X_weekday = np.array(X_weekday, dtype=np.int32)
        Y = np.array(Y, dtype=np.float32)

        meta = dict(
            expected_stocks=expected_stocks,
            dim_in_past=len(stock_cols),
            dim_out=len(expected_stocks),
            dim_in_features=len(macro_features),
        )
        return X_past, X_std_full, X_all, X_weekday, Y, meta

    def _prepare_sampling_core(self, data: pd.DataFrame):
        expected_stocks = self.expected_stocks or sorted(data["sym_root"].unique())
        macro_features = self._collect_macros(data)

        pivot_features = self.stock_features + [self.target_col]
        df_pivot = self._pivot_stock_features(data, expected_stocks, pivot_features)
        df_merged = self._merge_macro(df_pivot, data, macro_features)

        if len(df_merged) < self.window_size + 1:
            raise ValueError(f"Not enough data for sampling: have {len(df_merged)}, need ≥ {self.window_size + 1}")

        stock_cols = [f"{s}_{f}" for s in expected_stocks for f in self.stock_features]
        past_window = df_merged.iloc[-(self.window_size + 1):-1]
        today = df_merged.iloc[-1]

        X_past = past_window[stock_cols].values.reshape(1, self.window_size, -1).astype(np.float32)
        X_std_window = past_window[stock_cols].std(axis=0).values.reshape(1, -1).astype(np.float32)
        X_all = today[macro_features].values.astype(np.float32).reshape(1, -1) if macro_features else np.zeros((1, 0), dtype=np.float32)
        X_weekday = np.array([[pd.to_datetime(today["date"]).weekday()]], dtype=np.int32)

        return X_past, X_std_window, X_all, X_weekday

    def _check_tensors(self, tensors: dict, context: str = ""):
        """
        Ensure that all tensors contain only finite values.
        If invalids are found, raise RuntimeError with details including bad columns.
        """
        for name, arr in tensors.items():
            if arr is None:
                continue
            arr = np.asarray(arr)
            if not np.isfinite(arr).all():
                nan_mask = np.isnan(arr)
                inf_mask = np.isinf(arr)
                nan_count = nan_mask.sum()
                inf_count = inf_mask.sum()

                msg = [f"[CGMInputBuilder] Invalid values in {name} during {context}:"]
                msg.append(f"  NaNs: {nan_count}, Infs: {inf_count}")

                # Try to infer feature names automatically
                bad_cols = []
                if name == "X_past" and hasattr(self, "expected_stocks") and self.stock_features:
                    stock_cols = [f"{s}_{f}" for s in self.expected_stocks for f in self.stock_features]
                    if arr.ndim == 3 and arr.shape[2] == len(stock_cols):
                        for j, col in enumerate(stock_cols):
                            if nan_mask[..., j].any() or inf_mask[..., j].any():
                                bad_cols.append(col)
                elif name == "X_all":
                    macro_cols = self._collect_macros(self.scaler.data)
                    if arr.ndim == 2 and arr.shape[1] == len(macro_cols):
                        for j, col in enumerate(macro_cols):
                            if nan_mask[:, j].any() or inf_mask[:, j].any():
                                bad_cols.append(col)
                elif name == "Y" and hasattr(self, "expected_stocks"):
                    ret_cols = [f"{s}_{self.target_col}" for s in self.expected_stocks]
                    if arr.ndim == 3 and arr.shape[1] == len(ret_cols):
                        for j, col in enumerate(ret_cols):
                            if nan_mask[:, j, :].any() or inf_mask[:, j, :].any():
                                bad_cols.append(col)

                if bad_cols:
                    preview = ", ".join(bad_cols[:10])
                    if len(bad_cols) > 10:
                        preview += ", ..."
                    msg.append(f"  Problematic columns: {preview}")

                raise RuntimeError("\n".join(msg))
