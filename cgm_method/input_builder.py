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

        # date-level (macro) prefixes
        self.macro_prefixes: List[str] = ["vix", "ltv"]

        # auto-filled from data
        self.stock_features: List[str] = []
        self.target_col: str = "ret_crsp"

        # meta
        self.scaler = None
        self.std_vector_full = None
        self.expected_stocks = None
        self.dim_in_past = None
        self.dim_out = None
        self.dim_in_features = None

        # debug
        self._last_training_dates: List[pd.Timestamp] = []

    # ---------------- public API ----------------

    def fit_prepare(self, train_data: pd.DataFrame):
        """Fit scaler on train_data, transform, and build training tensors."""
        # drop initial warm-up rows per stock (for rolling features)
        train_data = self._drop_incomplete_periods(train_data)

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
        """Use the fitted scaler to transform `data` and prepare inputs for predict()."""
        if self.scaler is None:
            raise RuntimeError("Call fit_prepare(...) first.")

        data = self._drop_incomplete_periods(data)
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

    def _drop_incomplete_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the first `max_window` rows per stock (sym_root) to avoid NaNs from rolling features.
        max_window is inferred from the largest lag/roll window present in the feature names.
        """
        import re
        max_window = 0
        for c in df.columns:
            m = re.search(r"roll(\d+)", c)
            if m:
                max_window = max(max_window, int(m.group(1)))
            m = re.search(r"hl(\d+)", c)
            if m:
                max_window = max(max_window, int(m.group(1)))

        if max_window == 0:
            return df

        df = df.sort_values(["sym_root", "date"])
        cleaned = (
            df.groupby("sym_root", group_keys=False)
              .apply(lambda g: g.iloc[max_window:].copy())
              .reset_index(drop=True)
        )
        print(f"[CGMInputBuilder] Dropped first {max_window} rows per stock to remove incomplete rolling windows.")
        return cleaned

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
        # preserve original column order
        return [c for c in df.columns if c in candidates]

    def _pivot_stock_features(self, df: pd.DataFrame, expected_stocks: list[str],
                              pivot_features: list[str]) -> pd.DataFrame:
        """
        Make a fully balanced long panel (every stock has every date), then pivot.
        We ffill within each stock, and bfill once at the very start to avoid leading NaNs.
        Output columns are ordered as [stock1_feat1, ..., stockN_featM], rows by date asc.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["sym_root"] = pd.Categorical(df["sym_root"], categories=expected_stocks, ordered=True)

        # All trading dates present in the dataset
        all_dates = np.sort(df["date"].unique())

        # Build a balanced long panel per stock
        balanced = []
        need_cols = ["date", "sym_root"] + pivot_features
        for s in expected_stocks:
            g = df.loc[df["sym_root"] == s, need_cols].copy()
            g = g.set_index("date").reindex(all_dates)  # create (possibly) missing dates
            # restore stock id
            g["sym_root"] = s
            # ffill for internal gaps, then bfill ONCE for initial gap (if any)
            g[pivot_features] = g[pivot_features].ffill()
            g[pivot_features] = g[pivot_features].bfill(limit=1)
            # keep only rows where we now have data
            balanced.append(g.reset_index().rename(columns={"index": "date"}))

        long_balanced = pd.concat(balanced, ignore_index=True)

        # deterministic ordering
        long_balanced = long_balanced.sort_values(["date", "sym_root"])

        # Pivot with values=pivot_features (MultiIndex cols (feature, stock))
        df_pivot = long_balanced.pivot(index="date", columns="sym_root", values=pivot_features)

        # Normalize MultiIndex → "stock_feature" regardless of internal order
        if isinstance(df_pivot.columns, pd.MultiIndex):
            # figure out which level is stock by membership
            lvl0_vals = set(df_pivot.columns.get_level_values(0))
            if any(s in lvl0_vals for s in expected_stocks):
                stock_level, feat_level = 0, 1
            else:
                stock_level, feat_level = 1, 0
            df_pivot.columns = [
                f"{col[stock_level]}_{col[feat_level]}" for col in df_pivot.columns
            ]
        else:
            df_pivot.columns = [str(c) for c in df_pivot.columns]

        # Enforce exact column order without creating NaN columns
        ordered = [f"{s}_{f}" for s in expected_stocks for f in pivot_features]
        keep_cols = [c for c in ordered if c in df_pivot.columns]
        df_pivot = df_pivot[keep_cols]

        # Final ordering
        df_pivot = df_pivot.sort_index()
        df_pivot = df_pivot.apply(pd.to_numeric, errors="coerce")

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

    # ---------------- core builders ----------------

    def _prepare_training_core(self, train_data: pd.DataFrame):
        expected_stocks = sorted(train_data["sym_root"].unique())
        self.stock_features = self._collect_stock_features(train_data)
        macro_features = self._collect_macros(train_data)

        # Pivot features + target so Y can be built
        pivot_features = self.stock_features + [self.target_col]
        df_pivot = self._pivot_stock_features(train_data, expected_stocks, pivot_features)

        # (optional but robust) drop dates with any missing stock-feature cell
        nan_per_row = df_pivot.isna().any(axis=1)
        n_bad = int(nan_per_row.sum())
        if n_bad > 0:
            ex = list(df_pivot.index[nan_per_row][:5])
            print(f"[CGMInputBuilder] Dropping {n_bad} dates with incomplete stock panel. Examples: {ex}")
            df_pivot = df_pivot.loc[~nan_per_row]
        if df_pivot.empty:
            raise ValueError(
                "[CGMInputBuilder] After dropping incomplete dates, the panel is empty. "
                "Consider imputing or removing sparse features/stocks."
            )

        df_merged = self._merge_macro(df_pivot, train_data, macro_features)

        if len(df_merged) <= self.window_size + 1:
            raise ValueError(f"Not enough data: have {len(df_merged)}, need ≥ {self.window_size + 2}")

        stock_cols = [f"{s}_{f}" for s in expected_stocks for f in self.stock_features]
        ret_cols   = [f"{s}_{self.target_col}" for s in expected_stocks]

        # tensors
        dates = pd.to_datetime(df_merged["date"].values)
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

        # map window-end dates for debugging
        self._last_training_dates = [dates[i] for i in range(self.window_size, len(df_merged) - 1)]

        # save debug tensors
        import os
        os.makedirs("debug_tensors", exist_ok=True)
        np.save("debug_tensors/X_past.npy", X_past)
        np.save("debug_tensors/X_std_full.npy", X_std_full)
        np.save("debug_tensors/X_all.npy", X_all)
        np.save("debug_tensors/X_weekday.npy", X_weekday)
        np.save("debug_tensors/Y.npy", Y)
        print(
            f"[CGMInputBuilder] Saved training tensors to ./debug_tensors/ "
            f"(shapes: X_past={X_past.shape}, X_std_full={X_std_full.shape}, "
            f"X_all={X_all.shape}, X_weekday={X_weekday.shape}, Y={Y.shape})"
        )

        meta = dict(
            expected_stocks=expected_stocks,
            dim_in_past=len(stock_cols),
            dim_out=len(expected_stocks),
            dim_in_features=len(macro_features),
        )
        return X_past, X_std_full, X_all, X_weekday, Y, meta

    def _prepare_sampling_core(self, data: pd.DataFrame):
        """
        Build inputs for predict() using the SAME policy as training:
          - balance-first pivot (via _pivot_stock_features)
          - no row dropping after pivot
          - identical column ordering
        Returns X_past, X_std_window, X_all, X_weekday.
        """
        expected_stocks = self.expected_stocks or sorted(data["sym_root"].unique())
        macro_features = self._collect_macros(data)

        # Use the exact features learned at training time
        if not self.stock_features:
            raise RuntimeError("stock_features not set. Call fit_prepare(...) first.")

        pivot_features = self.stock_features + [self.target_col]

        # Balanced pivot (ffill + bfill(1)) — same as training
        df_pivot = self._pivot_stock_features(data, expected_stocks, pivot_features)

        # Merge macros without dropping rows
        df_merged = self._merge_macro(df_pivot, data, macro_features)

        # Need at least W+1 rows to form the last window (predict next day)
        if len(df_merged) < self.window_size + 1:
            raise ValueError(
                f"Not enough data for sampling: have {len(df_merged)}, need ≥ {self.window_size + 1}"
            )

        stock_cols = [f"{s}_{f}" for s in expected_stocks for f in self.stock_features]
        past_window = df_merged.iloc[-(self.window_size + 1):-1]  # [t-W, ..., t-1]
        today = df_merged.iloc[-1]  # t

        # Build tensors (mirror training shapes/dtypes)
        X_past = past_window[stock_cols].values.reshape(1, self.window_size, -1).astype(np.float32)
        X_std_window = past_window[stock_cols].std(axis=0).values.reshape(1, -1).astype(np.float32)

        if macro_features:
            X_all = today[macro_features].values.astype(np.float32).reshape(1, -1)
        else:
            X_all = np.zeros((1, 0), dtype=np.float32)

        X_weekday = np.array([[pd.to_datetime(today["date"]).weekday()]], dtype=np.int32)

        # For debugging/alignment in evaluation (optional but handy)
        self._last_sampling_date = pd.to_datetime(today["date"])

        return X_past, X_std_window, X_all, X_weekday

    # ---------------- minimal tensor checker ----------------

    def _check_tensors(self, tensors: dict, context: str = ""):
        """
        Minimal finiteness check:
          - If a tensor has NaN/Inf, save bad rows/cells to CSV and raise with brief summary.
          - Saves to ./debug_bad_rows/.
        """
        import os
        import pandas as pd

        os.makedirs("debug_bad_rows", exist_ok=True)

        for name, arr in tensors.items():
            if arr is None:
                continue

            arr = np.asarray(arr)
            if np.isfinite(arr).all():
                continue  # all good

            nan_mask = np.isnan(arr)
            inf_mask = np.isinf(arr)
            nan_count = int(nan_mask.sum())
            inf_count = int(inf_mask.sum())

            rows_path = None
            cells_path = None

            # X_past: (N, W, D)
            if name == "X_past" and hasattr(self, "expected_stocks") and self.stock_features and arr.ndim == 3:
                N, W, D = arr.shape
                row_bad = ~np.isfinite(arr).all(axis=(1, 2))
                rows_df = pd.DataFrame({
                    "sample_idx": np.arange(N)[row_bad],
                    "window_end_date": (
                        [self._last_training_dates[i] for i in np.arange(N)[row_bad]]
                        if hasattr(self, "_last_training_dates") else None
                    ),
                })
                rows_path = os.path.join("debug_bad_rows", f"bad_{name}_{context}_rows.csv")
                rows_df.to_csv(rows_path, index=False)

                # per-cell info if we can map features
                stock_cols = [f"{s}_{f}" for s in self.expected_stocks for f in self.stock_features]
                if D == len(stock_cols):
                    s_idx, t_idx, f_idx = np.where(~np.isfinite(arr))
                    cells_df = pd.DataFrame({
                        "sample_idx": s_idx,
                        "timestep": t_idx,
                        "feature": [stock_cols[j] for j in f_idx],
                        "is_nan": np.isnan(arr[s_idx, t_idx, f_idx]),
                        "is_inf": np.isinf(arr[s_idx, t_idx, f_idx]),
                        "window_end_date": (
                            [self._last_training_dates[i] for i in s_idx]
                            if hasattr(self, "_last_training_dates") else None
                        ),
                    })
                    cells_path = os.path.join("debug_bad_rows", f"bad_{name}_{context}_cells.csv")
                    cells_df.to_csv(cells_path, index=False)

            # X_all: (N, F)
            elif name == "X_all" and arr.ndim == 2:
                row_bad = ~np.isfinite(arr).all(axis=1)
                rows_df = pd.DataFrame({"sample_idx": np.arange(arr.shape[0])[row_bad]})
                rows_path = os.path.join("debug_bad_rows", f"bad_{name}_{context}_rows.csv")
                rows_df.to_csv(rows_path, index=False)

            # Y: (N, S, 1)
            elif name == "Y" and arr.ndim == 3:
                row_bad = ~np.isfinite(arr).all(axis=(1, 2))
                rows_df = pd.DataFrame({"sample_idx": np.arange(arr.shape[0])[row_bad]})
                rows_path = os.path.join("debug_bad_rows", f"bad_{name}_{context}_rows.csv")
                rows_df.to_csv(rows_path, index=False)

            where = []
            if rows_path: where.append(f"rows={rows_path}")
            if cells_path: where.append(f"cells={cells_path}")
            where_str = (" -> " + ", ".join(where)) if where else ""
            raise RuntimeError(
                f"[CGMInputBuilder] Invalid values in {name} during {context}: "
                f"NaNs={nan_count}, Infs={inf_count}{where_str}"
            )
