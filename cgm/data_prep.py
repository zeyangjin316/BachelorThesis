import numpy as np
import pandas as pd

def prepare_cgm_inputs(train_data, window_size=20):
    """
    Prepares CGM input tensors using a rolling window for X_past.
    Ensures all 10 stocks and all features are present in the pivoted data.
    """

    # Step 1: Get expected stocks and features
    expected_stocks = sorted(train_data['sym_root'].unique())
    stock_features = ['ret_crsp', 'open_crsp', 'close_crsp', 'log_ret_lag_close_to_open']
    macro_features = [col for col in train_data.columns if col.startswith('vix') or col.startswith('ltv')]

    # Step 2: Pivot stock-level features
    df_pivot = train_data.pivot(index='date', columns='sym_root', values=stock_features)
    df_pivot.columns = [f"{stock}_{feat}" for feat, stock in df_pivot.columns]

    # Step 3: Validate expected pivot columns
    expected_columns = [f"{stock}_{feat}" for stock in expected_stocks for feat in stock_features]
    missing_columns = set(expected_columns) - set(df_pivot.columns)

    if missing_columns:
        print("Missing columns after pivot:")
        for col in sorted(missing_columns):
            print(f"  - {col}")
        raise ValueError("Aborting due to missing stock-feature columns after pivot.")

    # Step 4: Merge macro data
    df_macro = train_data.drop_duplicates(subset='date')[['date'] + macro_features]
    df_macro['date'] = pd.to_datetime(df_macro['date'])
    df_pivot = df_pivot.reset_index()
    df_merged = df_pivot.merge(df_macro, on='date', how='inner')
    df_merged = df_merged.sort_values('date').reset_index(drop=True)

    # Check if we have enough data
    if len(df_merged) <= window_size + 1:
        raise ValueError(f"Not enough data points. Got {len(df_merged)} rows, need at least {window_size + 2} rows.")

    # Step 5: Identify column groups
    stock_cols = expected_columns
    macro_cols = macro_features
    ret_cols = [f"{stock}_ret_crsp" for stock in expected_stocks]

    # Step 6: Validate return vector shape
    sample_row = df_merged.iloc[window_size]  # Changed from window_size + 1
    sample_target = sample_row[ret_cols].values.reshape(-1, 1)
    if sample_target.shape[0] != 10:
        raise ValueError(f"Target Y dimension is {sample_target.shape[0]}, expected 10.")

    # Step 7: Build tensors
    X_past, X_std, X_all, X_weekday, Y = [], [], [], [], []
    full_std = df_merged[stock_cols].std().values

    for i in range(window_size, len(df_merged) - 1):
        past_window = df_merged.iloc[i - window_size:i][stock_cols].values
        today = df_merged.iloc[i]
        tomorrow = df_merged.iloc[i + 1]

        X_past.append(past_window)
        X_std.append(full_std)
        X_all.append(today[macro_cols].values)
        X_weekday.append([pd.to_datetime(today['date']).weekday()])
        Y.append(tomorrow[ret_cols].values.reshape(-1, 1))

    return (
        np.array(X_past, dtype=np.float32),     # (N, window_size, D)
        np.array(X_std, dtype=np.float32),      # (N, D)
        np.array(X_all, dtype=np.float32),      # (N, F)
        np.array(X_weekday, dtype=np.int32),    # (N, 1)
        np.array(Y, dtype=np.float32)           # (N, 10, 1)
    )


def prepare_cgm_inputs_for_sampling(data: pd.DataFrame, window_size: int):
    """
    Prepares CGM inputs for a single test day (multi-asset), matching training dimensions.
    Returns:
        X_past:     (1, window_size, 40)  ← 10 stocks × 4 features
        X_std:      (1, 40)
        X_all:      (1, F)
        X_weekday:  (1, 1)
    """

    expected_stocks = sorted(data['sym_root'].unique())
    stock_features = ['ret_crsp', 'open_crsp', 'close_crsp', 'log_ret_lag_close_to_open']
    macro_features = [col for col in data.columns if col.startswith('vix') or col.startswith('ltv')]

    # Step 1: Pivot stock features
    df_pivot = data.pivot(index='date', columns='sym_root', values=stock_features)
    df_pivot.columns = [f"{stock}_{feat}" for feat, stock in df_pivot.columns]

    expected_columns = [f"{stock}_{feat}" for stock in expected_stocks for feat in stock_features]
    missing = set(expected_columns) - set(df_pivot.columns)
    if missing:
        raise ValueError(f"Missing stock-feature columns after pivot: {missing}")

    # Step 2: Merge macro + compute weekday
    df_macro = data.drop_duplicates(subset='date')[['date'] + macro_features]
    df_macro['date'] = pd.to_datetime(df_macro['date'])
    df_macro['weekday'] = df_macro['date'].dt.weekday

    df_pivot = df_pivot.reset_index()
    df_merged = df_pivot.merge(df_macro, on='date', how='inner').sort_values('date').reset_index(drop=True)

    if len(df_merged) < window_size + 1:
        raise ValueError(f"Not enough data for sampling (have {len(df_merged)}, need {window_size + 1})")

    # Step 3: Slice past window and current row
    past_window = df_merged.iloc[-(window_size + 1):-1]  # (window_size, 40)
    today = df_merged.iloc[-1]

    # Step 4: Prepare input arrays
    X_past = past_window[expected_columns].values.reshape(1, window_size, -1)    # (1, window_size, 40)
    X_std = past_window[expected_columns].std(axis=0).values.reshape(1, -1)      # (1, 40)
    X_all = today[macro_features].values.astype(np.float32).reshape(1, -1)       # (1, F)
    X_weekday = np.array([[pd.to_datetime(today['date']).weekday()]], dtype=np.int32)  # (1, 1)

    return X_past.astype(np.float32), X_std.astype(np.float32), X_all, X_weekday