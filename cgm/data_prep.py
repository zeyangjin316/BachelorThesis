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

    # Step 5: Identify column groups
    stock_cols = expected_columns
    macro_cols = macro_features
    ret_cols = [f"{stock}_ret_crsp" for stock in expected_stocks]

    # Step 6: Validate return vector shape
    sample_row = df_merged.iloc[window_size + 1]
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

    """print(f"Prepared CGM tensors with {len(Y)} samples. Y dimension: {Y[0].shape[0]}")

    print(f"Y[0] shape: {Y[0].shape}")
    print(f"ret_cols ({len(ret_cols)}): {ret_cols}")

    # Check one target row manually
    print("First Y sample:\n", Y[0].flatten())"""

    return (
        np.array(X_past, dtype=np.float32),     # (N, window_size, D)
        np.array(X_std, dtype=np.float32),      # (N, D)
        np.array(X_all, dtype=np.float32),      # (N, F)
        np.array(X_weekday, dtype=np.int32),    # (N, 1)
        np.array(Y, dtype=np.float32)           # (N, 10, 1)
    )
