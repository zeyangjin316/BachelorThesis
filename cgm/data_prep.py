import numpy as np
import pandas as pd

def prepare_cgm_inputs(train_data):
    """
    Prepares CGM input tensors from the full dataset.
    Each day is one sample. No rolling past window is used.

    Assumes:
    - Long-format dataframe: one row per stock per day
    - Stock-specific columns: 'ret_crsp', 'open_crsp', etc.
    - Macro columns: those starting with 'vix' or 'ltv'
    """
    # === Step 1: Keep only one row per date for VIX/LTV macro variables ===
    df_market = train_data.drop_duplicates(subset=['date'])[
        ['date'] + [col for col in train_data.columns if col.startswith('vix') or col.startswith('ltv')]
    ]

    # === Step 2: Pivot the stock-level features into wide format (1 row per date) ===
    stock_features = ['ret_crsp', 'open_crsp', 'close_crsp', 'log_ret_lag_close_to_open']
    df_pivot = train_data.pivot(index='date', columns='sym_root', values=stock_features)

    # Flatten multiindex columns → e.g. ('ret_crsp', 'MSFT') → 'MSFT_ret_crsp'
    df_pivot.columns = [f"{stock}_{feat}" for feat, stock in df_pivot.columns]

    # === Step 3: Merge macro variables with wide stock feature table ===
    df_final = df_pivot.merge(df_market, on='date')
    df_final = df_final.sort_values('date').reset_index(drop=True)

    # === Step 4: Identify feature columns ===
    stock_cols = [col for col in df_final.columns if '_' in col and not col.startswith(('vix', 'ltv'))]
    macro_cols = [col for col in df_final.columns if col.startswith(('vix', 'ltv'))]
    ret_cols = [col for col in df_final.columns if col.endswith('_ret_crsp')]

    # === Step 5: Build tensors ===
    X_past = []       # Will use stock features as if all observed "today"
    X_std = []        # Std dev of stock features across the full train set
    X_all = []        # Macro inputs (vix/ltv)
    X_weekday = []    # Weekday (0–6)
    Y = []            # Target: return vector of all 10 stocks

    for i in range(len(df_final) - 1):
        today = df_final.iloc[i]
        tomorrow = df_final.iloc[i + 1]

        # Input: all stock features "today" → as single timestep matrix
        stock_matrix = today[stock_cols].values.reshape(1, -1)    # (1, D)
        X_past.append(stock_matrix)

        # Use std of entire train set (same for all days)
        X_std.append(df_final[stock_cols].std().values)           # (D,)

        # Input: macro values (vix, ltv) "today"
        X_all.append(today[macro_cols].values)                    # (F,)

        # Input: weekday
        X_weekday.append([pd.to_datetime(today['date']).weekday()])

        # Target: returns tomorrow for all 10 stocks
        Y.append(tomorrow[ret_cols].values.reshape(-1, 1))        # (10, 1)

    return (
        np.array(X_past),     # shape: (N, 1, D)
        np.array(X_std),      # shape: (N, D)
        np.array(X_all),      # shape: (N, F)
        np.array(X_weekday),  # shape: (N, 1)
        np.array(Y)           # shape: (N, 10, 1)
    )