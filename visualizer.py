# plot_multivariate_forecasts.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= USER INPUTS (edit these) =========
PATH_CGM = "results/CGM/20250818-171546/samples_cgm.npy"
PATH_TS  = "results/TWOSTEP/20250817-202824/samples_two_step.npy"
PATH_CSV = "data_for_kit.csv"  # has columns: date, sym_root, ret_crsp, ...

# This MUST match the order of symbols in axis=1 of your .npy files
symbols_order = ["MSFT","XOM","GE","AAPL","CAT","BA","PFE","JNJ","MRK","JPM"]  # <-- edit

# Which symbols to plot (subset of the above). Example: plot all.
symbols_to_plot = symbols_order  # or e.g. ["MSFT","AAPL","BA"]
# ============================================

# ---- Load samples and compute mean over samples axis ----
samples_cgm = np.load(PATH_CGM)        # shape (n_origins, n_symbols, n_samples)
samples_ts  = np.load(PATH_TS)         # same shape

assert samples_cgm.ndim == 3 and samples_ts.ndim == 3, "Expected 3D arrays"
assert samples_cgm.shape == samples_ts.shape, "Shapes differ between CGM and Two-Step"

n_origins, n_symbols, n_samples = samples_cgm.shape
assert n_symbols == len(symbols_order), "symbols_order length must equal axis=1 size"

# Mean forecast for each origin & symbol
mean_cgm = samples_cgm.mean(axis=2)    # (n_origins, n_symbols)
mean_ts  = samples_ts.mean(axis=2)     # (n_origins, n_symbols)

# Wrap in DataFrames (no dates yet)
df_mean_cgm = pd.DataFrame(mean_cgm, columns=symbols_order)
df_mean_ts  = pd.DataFrame(mean_ts,  columns=symbols_order)

# ---- Load realized returns and align dates ----
df_all = pd.read_csv(PATH_CSV, parse_dates=["date"])

# Keep only needed symbols & columns
df_all = df_all.loc[df_all["sym_root"].isin(symbols_order), ["date","sym_root","ret_crsp"]]

# We want the last n_origins dates *common to all symbols* (so panel is rectangular)
common_dates = (
    df_all.groupby("sym_root")["date"]
          .apply(lambda s: set(s.sort_values().unique()))
          .agg(lambda sets: set.intersection(*sets))  # intersection across symbols
)
common_dates = sorted(common_dates)
if len(common_dates) < n_origins:
    raise ValueError(f"Not enough common dates: have {len(common_dates)}, need {n_origins}")

test_dates = common_dates[-n_origins:]   # align with your rolling forecast period

# Pivot realized to (date x symbol) and slice to test period
df_real = (
    df_all[df_all["date"].isin(test_dates)]
      .pivot(index="date", columns="sym_root", values="ret_crsp")
      .sort_index()
)

# Sanity: ensure correct shape and column order
df_real = df_real[symbols_order]              # reorder columns
assert df_real.shape == (n_origins, n_symbols)
df_mean_cgm.index = df_real.index             # add dates as index
df_mean_ts.index  = df_real.index

# ---- Plot: time series per selected symbol ----
for sym in symbols_to_plot:
    plt.figure(figsize=(9, 5))
    plt.plot(df_mean_cgm.index, df_mean_cgm[sym], linewidth=2, label="CGM avg forecast")
    plt.plot(df_mean_ts.index,  df_mean_ts[sym],  linewidth=2, label="Two-Step avg forecast")
    plt.plot(df_real.index,     df_real[sym],     linewidth=2, linestyle="--", label="Realized")
    plt.title(f"{sym} — Avg 1-step Forecast vs Realized")
    plt.xlabel("Date / forecast origin")
    plt.ylabel("ret_crsp")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---- Optional: quick panel view on one figure for multiple symbols ----
# Uncomment to see small multiples (rows = symbols)
# import math
# k = len(symbols_to_plot)
# cols = 2
# rows = math.ceil(k / cols)
# plt.figure(figsize=(12, 3.2*rows))
# for i, sym in enumerate(symbols_to_plot, 1):
#     ax = plt.subplot(rows, cols, i)
#     ax.plot(df_mean_cgm.index, df_mean_cgm[sym], linewidth=2, label="CGM")
#     ax.plot(df_mean_ts.index,  df_mean_ts[sym],  linewidth=2, label="Two-Step")
#     ax.plot(df_real.index,     df_real[sym],     linewidth=2, linestyle="--", label="Realized")
#     ax.set_title(sym)
#     ax.grid(alpha=0.3)
#     if i == 1:
#         ax.legend()
# plt.tight_layout()
# plt.show()