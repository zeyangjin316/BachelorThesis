# plot_multivariate_forecasts.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.data_handling import DataHandler

import os
os.makedirs("results/plots", exist_ok=True)

# ========= USER INPUTS (edit these) =========
PATH_CGM = "results/CGM/20250824-142246/samples_cgm.npy"
PATH_TS = "results/COMPARISON/20250819-232955/samples_two_step.npy"

# This MUST match the order of symbols in axis=1 of your .npy files
symbols_order = ["MSFT", "XOM", "GE", "AAPL", "CAT", "BA", "PFE", "JNJ", "MRK", "JPM"]  # <-- edit

# Which symbols to plot (subset of the above). Example: plot all.
symbols_to_plot = symbols_order  # or e.g. ["MSFT","AAPL","BA"]

# Which sample to visualize (0-indexed)
sample_to_plot = 0  # <-- edit this to select which sample (0 to n_samples-1)
# ============================================

# ---- Load samples ----
samples_cgm = np.load(PATH_CGM)  # shape (n_origins, n_symbols, n_samples)
samples_ts = np.load(PATH_TS)  # same shape

assert samples_cgm.ndim == 3 and samples_ts.ndim == 3, "Expected 3D arrays"
assert samples_cgm.shape == samples_ts.shape, "Shapes must match if both start at same date"

n_origins, n_symbols, n_samples = samples_cgm.shape
assert n_symbols == len(symbols_order), "symbols_order length must equal axis=1 size"

# Select one specific sample for each origin & symbol
sample_cgm = samples_cgm[:, :, sample_to_plot]  # (n_origins, n_symbols)
sample_ts = samples_ts[:, :, sample_to_plot]  # (n_origins, n_symbols)

print(f"Samples shape: {sample_cgm.shape}")

# ---- Load data using DataHandler (same configuration as experiments) ----
data_handler = DataHandler(split_point=0.9)  # Same split as experiments
data_dict = data_handler.get_data(standardize=False, filter_features=True, exclude_pandemic=True)

# Get test set (this contains the dates that both models used)
test_set = data_dict['test_set']
test_set = test_set[test_set['sym_root'].isin(symbols_order)]

# Get the first n_origins test dates (both models start at same date)
test_dates = sorted(test_set['date'].unique())[:n_origins]

print(f"Sample date range:")
print(f"First date: {test_dates[0]}")
print(f"Last date: {test_dates[-1]}")
print(f"Total days: {len(test_dates)}")

# ---- Create sample DataFrames with the correct dates ----
df_sample_cgm = pd.DataFrame(sample_cgm, index=test_dates, columns=symbols_order)
df_sample_ts = pd.DataFrame(sample_ts, index=test_dates, columns=symbols_order)

# ---- Get realized returns for the sample dates ----
real_data = test_set[test_set['date'].isin(test_dates)]
df_real = real_data.pivot(index='date', columns='sym_root', values='ret_crsp')
df_real = df_real.reindex(test_dates).reindex(columns=symbols_order)

print(f"Real data shape: {df_real.shape}")

# ---- Plot: time series per selected symbol ----
for sym in symbols_to_plot:
    plt.figure(figsize=(10, 6))

    plt.plot(df_sample_cgm.index, df_sample_cgm[sym], linewidth=2,
             label=f"CGM sample {sample_to_plot}")
    plt.plot(df_sample_ts.index, df_sample_ts[sym], linewidth=2, color="green",
             label=f"Two-Step sample {sample_to_plot}")
    plt.plot(df_real.index, df_real[sym], linewidth=2, color="red",
             label="Realized")

    # dynamic y-axis range with margin
    all_vals = pd.concat([df_sample_cgm[sym], df_sample_ts[sym], df_real[sym]])
    y_min, y_max = all_vals.min(), all_vals.max()
    y_range = y_max - y_min
    plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.title(f"{sym} — Forecast vs Realized")
    plt.xlabel("Date")
    plt.ylabel("ret_crsp")
    #plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # save to results/plots/
    filepath = f"results/plots/{sym}_sample{sample_to_plot}.png"
    plt.savefig(filepath, dpi=300)  # high-res PNG
    # plt.savefig(filepath.replace(".png", ".pdf"))  # optional PDF version

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
#     ax.plot(df_sample_cgm.index, df_sample_cgm[sym], linewidth=2, label="CGM")
#     ax.plot(df_sample_ts.index, df_sample_ts[sym], linewidth=2, label="Two-Step")
#     ax.plot(df_real.index, df_real[sym], linewidth=2, linestyle="--", label="Realized")
#     ax.set_title(sym)
#     ax.grid(alpha=0.3)
#     if i == 1:
#         ax.legend()
# plt.tight_layout()
# plt.show()

