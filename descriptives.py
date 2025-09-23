import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt


# ---------- column + data utilities ----------

def _extract_df_from_data_dict(data_dict) -> pd.DataFrame:
    """Robustly extract a DataFrame from DataHandler.get_data() output."""
    if isinstance(data_dict, pd.DataFrame):
        df = data_dict
    elif isinstance(data_dict, dict):
        for k in ["full_set", "full_data", "all_data", "data", "df"]:
            if k in data_dict:
                df = data_dict[k]
                break
        else:
            # common fallback: concatenate train/test
            if "train_set" in data_dict and "validation_set" in data_dict:
                df = pd.concat([data_dict["train_set"], data_dict["validation_set"]], ignore_index=True)
            elif "train_set" in data_dict and "test_set" in data_dict:
                df = pd.concat([data_dict["train_set"], data_dict["test_set"]], ignore_index=True)
            else:
                raise KeyError(f"Could not locate a dataset. Keys: {list(data_dict.keys())}")
    else:
        raise TypeError(f"Unexpected get_data() return type: {type(data_dict)}")

    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns are standardized: date, sym_root, ret_crsp."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    # detect symbol column
    symbol_candidates = ["sym_root", "ticker", "TICKER", "symbol"]
    sym_col = next((c for c in symbol_candidates if c in df.columns), None)
    if sym_col is None:
        raise KeyError(f"No symbol column found; looked for {symbol_candidates}. Columns: {df.columns.tolist()}")

    # detect date column
    date_candidates = ["date", "Date", "DATE"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise KeyError(f"No date column found; looked for {date_candidates}. Columns: {df.columns.tolist()}")

    # detect returns column (ret_crsp expected)
    if "ret_crsp" not in df.columns:
        raise KeyError("Expected 'ret_crsp' column not found in data.")

    # standardize names
    if sym_col != "sym_root":
        df.rename(columns={sym_col: "sym_root"}, inplace=True)
    if date_col != "date":
        df.rename(columns={date_col: "date"}, inplace=True)

    # types
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------- fixed color mapping to match LaTeX Table \ref{tab:symbols_desc} ----------
FIXED_SYMBOL_ORDER = ["AAPL","MSFT","JPM","XOM","GE","CAT","BA","PFE","JNJ","MRK"]
FIXED_TAB10_COLORS = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple",
                      "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]

def make_fixed_symbol_colors(symbols: List[str]) -> Dict[str, tuple]:
    base_map = {sym: mcolors.to_rgba(col)
                for sym, col in zip(FIXED_SYMBOL_ORDER, FIXED_TAB10_COLORS)}
    out: Dict[str, tuple] = {}
    for i, s in enumerate(symbols):
        out[s] = base_map.get(s, mcolors.to_rgba(cm.get_cmap("tab20")(i % 20)))
    return out


# --- Descriptive plotting functions ---
def describe_target_ret_crsp(data_handler,
                             start_date: str,
                             end_date: str,
                             rolling_window: int,
                             save_dir: str,
                             show: bool = False,
                             save_png: bool = True,
                             add_legend: bool = True):
    """Descriptive plots for all symbols in dataset."""
    os.makedirs(save_dir, exist_ok=True)
    data_dict = data_handler.get_data(split_point=None)  # full dataset
    df = data_dict["full_set"]
    symbols = df["TICKER"].unique()
    colors = make_fixed_symbol_colors(symbols)

    # --- Daily returns ---
    plt.figure(figsize=(12, 6))
    for sym in symbols:
        sub = df[df["TICKER"] == sym]
        plt.plot(sub.index, sub["ret_crsp"], color=colors[sym], label=sym, lw=0.8)
    plt.title("Daily Returns of DJIA Constituents")
    plt.ylabel("Return")
    if add_legend:
        plt.legend(ncol=5, fontsize=8)
    if save_png:
        plt.savefig(os.path.join(save_dir, "ret_individual_by_symbol.png"), dpi=300)
    if show:
        plt.show()
    plt.close()

    # --- Rolling volatility ---
    plt.figure(figsize=(12, 6))
    for sym in symbols:
        sub = df[df["TICKER"] == sym]
        plt.plot(sub.index, sub["ret_crsp"].rolling(rolling_window).std(),
                 color=colors[sym], label=sym, lw=0.8)
    plt.title(f"Rolling {rolling_window}-day Volatility of Returns")
    plt.ylabel("Std Dev")
    if add_legend:
        plt.legend(ncol=5, fontsize=8)
    if save_png:
        plt.savefig(os.path.join(save_dir, f"rolling_vol_{rolling_window}d.png"), dpi=300)
    if show:
        plt.show()
    plt.close()

    # --- Histogram per symbol ---
    for sym in symbols:
        sub = df[df["TICKER"] == sym]
        plt.figure(figsize=(6, 4))
        plt.hist(sub["ret_crsp"], bins=50, color=colors[sym], alpha=0.7)
        plt.title(f"Histogram of Returns — {sym}")
        if save_png:
            plt.savefig(os.path.join(save_dir, f"hist_{sym}.png"), dpi=300)
        if show:
            plt.show()
        plt.close()

    # --- Boxplot (all symbols) ---
    plt.figure(figsize=(12, 6))
    data_to_plot = [df[df["TICKER"] == sym]["ret_crsp"] for sym in symbols]
    plt.boxplot(data_to_plot, labels=symbols)
    plt.title("Boxplot of Returns by Symbol")
    if save_png:
        plt.savefig(os.path.join(save_dir, "boxplot_returns.png"), dpi=300)
    if show:
        plt.show()
    plt.close()

    # --- QQ-plots (per symbol) ---
    import statsmodels.api as sm
    for sym in symbols:
        sub = df[df["TICKER"] == sym]["ret_crsp"].dropna()
        plt.figure(figsize=(6, 6))
        sm.qqplot(sub, line="s")
        plt.title(f"QQ-Plot of Returns — {sym}")
        if save_png:
            plt.savefig(os.path.join(save_dir, f"qq_{sym}.png"), dpi=300)
        if show:
            plt.show()
        plt.close()

def plot_returns_and_vol_by_sector(
    data_handler,
    symbol_to_industry: Dict[str, str],
    symbols_order: list,
    rolling_window: int,
    save_dir: str,
    show: bool = False,
    save_png: bool = True,
    exclude_pandemic: bool = False,
):
    """
    TWO composite figures (returns & rolling vol).
    Layout: 2 columns × rows = ceil(#sectors / 2).
    Each sector cell contains a vertical stack of subplots (one per symbol in that sector).
    Y-axis scaling is ALWAYS global across ALL subplots in a figure.
    Figure size is FIXED (consistent output).
    Inner subplot heights are UNIFORM across sectors (padding with blank rows).
    """
    import os, math, numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from collections import OrderedDict

    # --- styling ---
    plt.rcParams.update({
        "axes.titlesize": 16,   # sector title
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9
    })

    os.makedirs(save_dir, exist_ok=True)

    # --- load + reshape data ---
    data_dict = data_handler.get_data(exclude_pandemic=exclude_pandemic)
    df = _normalize_columns(_extract_df_from_data_dict(data_dict))

    if symbols_order is not None:
        df = df[df["sym_root"].isin(symbols_order)]

    ret_wide = df.pivot(index="date", columns="sym_root", values="ret_crsp").sort_index()

    symbols = list(ret_wide.columns) if symbols_order is None else list(symbols_order)
    colors = make_fixed_symbol_colors(symbols)

    # --- sector groups (Energy + Financials merged) ---
    MERGE_SECTORS = {"Energy", "Financials"}
    MERGED_NAME = "Energy & Financials"

    sector_groups: Dict[str, list] = OrderedDict()
    for s in symbols:
        sector_full = symbol_to_industry.get(s, "Other")
        sector = sector_full.split(" -- ")[0] if isinstance(sector_full, str) else "Other"
        if sector in MERGE_SECTORS:
            sector = MERGED_NAME
        sector_groups.setdefault(sector, []).append(s)

    sectors = sorted(
        sector_groups.keys(),
        key=lambda sec: (-len(sector_groups[sec]), sec)
    )

    n_sectors = len(sectors)
    ncols = 2
    nrows = max(1, math.ceil(n_sectors / ncols))

    # --- GLOBAL LIMITS (always-global scaling) ---
    gmax_ret = float(np.nanmax(np.abs(ret_wide[symbols].to_numpy()))) if ret_wide.shape[1] else 1.0
    if not np.isfinite(gmax_ret) or gmax_ret == 0:
        gmax_ret = 1.0

    rolls = {}
    gmax_vol = 0.0
    for sym in symbols:
        s = ret_wide.get(sym)
        if s is None:
            continue
        r = s.dropna().rolling(rolling_window).std()
        rolls[sym] = r
        vmax = float(np.nanmax(r.values)) if r.size else 0.0
        if np.isfinite(vmax):
            gmax_vol = max(gmax_vol, vmax)
    if gmax_vol == 0 or not np.isfinite(gmax_vol):
        gmax_vol = 1.0

    # --- plotting helper ---
    def plot_nested(kind: str, fname: str):
        # fixed figure size (same for all outputs)
        fig = plt.figure(figsize=(16, 10))
        outer = GridSpec(nrows, ncols, figure=fig, wspace=0.2, hspace=0.25)

        # compute uniform inner row count across all sectors
        nmax = max(len([s for s in sector_groups[sec] if s in symbols]) for sec in sectors)

        for idx, sector in enumerate(sectors):
            r, c = divmod(idx, ncols)
            cell = outer[r, c]
            syms = [s for s in sector_groups[sector] if s in symbols]
            n = len(syms)
            if n == 0:
                ax = fig.add_subplot(cell); ax.axis("off"); continue

            # same number of inner rows for every sector cell
            inner = GridSpecFromSubplotSpec(nmax, 1, subplot_spec=cell, hspace=0.25)
            shared_x = None
            last_ax_with_data = None

            # sector header
            sec_ax = fig.add_subplot(cell)
            sec_ax.set_title(sector, fontsize=16, fontweight="bold", pad=6)
            sec_ax.axis("off")

            for i_row in range(nmax):
                ax = fig.add_subplot(inner[i_row], sharex=shared_x)
                if shared_x is None:
                    shared_x = ax

                if i_row < n:
                    sym = syms[i_row]
                    if kind == "returns":
                        s = ret_wide[sym].dropna()
                        ax.plot(s.index, s.values, lw=1.05, color=colors.get(sym))
                        ax.set_ylim(-gmax_ret, gmax_ret)
                        ax.set_ylabel("Ret", labelpad=6)
                    else:
                        roll = rolls.get(sym)
                        if roll is not None and not roll.empty:
                            ax.plot(roll.index, roll.values, lw=1.05, color=colors.get(sym))
                            ax.set_ylim(0, gmax_vol)
                            ax.set_ylabel(f"Std({rolling_window}d)", labelpad=6)

                    ax.grid(alpha=0.35)
                    ax.set_title(sym, loc="left", fontsize=11, fontweight="bold", pad=2)
                    last_ax_with_data = ax
                    ax.tick_params(labelbottom=False)  # hide except for last
                else:
                    # blank rows to equalize heights
                    ax.axis("off")

            # put xlabel only on the last subplot with actual data
            if last_ax_with_data is not None:
                last_ax_with_data.tick_params(labelbottom=True)
                last_ax_with_data.set_xlabel("Date", labelpad=6)

        fig.subplots_adjust(left=.06, right=.99, top=.95, bottom=.06)
        if save_png:
            fig.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # --- make both figures ---
    plot_nested("returns", "returns_nested.png")
    plot_nested("vol", "vol_nested.png")






