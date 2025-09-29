import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import matplotlib.ticker as mticker


def _savefig_both(fig, save_dir: str, stem: str, png_dpi: int = 300, pdf_bbox: str = "tight"):
    """
    Save a figure as both PDF (vector) and PNG.
    PDF: for LaTeX (crisp, scalable)
    PNG: quick browsing/preview
    """
    base = os.path.join(save_dir, stem)
    # PDF (vector)
    fig.savefig(base + ".pdf", bbox_inches=pdf_bbox)
    # PNG (bitmap)
    fig.savefig(base + ".png", dpi=png_dpi, bbox_inches="tight")

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
FIXED_SYMBOL_COLORS = {
    "AAPL": mcolors.to_rgba("tab:blue"),
    "MSFT": mcolors.to_rgba("tab:orange"),
    "JPM":  mcolors.to_rgba("tab:green"),
    "XOM":  mcolors.to_rgba("tab:red"),
    "GE":   mcolors.to_rgba("tab:purple"),
    "CAT":  mcolors.to_rgba("tab:brown"),
    "BA":   mcolors.to_rgba("tab:pink"),
    "PFE":  mcolors.to_rgba("tab:gray"),
    "JNJ":  mcolors.to_rgba("tab:olive"),
    "MRK":  mcolors.to_rgba("tab:cyan"),
}

def make_fixed_symbol_colors(symbols: list[str]) -> dict[str, tuple]:
    out = {}
    fallback = cm.get_cmap("tab20")
    k = 0
    for s in symbols:
        if s in FIXED_SYMBOL_COLORS:
            out[s] = FIXED_SYMBOL_COLORS[s]
        else:
            out[s] = fallback(k % 20)
            k += 1
    return out


def describe_target_ret_crsp(data_handler,
                             rolling_window: int,
                             save_dir: str,
                             show: bool = False,
                             save_png: bool = True,
                             add_legend: bool = True):
    """Descriptive plots for all symbols in dataset (incl. VIX scatter), using FIXED_SYMBOL_ORDER colors."""
    import os, numpy as np, pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from scipy import stats as sps

    os.makedirs(save_dir, exist_ok=True)

    # ---------------- helpers ----------------
    def _prep_df(handler) -> pd.DataFrame:
        raw = handler.get_data()
        df_ = _normalize_columns(_extract_df_from_data_dict(raw))
        return df_.sort_values("date").set_index("date")

    def _find_vix_col(df_) -> str | None:
        cand = ["vix_close", "VIX_Close", "vix", "VIX", "vix_cls", "vixclose"]
        low = {c.lower(): c for c in df_.columns}
        for c in cand:
            if c in low:
                return low[c]
        for c in df_.columns:
            if "vix" in str(c).lower():
                return c
        return None

    def _rc():
        return plt.rc_context({
            "figure.dpi": 1200,
            "savefig.dpi": 1200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 20,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        })

    def _style(ax):
        ax.grid(alpha=0.25)
        ax.tick_params(length=2)

    # ---------------- data & colors ----------------
    df = _prep_df(data_handler)
    symbols = df["sym_root"].unique().tolist()
    colors = make_fixed_symbol_colors(symbols)  # <- uses FIXED_SYMBOL_ORDER

    with _rc():

        # --- Daily returns ---
        fig, ax = plt.subplots(figsize=(17, 8))
        for s in symbols:
            sub = df.loc[df["sym_root"] == s, "ret_crsp"]
            ax.plot(sub.index, sub.values, lw=0.9, label=s, color=colors[s])
        ax.set_xlabel("Date"); ax.set_ylabel("ret_crsp")
        _style(ax)
        if add_legend: ax.legend(ncol=5, frameon=False, handleheight=1.5)
        if save_png: _savefig_both(fig, save_dir, "ret_individual_by_symbol")
        if show: plt.show()
        plt.close(fig)

        # --- Rolling volatility ---
        fig, ax = plt.subplots(figsize=(17, 8))
        for s in symbols:
            sub = df.loc[df["sym_root"] == s, "ret_crsp"].dropna()
            roll = sub.rolling(rolling_window).std()
            ax.plot(roll.index, roll.values, lw=1.0, label=s, color=colors[s])
        ax.set_xlabel("Date"); ax.set_ylabel(f"Rolling Std ({rolling_window}d)")
        _style(ax)
        if add_legend: ax.legend(ncol=5, frameon=False, handleheight=1.5)
        if save_png: _savefig_both(fig, save_dir, f"rolling_vol_{rolling_window}d")
        if show: plt.show()
        plt.close(fig)

        # --- Histograms ---
        n = len(symbols)
        ncols = 5
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.0, nrows * 3.0),
                                 constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)

        bins = 40
        global_min = float(df["ret_crsp"].min())
        global_max_val = float(df["ret_crsp"].max())
        bin_edges = np.linspace(global_min, global_max_val, bins + 1)

        # global max count across assets (on raw counts)
        global_max_count = 0
        per_symbol_counts = {}  # cache (counts, bin_edges) per symbol
        for s in symbols:
            sub = df.loc[df["sym_root"] == s, "ret_crsp"].dropna().to_numpy()
            counts, _ = np.histogram(sub, bins=bin_edges)
            per_symbol_counts[s] = counts
            if counts.size:
                global_max_count = max(global_max_count, counts.max())

        squash_factor = 0.5  # 0.5 = half as tall; tweak as you like

        for ax, s in zip(axes, symbols):
            counts = per_symbol_counts.get(s)
            if counts is None or counts.size == 0:
                ax.axis("off")
                continue

            # draw bars with SCALED heights (so nothing clips)
            ax.bar(
                bin_edges[:-1],
                counts * squash_factor,
                width=np.diff(bin_edges),
                align="edge",
                alpha=0.9,
                color=colors[s],
                edgecolor="none",
            )

            # shared y-limit based on scaled global max
            ax.set_ylim(0, global_max_count * squash_factor)

            # show TRUE counts on the ticks (undo the scaling in labels)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, pos: f"{int(round(v / squash_factor))}" if v > 0 else "0")
            )

            ax.set_title(s, pad=2)
            _style(ax)

        for ax in axes[len(symbols):]:
            ax.axis("off")

        if save_png:
            _savefig_both(fig, save_dir, "histograms_all_assets")
        if show:
            plt.show()
        plt.close(fig)


        # --- QQ-plots ---
        n = len(symbols)
        ncols = 5
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.0, nrows * 3.0),
                                 constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)

        y_cap_pct = None  # percentile cap for shared y-limit (None to disable)

        qq_data = {}
        global_x_max = 0.0  # theoretical quantiles (≈3 for N(0,1))
        global_y_max = 0.0  # empirical ordered values only

        for s in symbols:
            x = df.loc[df["sym_root"] == s, "ret_crsp"].dropna().to_numpy()
            if x.size == 0:
                qq_data[s] = (None, None)
                continue

            x_sorted = np.sort(x)
            n_s = x_sorted.size
            probs = (np.arange(1, n_s + 1) - 0.5) / n_s
            z_theory = sps.norm.ppf(probs)

            qq_data[s] = (z_theory, x_sorted)

            # global X range
            local_x = float(np.nanmax(np.abs(z_theory)))
            if np.isfinite(local_x):
                global_x_max = max(global_x_max, local_x)

            # global Y range
            if y_cap_pct is not None:
                local_y = float(np.nanpercentile(np.abs(x_sorted), y_cap_pct))
            else:
                local_y = float(np.nanmax(np.abs(x_sorted)))
            if np.isfinite(local_y):
                global_y_max = max(global_y_max, local_y)

        # fallbacks
        if not np.isfinite(global_x_max) or global_x_max == 0:
            global_x_max = 3.0
        if not np.isfinite(global_y_max) or global_y_max == 0:
            global_y_max = 1.0

        xlim = global_x_max * 1.05
        ylim = global_y_max * 1.05

        for ax, s in zip(axes, symbols):
            z_theory, x_sorted = qq_data[s]
            if z_theory is None:
                ax.axis("off")
                continue

            # fitted line (least-squares regression of data on theory)
            slope, intercept = np.polyfit(z_theory, x_sorted, 1)
            ax.plot([-xlim, xlim],
                    [slope * (-xlim) + intercept, slope * xlim + intercept],
                    color="black", lw=1.0, alpha=0.9, zorder=1)

            # scatter points
            ax.scatter(z_theory, x_sorted, s=9, alpha=0.9, edgecolors="none",
                       facecolors=colors[s], zorder=2)

            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            ax.set_title(s, pad=2)
            ax.set_xlabel("Theoretical quantiles", fontsize=12)
            ax.set_ylabel("Ordered Values", fontsize=12)
            _style(ax)

        for ax in axes[len(symbols):]:
            ax.axis("off")

        if save_png:
            _savefig_both(fig, save_dir, "qqplots_all_assets")
        if show:
            plt.show()
        plt.close(fig)

        # --- Correlation heatmap ---
        wide = df.pivot_table(index=df.index, columns="sym_root", values="ret_crsp")
        corr = wide.corr()
        lab = corr.columns.tolist();
        mat = corr.values

        fig, ax = plt.subplots(figsize=(12, 6))  # less tall
        im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto", cmap="Blues")

        ax.set_xticks(np.arange(len(lab)));
        ax.set_yticks(np.arange(len(lab)))
        ax.set_xticklabels(lab, rotation=45, ha="right");
        ax.set_yticklabels(lab)

        _style(ax)

        # bigger, bold correlation values (black, except diagonal white)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=14,
                    fontweight="bold",
                    color=("white" if i == j and val == 1.0 else "black"),
                )

        fig.tight_layout()
        if save_png:
            _savefig_both(fig, save_dir, "corr_heatmap")
        if show:
            plt.show()
        plt.close(fig)

        # --- Boxplots (all vs. excl. pandemic) ---
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        covid_start = pd.to_datetime("2020-03-01")
        pre_covid_df = df.loc[df.index < covid_start]

        # data per symbol
        all_data_by_sym = {s: df.loc[df["sym_root"] == s, "ret_crsp"].dropna().values for s in symbols}
        pre_data_by_sym = {s: pre_covid_df.loc[pre_covid_df["sym_root"] == s, "ret_crsp"].dropna().values for s in
                           symbols}

        # build figure
        fig, ax = plt.subplots(figsize=(14, 6.5))

        # spacing along x
        xs = np.arange(len(symbols), dtype=float)
        width = 0.30
        pos_all = xs - width / 2
        pos_pre = xs + width / 2

        # helper to draw a styled boxplot at given positions
        def draw_boxplot(datalist, positions, hatch=False):
            bp = ax.boxplot(
                datalist,
                positions=positions,
                widths=width * 0.95,
                patch_artist=True,
                showmeans=False,
                showfliers=False,  # clean look; whiskers reflect IQR rule
                whis=1.5
            )
            # style
            for i, box in enumerate(bp["boxes"]):
                s = symbols[i]
                col = colors[s]
                box.set_facecolor(col)
                box.set_alpha(0.6)
                box.set_edgecolor("#444444")
                if hatch:
                    box.set_hatch("//")
                # whiskers/caps/medians
            for w in bp["whiskers"] + bp["caps"]:
                w.set_color("#444444")
            for med in bp["medians"]:
                med.set_color("#222222");
                med.set_linewidth(1.3)
            return bp

        # draw both sets
        all_list = [all_data_by_sym[s] for s in symbols]
        pre_list = [pre_data_by_sym[s] for s in symbols]
        bp_all = draw_boxplot(all_list, pos_all, hatch=False)
        bp_pre = draw_boxplot(pre_list, pos_pre, hatch=True)

        # mean markers (white diamond) for each box
        means_all = [np.nanmean(v) if len(v) else np.nan for v in all_list]
        means_pre = [np.nanmean(v) if len(v) else np.nan for v in pre_list]
        ax.scatter(pos_all, means_all, marker="D", s=36, facecolors="white", edgecolors="#222222", zorder=3)
        ax.scatter(pos_pre, means_pre, marker="D", s=36, facecolors="white", edgecolors="#222222", zorder=3)

        # zero reference
        ax.axhline(0, color="#666666", lw=1.0, ls="--", alpha=0.8)

        # x axis
        ax.set_xticks(xs)
        ax.set_xticklabels(symbols, rotation=0)
        ax.set_xlim(xs[0] - 0.75, xs[-1] + 0.75)
        ax.set_xlabel("Symbol")
        ax.set_ylabel("ret_crsp")
        _style(ax)

        # optional: harmonize y-range using whisker extents (ignores outliers we hid)
        ymin, ymax = np.inf, -np.inf
        for arr in all_list + pre_list:
            if len(arr):
                q1, q3 = np.nanpercentile(arr, [25, 75])
                iqr = q3 - q1
                lo = q1 - 1.5 * iqr
                hi = q3 + 1.5 * iqr
                ymin = min(ymin, lo)
                ymax = max(ymax, hi)
        if np.isfinite(ymin) and np.isfinite(ymax):
            pad = 0.05 * (ymax - ymin + 1e-12)
            ax.set_ylim(ymin - pad, ymax + pad)

        # legend (generic patches so colors don't matter)
        leg_all = mpatches.Patch(facecolor="0.7", edgecolor="#444444", label="All data")
        leg_pre = mpatches.Patch(facecolor="0.7", edgecolor="#444444", hatch="//",
                                 label="Excl. pandemic (Mar 2020-)")
        leg_mean = Line2D([0], [0], marker='D', color='none', markerfacecolor='white',
                          markeredgecolor='#222222', markersize=7, label="Mean")
        ax.legend(handles=[leg_all, leg_pre, leg_mean], frameon=True, loc="upper right", fontsize=15)

        fig.tight_layout()
        if save_png:
            _savefig_both(fig, save_dir, "boxplot_by_symbol")
        if show:
            plt.show()
        plt.close(fig)

        # --- Scatter returns vs VIX ---
        vix_col = _find_vix_col(df)
        if vix_col is not None:
            fig, ax = plt.subplots(figsize=(11, 8))

            pandemic_start = pd.to_datetime("2020-03-01")
            pandemic_end   = pd.to_datetime("2020-06-30")

            for s in symbols:
                sub = df.loc[df["sym_root"] == s, [vix_col, "ret_crsp"]].dropna()

                # 1) All data (base layer)
                ax.scatter(
                    sub[vix_col], sub["ret_crsp"],
                    s=12, alpha=0.35, color=colors[s], edgecolors="none", label=None
                )

                # 2) Pandemic
                sub_pand = sub.loc[(sub.index >= pandemic_start) & (sub.index <= pandemic_end)]
                if not sub_pand.empty:
                    ax.scatter(
                        sub_pand[vix_col], sub_pand["ret_crsp"],
                        s=30, alpha=0.8, color=colors[s],
                        edgecolors="red", linewidths=1.0, zorder=3, label=None
                    )

            # legend entry for pandemic
            import matplotlib.patches as mpatches
            leg_pand = mpatches.Patch(facecolor="white", edgecolor="red", label="Pandemic (Mar–Jun 2020)")

            # dynamic axis label from column name
            x_label = vix_col.replace("_", " ").title()  # e.g. "vix_close" -> "Vix Close"
            ax.set_xlabel(x_label)
            ax.set_ylabel("ret_crsp")
            _style(ax)

            if add_legend:
                ax.legend(handles=[leg_pand], frameon=False, loc="upper right")

            fig.tight_layout()
            _savefig_both(fig, save_dir, "scatter_ret_vs_vix")
            plt.close(fig)





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






