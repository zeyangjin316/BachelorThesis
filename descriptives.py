import os
from typing import Iterable, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from data.data_handling import DataHandler

import matplotlib.cm as cm
import matplotlib.colors as mcolors


def _make_symbol_colors(symbols):
    if len(symbols) <= 10:
        cmap = cm.get_cmap("tab10", len(symbols))
    elif len(symbols) <= 20:
        cmap = cm.get_cmap("tab20", len(symbols))
    else:
        cmap = cm.get_cmap("hsv", len(symbols))
    return {s: mcolors.to_rgba(cmap(i)) for i, s in enumerate(symbols)}



def describe_target_ret_crsp(
        data_handler: DataHandler,
        *,
        symbols_subset: Optional[Iterable[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rolling_window: int = 30,
        save_dir: str = "figures",  # main figures folder
        csv_dir: str = "results/descriptives",  # CSV folder
        filter_features: bool = True,
        exclude_pandemic: bool = False,
        show: bool = True,
        save_png: bool = True,
        add_legend: bool = True,
) -> Dict[str, str]:
    """
    Create descriptive statistics and plots focused on the target `ret_crsp`.
    Saves all outputs to `save_dir` and summary statistics as CSV in `csv_dir`.
    """
    import shutil
    import math
    import scipy.stats as stats
    import seaborn as sns

    # Always overwrite past results
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir, exist_ok=True)

    # ---- Load dataset using DataHandler ----
    data_dict = data_handler.get_data(
        filter_duplicates=filter_features,
        exclude_pandemic=exclude_pandemic,
    )
    if "full_set" in data_dict:
        data = data_dict["full_set"]
    elif "full_data" in data_dict:
        data = data_dict["full_data"]
    elif "all_data" in data_dict:
        data = data_dict["all_data"]
    elif "data" in data_dict:
        data = data_dict["data"]
    elif "df" in data_dict:
        data = data_dict["df"]
    elif "train_set" in data_dict and "validation_set" in data_dict:
        data = pd.concat([data_dict["train_set"], data_dict["validation_set"]], ignore_index=True)
    elif "train_set" in data_dict and "test_set" in data_dict:
        data = pd.concat([data_dict["train_set"], data_dict["test_set"]], ignore_index=True)
    else:
        raise KeyError(f"Could not locate a dataset in DataHandler.get_data() output. "
                       f"Available keys: {list(data_dict.keys())}.")

    if isinstance(data, pd.Series):
        data = data.to_frame()
    elif not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame for `data`, got {type(data)}")

    # Normalize column names
    data = data.copy()
    data.columns = [c.replace(" ", "_") for c in data.columns]

    # Add alias for VIX if needed
    if "vix_close" not in data.columns:
        for alt in ["vixclose", "vix", "vix_Close"]:
            if alt in data.columns:
                data["vix_close"] = data[alt]
                break

    # Filter dates
    if start_date is not None:
        data = data[data["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        data = data[data["date"] <= pd.to_datetime(end_date)]

    # Filter symbols if requested
    if symbols_subset is not None:
        symbols_subset = list(symbols_subset)
        data = data[data["sym_root"].isin(symbols_subset)]

    # Pivot to (date x symbol)
    ret_wide = data.pivot(index="date", columns="sym_root", values="ret_crsp").sort_index()
    symbols = list(ret_wide.columns if symbols_subset is None else symbols_subset)

    outputs: Dict[str, str] = {}

    symbol_colors = _make_symbol_colors(symbols)

    # ---- 1) Per-symbol summary stats ----
    def _sharpe_like(x: pd.Series) -> float:
        s = x.std()
        return float(x.mean() / s) if s and not np.isnan(s) else np.nan

    summary = pd.DataFrame({
        "mean": ret_wide[symbols].mean(skipna=True),
        "median": ret_wide[symbols].median(skipna=True),
        "std": ret_wide[symbols].std(skipna=True),
        "skew": ret_wide[symbols].skew(skipna=True),
        "kurt": ret_wide[symbols].kurtosis(skipna=True),
        "min": ret_wide[symbols].min(skipna=True),
        "max": ret_wide[symbols].max(skipna=True),
        "count": ret_wide[symbols].count(),
        "sharpe_like": ret_wide[symbols].apply(_sharpe_like, axis=0),
    }).sort_index()

    # Save as CSV
    csv_path = os.path.join(csv_dir, "ret_crsp_summary_by_symbol.csv")
    summary.to_csv(csv_path)
    outputs["summary_csv"] = csv_path

    # Render summary as PNG table
    summary_to_show = summary.round({
        "mean": 4, "median": 4, "std": 4, "skew": 3, "kurt": 2,
        "min": 3, "max": 3, "sharpe_like": 2
    })
    n_rows = len(summary_to_show)
    row_height = 0.35
    fig_height = max(2.5, min(11.0, n_rows * row_height))
    fig_width = 11.7
    plt.figure(figsize=(fig_width, fig_height))
    plt.axis("off")
    tbl = plt.table(
        cellText=summary_to_show.values,
        rowLabels=summary_to_show.index.tolist(),
        colLabels=summary_to_show.columns.tolist(),
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    table_png_path = os.path.join(save_dir, "summary_stats_returns.png")
    plt.tight_layout(pad=0.5)
    if save_png:
        plt.savefig(table_png_path, dpi=300, bbox_inches="tight")
        outputs["summary_table_png"] = table_png_path
    if show:
        plt.show()
    else:
        plt.close()

    # ---- 2) Individual returns by symbol ----
    plt.figure(figsize=(12, 7))
    for s in symbols:
        series = ret_wide[s]
        plt.plot(series.index, series.values, linewidth=1.2, label=s, color=symbol_colors[s])

    plt.xlabel("Date")
    plt.ylabel("ret_crsp")
    if add_legend:
        plt.legend(ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_png:
        p = os.path.join(save_dir, "ret_individual_by_symbol.png")
        plt.savefig(p, dpi=300)
        outputs["ret_individual_png"] = p
    if show:
        plt.show()
    else:
        plt.close()

    # ---- 3) Rolling volatility ----
    plt.figure(figsize=(12, 7))
    for s in symbols:
        roll_vol = ret_wide[s].rolling(rolling_window).std()
        plt.plot(
            roll_vol.index, roll_vol.values,
            linewidth=1.2, label=f"{s}", color=symbol_colors[s]
        )
    plt.xlabel("Date")
    plt.ylabel(f"Rolling Std ({rolling_window}d)")
    if add_legend:
        plt.legend(ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_png:
        p = os.path.join(save_dir, f"rolling_vol_{rolling_window}d.png")
        plt.savefig(p, dpi=300)
        outputs["rolling_vol_png"] = p
    if show:
        plt.show()
    else:
        plt.close()

    # ---- 4) Histograms ----
    n_assets = len(symbols)
    ncols = 5
    nrows = math.ceil(n_assets / ncols)

    # compute global y-limit across all histograms
    max_y = 0
    for s in symbols:
        x = ret_wide[s].dropna()
        counts, _ = np.histogram(x, bins=40, density=True)
        if counts.size:
            max_y = max(max_y, counts.max())

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6), constrained_layout=True)  # wider, less tall
    axes = axes.flatten()
    for i, s in enumerate(symbols):
        ax = axes[i]
        x = ret_wide[s].dropna()
        ax.hist(x, bins=40, density=True, color=symbol_colors[s], alpha=0.75)
        ax.set_title(s, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if max_y > 0:
            ax.set_ylim(0, max_y * 1.05)  # uniform y scale
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if save_png:
        p = os.path.join(save_dir, "histograms_all_assets.png")
        plt.savefig(p, dpi=300)
        outputs["histograms_all_assets"] = p
    if show:
        plt.show()
    else:
        plt.close(fig)

    # ---- 5) QQ-Plots ----
    ncols = 5
    nrows = math.ceil(n_assets / ncols)

    # compute global y limits for QQ plots
    qq_ymin, qq_ymax = float("inf"), -float("inf")
    for s in symbols:
        data_s = ret_wide[s].dropna().values
        if data_s.size == 0:
            continue
        # probplot with fit=False gives (osm, osr)
        _, osr = stats.probplot(data_s, dist="norm", fit=False)
        qq_ymin = min(qq_ymin, np.min(osr))
        qq_ymax = max(qq_ymax, np.max(osr))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6), constrained_layout=True)  # wider, less tall
    axes = axes.flatten()
    for i, s in enumerate(symbols):
        ax = axes[i]
        data_s = ret_wide[s].dropna().values
        if data_s.size:
            stats.probplot(data_s, dist="norm", plot=ax)  # draws points + fit line
            lines = ax.get_lines()
            if lines:
                lines[0].set_color(symbol_colors[s])  # data points
                if len(lines) > 1:
                    lines[1].set_color("black")  # fit line
        ax.set_title(s, fontsize=9)
        ax.tick_params(labelsize=7)
        if np.isfinite(qq_ymin) and np.isfinite(qq_ymax) and qq_ymin < qq_ymax:
            ax.set_ylim(qq_ymin, qq_ymax)  # uniform y scale
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if save_png:
        p = os.path.join(save_dir, "qqplots_all_assets.png")
        plt.savefig(p, dpi=300)
        outputs["qqplots_all_assets"] = p
    if show:
        plt.show()
    else:
        plt.close(fig)

    # ---- 6) Boxplot (compare with vs. without COVID period) ----
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    # Build full-sample and "excl. pandemic" datasets from the same ret_wide
    pandemic_start = pd.to_datetime("2020-03-01")
    pandemic_end = pd.to_datetime("2020-06-30")
    mask_covid = (ret_wide.index >= pandemic_start) & (ret_wide.index <= pandemic_end)

    ret_wide_excl = ret_wide.loc[~mask_covid]

    # Prepare data per symbol (drop NAs)
    data_full = [ret_wide[s].dropna().values for s in symbols]
    data_excl = [ret_wide_excl[s].dropna().values for s in symbols]

    # Also compute means (to show that the mean ≠ 0)
    means_full = [np.nanmean(a) if len(a) else np.nan for a in data_full]
    means_excl = [np.nanmean(a) if len(a) else np.nan for a in data_excl]

    plt.figure(figsize=(12, 6))

    # positions for side-by-side boxes
    x = np.arange(1, len(symbols) + 1, dtype=float)
    offset = 0.18
    w = 0.32

    bp_full = plt.boxplot(
        data_full, positions=x - offset, widths=w, vert=True,
        showfliers=False, patch_artist=True
    )
    bp_excl = plt.boxplot(
        data_excl, positions=x + offset, widths=w, vert=True,
        showfliers=False, patch_artist=True
    )

    # color each box with its symbol color; use alpha/hatch to distinguish groups
    for patch, s in zip(bp_full["boxes"], symbols):
        patch.set_facecolor(symbol_colors[s])
        patch.set_alpha(0.45)  # all data: lighter fill
        patch.set_edgecolor("black")

    for patch, s in zip(bp_excl["boxes"], symbols):
        patch.set_facecolor(symbol_colors[s])
        patch.set_alpha(0.9)  # excl. pandemic: stronger fill
        patch.set_hatch("//")  # print-friendly distinction
        patch.set_edgecolor("black")

    # style other box elements for visibility
    for med in bp_full["medians"] + bp_excl["medians"]:
        med.set_color("black")
    for whisk in bp_full["whiskers"] + bp_excl["whiskers"]:
        whisk.set_color("gray")
    for cap in bp_full["caps"] + bp_excl["caps"]:
        cap.set_color("gray")

    # --- NEW: overlay per-group mean markers (diamonds), colored by symbol
    for xi, s, m_full, m_excl in zip(x, symbols, means_full, means_excl):
        if np.isfinite(m_full):
            plt.scatter(
                xi - offset, m_full, marker="D", s=42,
                facecolors=symbol_colors[s], edgecolors="black", linewidths=0.6, zorder=3
            )
        if np.isfinite(m_excl):
            plt.scatter(
                xi + offset, m_excl, marker="D", s=42,
                facecolors=symbol_colors[s], edgecolors="black", linewidths=0.6, zorder=3
            )

    # --- NEW: horizontal zero line for quick reference
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)

    # axes/labels/ticks
    plt.xlabel("Symbol")
    plt.ylabel("ret_crsp")
    plt.xticks(x, symbols, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # legend (sample-independent)
    legend_handles = [
        mpatches.Patch(facecolor="gray", alpha=0.45, edgecolor="black", label="All data"),
        mpatches.Patch(facecolor="gray", alpha=0.9, hatch="//", edgecolor="black",
                       label="Excl. pandemic (Mar–Jun 2020)"),
        mlines.Line2D([], [], color="black", marker="D", linestyle="None", markersize=6,
                      markerfacecolor="white", markeredgecolor="black", label="Mean"),
    ]
    plt.legend(handles=legend_handles, loc="best", frameon=True, framealpha=0.9)

    # keep the SAME filename and outputs key as before
    if save_png:
        p = os.path.join(save_dir, "boxplot_by_symbol.png")
        plt.savefig(p, dpi=300)
        outputs["boxplot_png"] = p
    if show:
        plt.show()
    else:
        plt.close()

    # ---- 7) Correlation heatmap ----
    corr = ret_wide[symbols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f",
                cmap="YlGnBu", center=0,
                xticklabels=corr.columns, yticklabels=corr.columns,
                cbar=False)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    if save_png:
        p = os.path.join(save_dir, "corr_heatmap.png")
        plt.savefig(p, dpi=300)
        outputs["corr_heatmap_png"] = p
    if show:
        plt.show()
    else:
        plt.close()

    # ---- 8) Scatter with VIX (pandemic highlighted, per symbol) ----
    if isinstance(data, pd.DataFrame) and "vix_close" in data.columns:
        vix_by_day = data.pivot_table(index="date", values="vix_close", aggfunc="mean").loc[ret_wide.index]

        # Define pandemic window
        pandemic_start = pd.to_datetime("2020-03-01")
        pandemic_end = pd.to_datetime("2020-06-30")
        mask_pandemic = (vix_by_day.index >= pandemic_start) & (vix_by_day.index <= pandemic_end)

        plt.figure(figsize=(9, 7))

        for s in symbols:
            y = ret_wide[s].reindex(vix_by_day.index)

            # Scatter all points in the symbol’s color
            plt.scatter(vix_by_day.values.flatten(), y.values,
                        alpha=0.6, s=20, color=symbol_colors[s], label=s)

            # Overlay pandemic points with red circles (no fill)
            plt.scatter(vix_by_day.loc[mask_pandemic].values.flatten(),
                        y.loc[mask_pandemic].values,
                        facecolors="none", edgecolors="red", linewidths=1.2, s=60)

        plt.xlabel("VIX Close (daily avg)")
        plt.ylabel("ret_crsp")
        if add_legend:
            plt.legend(ncol=2, fontsize=8, frameon=True, framealpha=0.85)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_png:
            p = os.path.join(save_dir, "scatter_ret_vs_vix.png")
            plt.savefig(p, dpi=300)
            outputs["scatter_ret_vs_vix_png"] = p
        if show:
            plt.show()
        else:
            plt.close()

    return outputs