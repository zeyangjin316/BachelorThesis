import os
from typing import Iterable, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from data.data_handling import DataHandler


class ForecastPlotter:
    """
    Plot forecast samples vs realized returns for multiple assets.

    Notes
    -----
    - Aligns forecast dates using the **last n_origins** test dates
      to match typical forecasting workflows.
    - Handles missing realized values by leaving gaps only for the
      affected asset (does not drop whole days).
    """

    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    def _make_symbol_colors(self, symbols):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        if len(symbols) <= 10:
            cmap = cm.get_cmap("tab10", len(symbols))
        elif len(symbols) <= 20:
            cmap = cm.get_cmap("tab20", len(symbols))
        else:
            cmap = cm.get_cmap("hsv", len(symbols))
        return {s: mcolors.to_rgba(cmap(i)) for i, s in enumerate(symbols)}

    def plot_multivariate_forecasts(
            self,
            *,
            model_paths: Dict[str, str],  # {"CGM": ".../samples_cgm.npy", "Two-Step": ".../samples_ts.npy", ...}
            symbols_order: List[str],
            symbols_to_plot: Optional[Iterable[str]] = None,
            sample_to_plot: int = 0,
            save_dir: str = "results/plots",
            exclude_pandemic: bool = True,
            show: bool = True,
            save_png: bool = True,
            save_pdf: bool = False,
            add_legend: bool = True,
            model_colors: Optional[Dict[str, tuple]] = None,  # per-model color override
            realized_color: Optional[str] = None,  # realized line color override
    ) -> Dict[str, str]:
        """
        Plot one selected sample index across an arbitrary number of models.

        Each model's .npy must be shaped (T, S, N) with S == len(symbols_order).
        We align all models by the **minimum T** across them (from the end).
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        if not model_paths:
            raise ValueError("model_paths cannot be empty (need at least one model name → .npy path).")

        os.makedirs(save_dir, exist_ok=True)

        # ---- Load all forecast samples & validate ----
        loaded = {}  # model -> np.ndarray (T,S,N)
        Ts, Ss, Ns = {}, {}, {}
        for model, path in model_paths.items():
            arr = np.load(path)
            if arr.ndim != 3:
                raise ValueError(f"{model}: expected 3D array (T,S,N). Got {arr.shape}.")
            if not np.isfinite(arr).all():
                raise ValueError(f"{model}: samples contain non-finite values.")
            T, S, N = arr.shape
            if S != len(symbols_order):
                raise ValueError(f"{model}: axis=1 size {S} must equal len(symbols_order)={len(symbols_order)}.")
            loaded[model] = arr
            Ts[model], Ss[model], Ns[model] = T, S, N

        # Ensure requested sample index exists for all models
        min_N = min(Ns.values())
        if sample_to_plot < 0 or sample_to_plot >= min_N:
            raise IndexError(
                f"sample_to_plot={sample_to_plot} is out of bounds for at least one model "
                f"(minimum available samples across models: {min_N})."
            )

        # Align by minimum number of forecast origins across all models
        T_align = min(Ts.values())
        for m in loaded:
            loaded[m] = loaded[m][-T_align:, :, :]  # end alignment

        # Slice the chosen sample (T, S) for each model → DataFrames with aligned index later
        model_frames: Dict[str, pd.DataFrame] = {}
        for m, arr in loaded.items():
            sample = arr[:, :, sample_to_plot]  # (T, S)
            model_frames[m] = pd.DataFrame(sample, columns=symbols_order)

        # ---- Load realized data similarly to your config ----
        data_dict = self.data_handler.get_data(
            exclude_pandemic=exclude_pandemic,
            target_only=True
        )
        test_set = data_dict["test_set"]
        test_set = test_set[test_set["sym_root"].isin(symbols_order)]

        # Get last T_align test dates to match samples
        all_test_dates = np.array(sorted(test_set["date"].unique()))
        if len(all_test_dates) < T_align:
            raise ValueError(f"Not enough test dates ({len(all_test_dates)}) to match samples ({T_align}).")
        test_dates = all_test_dates[-T_align:]

        # Attach index to model dataframes
        for m in model_frames:
            model_frames[m].index = test_dates

        # Realized wide table
        df_real = (
            test_set.pivot(index="date", columns="sym_root", values="ret_crsp")
            .reindex(test_dates)  # keep alignment
            .reindex(columns=symbols_order)  # ensure column order
        )

        # Determine subset to plot
        symbols = list(symbols_order) if symbols_to_plot is None else list(symbols_to_plot)
        unknown = [s for s in symbols if s not in symbols_order]
        if unknown:
            raise ValueError(f"symbols_to_plot contains unknown symbols: {unknown}")

        # Colors for models (auto if not provided)
        models_in_order = list(model_paths.keys())
        if model_colors is None:
            n = len(models_in_order)
            if n <= 10:
                cmap = cm.get_cmap("tab10", n)
            elif n <= 20:
                cmap = cm.get_cmap("tab20", n)
            else:
                cmap = cm.get_cmap("hsv", n)
            model_colors = {m: mcolors.to_rgba(cmap(i)) for i, m in enumerate(models_in_order)}

        # Realized line color (default: black)
        realized_color = realized_color or "black"

        saved: Dict[str, str] = {}

        for sym in symbols:
            plt.figure(figsize=(10, 6))

            # Plot each model's selected sample for this symbol
            for m in models_in_order:
                df_m = model_frames[m]
                plt.plot(
                    df_m.index,
                    df_m[sym],
                    linewidth=2,
                    label=f"{m} sample {sample_to_plot}",  # <-- labeled colored lines
                    color=model_colors.get(m)
                )

            # Plot realized (gaps where NaN)
            plt.plot(
                df_real.index,
                df_real[sym],
                linewidth=2,
                label="Realized",
                color=realized_color,
                alpha=0.9
            )

            # Y padding
            all_vals = [model_frames[m][sym] for m in models_in_order] + [df_real[sym]]
            all_vals = pd.concat(all_vals)
            y_min, y_max = all_vals.min(skipna=True), all_vals.max(skipna=True)
            if pd.notna(y_min) and pd.notna(y_max):
                pad = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
                plt.ylim(y_min - pad, y_max + pad)

            plt.title(f"{sym} — Forecast vs Realized")
            plt.xlabel("Date")
            plt.ylabel("ret_crsp")
            if add_legend:
                plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            base = os.path.join(save_dir, f"{sym}_sample{sample_to_plot}")
            last_path = ""
            if save_png:
                png = base + ".png"
                plt.savefig(png, dpi=300)
                last_path = png
            if save_pdf:
                pdf = base + ".pdf"
                plt.savefig(pdf)
                last_path = pdf or last_path

            if show:
                plt.show()
            else:
                plt.close()

            if last_path:
                saved[sym] = last_path

        return saved

    def describe_target_ret_crsp(
            self,
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
        data_dict = self.data_handler.get_data(
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

        symbol_colors = self._make_symbol_colors(symbols)

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

    def plot_forecasts_grouped_by_industry(
            self,
            *,
            model_files: Dict[str, str],  # {"CGM": ".../samples.npy", "Gaussian Copula": "...", ...}
            symbols_order: List[str],
            symbol_to_industry: Dict[str, str],  # {"AAPL": "Technology -- Consumer Electronics", ...}
            sample_to_plot: int = 0,
            industries_order: Optional[List[str]] = None,
            save_path: str = "results/plots/forecasts_by_industry.png",
            exclude_pandemic: bool = True,
            show: bool = True,
            add_legend: bool = True,
            inline_labels: bool = True,
            model_colors: Optional[Dict[str, tuple]] = None,  # RGBA tuples per model
    ):
        """
        One figure with subplots grouped by industry.
        - Color encodes the model (consistent across subplots).
        - Line style encodes the symbol (consistent within a subplot).
        - Inline labels mark each symbol on the realized line.
        """
        import itertools
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # ---- load and align model arrays (T, S, N) ----
        arrays = {}
        T_list, S_list, N_list = [], [], []
        for label, path in model_files.items():
            arr = np.load(path)
            if arr.ndim != 3:
                raise ValueError(f"{label}: expected (T,S,N), got {arr.shape}")
            if arr.shape[1] != len(symbols_order):
                raise ValueError(f"{label}: symbols_order length {len(symbols_order)} != S {arr.shape[1]}")
            arrays[label] = arr
            T_list.append(arr.shape[0]);
            S_list.append(arr.shape[1]);
            N_list.append(arr.shape[2])

        min_N = min(N_list)
        if not (0 <= sample_to_plot < min_N):
            raise IndexError(f"sample_to_plot={sample_to_plot} out of bounds (min N across models = {min_N})")

        min_T = min(T_list)
        arrays = {k: v[-min_T:, :, :] for k, v in arrays.items()}  # align from end

        # ---- realized data aligned to the same dates ----
        data_dict = self.data_handler.get_data(exclude_pandemic=exclude_pandemic, target_only=True)
        test_set = data_dict["test_set"]
        test_set = test_set[test_set["sym_root"].isin(symbols_order)]
        all_test_dates = np.array(sorted(test_set["date"].unique()))
        if len(all_test_dates) < min_T:
            raise ValueError(f"Not enough test dates ({len(all_test_dates)}) for T={min_T}")
        test_dates = all_test_dates[-min_T:]

        # build per-model DataFrame for selected sample
        df_models = {
            label: pd.DataFrame(arr[:, :, sample_to_plot], index=test_dates, columns=symbols_order)
            for label, arr in arrays.items()
        }
        df_real = (
            test_set.pivot(index="date", columns="sym_root", values="ret_crsp")
            .reindex(test_dates)
            .reindex(columns=symbols_order)
        )

        # ---- industries & symbols per industry ----
        # keep only symbols in symbols_order, preserve symbols_order order in each industry
        industries = {}
        for sym in symbols_order:
            ind = symbol_to_industry.get(sym, "Other")
            industries.setdefault(ind, []).append(sym)

        if industries_order is None:
            industries_order = list(industries.keys())
        else:
            # ensure only industries that exist and preserve custom order
            industries_order = [i for i in industries_order if i in industries]

        n_ind = len(industries_order)
        if n_ind == 0:
            raise ValueError("No industries to plot.")

        # ---- colors for models (RGBA tuples expected) ----
        if model_colors is None:
            # fallback to matplotlib default cycle converted to RGBA
            default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
            cyc = itertools.cycle(default_colors if default_colors else ["C0", "C1", "C2", "C3", "C4", "C5"])
            model_colors = {}
            for m in model_files.keys():
                model_colors[m] = mcolors.to_rgba(next(cyc))

        # ---- line styles for symbols (cycled per subplot) ----
        style_cycle_master = ['-', '--', ':', '-.']

        # ---- figure layout (roughly square) ----
        ncols = int(np.ceil(np.sqrt(n_ind)))
        nrows = int(np.ceil(n_ind / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.8 * nrows), sharex=False, sharey=False)
        axes = np.array(axes).reshape(-1)  # flatten for easy indexing

        # ---- plot per industry ----
        for ax_idx, industry in enumerate(industries_order):
            ax = axes[ax_idx]
            syms = industries[industry]
            if not syms:
                ax.set_visible(False)
                continue

            # style cycle for symbols within this industry
            style_cycle = itertools.cycle(style_cycle_master)

            # track y range for padding
            all_vals = []

            # plot each model across all symbols
            for model_name, dfm in df_models.items():
                color = model_colors[model_name]
                # to make legend compact, we add one handle per model (using first symbol)
                first_symbol = True
                # for symbol-specific styles, we need stable mapping
                sym_to_style = {s: st for s, st in
                                zip(syms, itertools.islice(itertools.cycle(style_cycle_master), len(syms)))}

                for s in syms:
                    ls = sym_to_style[s]
                    series = dfm[s]
                    ax.plot(series.index, series.values,
                            linewidth=1.6, color=color, linestyle=ls,
                            label=(f"{model_name}" if first_symbol else None))
                    first_symbol = False
                    all_vals.append(series)

            # realized lines per symbol (black, styled by symbol)
            sym_to_style = {s: st for s, st in
                            zip(syms, itertools.islice(itertools.cycle(style_cycle_master), len(syms)))}
            for s in syms:
                rs = df_real[s]
                ax.plot(rs.index, rs.values, color="black", linestyle=sym_to_style[s], linewidth=1.8,
                        label=None)
                if inline_labels:
                    last = rs.dropna()
                    if not last.empty:
                        ax.text(last.index[-1], last.iloc[-1], s, ha="left", va="center", fontsize=9, color="black")

                all_vals.append(rs)

            # y padding
            all_vals_cat = pd.concat(all_vals)
            y_min = all_vals_cat.min(skipna=True)
            y_max = all_vals_cat.max(skipna=True)
            if np.isfinite(y_min) and np.isfinite(y_max):
                pad = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
                ax.set_ylim(y_min - pad, y_max + pad)

            ax.set_title(industry, fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_xlabel("Date")
            ax.set_ylabel("ret_crsp")

            # compact legend: models only (colors), symbols shown by inline labels + linestyle
            if add_legend:
                # create one legend per subplot listing models by color
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, loc="upper left", frameon=True, framealpha=0.9, fontsize=9)

        # hide any extra axes
        for k in range(n_ind, len(axes)):
            axes[k].set_visible(False)

        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

        return {"figure": save_path}


if __name__ == "__main__":
    os.makedirs("results/plots", exist_ok=True)

    model_paths = {
        "CGM":              "results/CGM/20250919-142103/samples_cgm.npy",
        "Gaussian Copula":  "results/TWOSTEP/20250919-133921/samples_two_step.npy",
        "Student-t Copula": "results/TWOSTEP/20250919-135018/samples_two_step.npy",
        "Skewed-t Copula":  "results/TWOSTEP/20250919-135921/samples_two_step.npy",
    }

    symbols_order = ["MSFT","XOM","GE","AAPL","CAT","BA","PFE","JNJ","MRK","JPM"]

    plotter = ForecastPlotter(DataHandler(split_point=0.9))
    plotter.plot_multivariate_forecasts(
        model_paths=model_paths,
        symbols_order=symbols_order,
        symbols_to_plot=symbols_order,
        sample_to_plot=0,
        save_dir="results/plots",
        exclude_pandemic=True,
        show=False,
        save_png=True,
        save_pdf=False,
        add_legend=True,  # ensures colored lines are labeled in legend

        model_colors={
            "CGM": mcolors.to_rgba("#66c2a5"),  # teal-green
            "Gaussian Copula": mcolors.to_rgba("#fc8d62"),  # orange
            "Student-t Copula": mcolors.to_rgba("#8da0cb"),  # periwinkle blue
            "Skewed-t Copula": mcolors.to_rgba("#e78ac3"),  # pink-magenta
        }
    )
    outputs = plotter.describe_target_ret_crsp(
        #symbols_subset=["MSFT", "AAPL", "JPM"],  # limit to a few symbols (optional)
        start_date="2010-01-01",  # filter period (optional)
        end_date="2023-12-31",
        rolling_window=30,  # 30-day rolling volatility
        save_dir="results/descriptives",  # where to save results
        show=False,  # don’t open plots interactively
        save_png=True,  # save plots
        add_legend=True  # include legends on plots
    )

    print("Outputs generated:")
    for k, v in outputs.items():
        print(f"{k}: {v}")

