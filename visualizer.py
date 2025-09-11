import os
from typing import Iterable, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.data_handling import DataHandler


class ForecastPlotter:
    """
    Utility for plotting multivariate forecast samples from two models (e.g., CGM vs Two-Step)
    against realized returns over the test period used by the experiments.

    Typical usage:
        plotter = ForecastPlotter(DataHandler(split_point=0.9))
        outputs = plotter.plot_multivariate_forecasts(
            path_cgm="results/CGM/20250826-101253/samples_cgm.npy",
            path_ts="results/COMPARISON/20250819-232955/samples_two_step.npy",
            symbols_order=["MSFT","XOM","GE","AAPL","CAT","BA","PFE","JNJ","MRK","JPM"],
            symbols_to_plot=None,   # None => plot all in symbols_order
            sample_to_plot=0,
            save_dir="results/plots",
            show=True,
            save_png=True,
            save_pdf=False,
        )
    """

    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    def _make_symbol_colors(self, symbols):
        """Return a {symbol: RGBA} mapping with distinct, stable colors."""
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # Prefer a qualitative map that works up to ~20 categories
        if len(symbols) <= 10:
            cmap = cm.get_cmap("tab10", len(symbols))
        elif len(symbols) <= 20:
            cmap = cm.get_cmap("tab20", len(symbols))
        else:
            # Fallback for many symbols: evenly spaced on HSV
            cmap = cm.get_cmap("hsv", len(symbols))

        return {s: mcolors.to_rgba(cmap(i)) for i, s in enumerate(symbols)}

    def plot_multivariate_forecasts(
        self,
        *,
        path_cgm: str,
        path_ts: str,
        symbols_order: List[str],
        symbols_to_plot: Optional[Iterable[str]] = None,
        sample_to_plot: int = 0,
        save_dir: str = "results/plots",
        standardize: bool = False,
        filter_features: bool = True,
        exclude_pandemic: bool = True,
        show: bool = True,
        save_png: bool = True,
        save_pdf: bool = False,
        add_legend: bool = True,
    ) -> Dict[str, str]:
        """
        Plot time series per selected symbol comparing a single forecast *sample* from
        two models (CGM and Two-Step) to realized returns.

        Parameters
        ----------
        path_cgm : str
            Path to CGM samples .npy of shape (n_origins, n_symbols, n_samples).
        path_ts : str
            Path to Two-Step samples .npy with the same shape semantics.
        symbols_order : list of str
            The order of symbols along axis=1 of both .npy files.
        symbols_to_plot : iterable of str, optional
            Subset of symbols to plot. Defaults to all in symbols_order.
        sample_to_plot : int
            Which sample index (0-based) to visualize.
        save_dir : str
            Directory to write plots.
        standardize, filter_features, exclude_pandemic : bool
            Passed to DataHandler.get_data to reproduce experiment config.
        show : bool
            If True, display figures with plt.show().
        save_png : bool
            If True, save a PNG per symbol.
        save_pdf : bool
            If True, also save a PDF per symbol.
        add_legend : bool
            If True, include a legend on each plot.

        Returns
        -------
        dict
            Mapping symbol -> saved PNG path (if save_png) or last saved path.
        """
        os.makedirs(save_dir, exist_ok=True)

        # ---- Load samples ----
        samples_cgm = np.load(path_cgm)  # (n_origins, n_symbols, n_samples)
        samples_ts = np.load(path_ts)    # same shape semantics

        if samples_cgm.ndim != 3 or samples_ts.ndim != 3:
            raise ValueError("Expected 3D arrays for both samples files")

        n_origins_cgm, n_symbols_cgm, n_samples_cgm = samples_cgm.shape
        n_origins_ts, n_symbols_ts, n_samples_ts = samples_ts.shape

        if n_symbols_cgm != len(symbols_order):
            raise ValueError("symbols_order length must equal axis=1 size of CGM array")
        if n_symbols_ts != len(symbols_order):
            raise ValueError("symbols_order length must equal axis=1 size of Two-Step array")

        # We allow different n_origins as long as they start the same day; we'll align by the min length.
        n_origins = min(n_origins_cgm, n_origins_ts)
        if n_origins == 0:
            raise ValueError("No forecast origins found in the provided samples.")

        if sample_to_plot < 0 or sample_to_plot >= min(n_samples_cgm, n_samples_ts):
            raise IndexError(
                f"sample_to_plot={sample_to_plot} out of bounds. "
                f"CGM has {n_samples_cgm} samples, Two-Step has {n_samples_ts}."
            )

        # Slice to aligned origins and select a common sample
        sample_cgm = samples_cgm[:n_origins, :, sample_to_plot]  # (n_origins, n_symbols)
        sample_ts  = samples_ts[:n_origins, :, sample_to_plot]

        # ---- Load data using DataHandler (same configuration as experiments) ----
        data_dict = self.data_handler.get_data(
            standardize=standardize,
            filter_features=filter_features,
            exclude_pandemic=exclude_pandemic,
        )

        test_set = data_dict["test_set"]
        test_set = test_set[test_set["sym_root"].isin(symbols_order)]

        # Get the first n_origins test dates (assumes both models start at same date)
        test_dates_all = sorted(test_set["date"].unique())
        if len(test_dates_all) < n_origins:
            raise ValueError(
                f"Not enough test dates ({len(test_dates_all)}) to match n_origins ({n_origins})."
            )
        test_dates = test_dates_all[:n_origins]

        # ---- Create sample DataFrames with the correct dates ----
        df_sample_cgm = pd.DataFrame(sample_cgm, index=test_dates, columns=symbols_order)
        df_sample_ts  = pd.DataFrame(sample_ts,  index=test_dates, columns=symbols_order)

        # ---- Get realized returns for the sample dates ----
        real_data = test_set[test_set["date"].isin(test_dates)]
        df_real = (
            real_data.pivot(index="date", columns="sym_root", values="ret_crsp")
            .reindex(test_dates)
            .reindex(columns=symbols_order)
        )

        # Determine which symbols to draw
        symbols = list(symbols_order) if symbols_to_plot is None else list(symbols_to_plot)
        missing = [s for s in symbols if s not in symbols_order]
        if missing:
            raise ValueError(f"symbols_to_plot contains unknown symbols: {missing}")

        saved_paths: Dict[str, str] = {}

        # ---- Plot: time series per selected symbol ----
        for sym in symbols:
            plt.figure(figsize=(10, 6))

            plt.plot(df_sample_cgm.index, df_sample_cgm[sym], linewidth=2, label=f"CGM sample {sample_to_plot}")
            plt.plot(df_sample_ts.index,  df_sample_ts[sym],  linewidth=2, label=f"Two-Step sample {sample_to_plot}")
            plt.plot(df_real.index,       df_real[sym],       linewidth=2, label="Realized")

            # dynamic y-axis range with margin
            all_vals = pd.concat([df_sample_cgm[sym], df_sample_ts[sym], df_real[sym]])
            y_min, y_max = all_vals.min(), all_vals.max()
            y_range = (y_max - y_min) if pd.notna(y_max) and pd.notna(y_min) else 1.0
            plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            plt.title(f"{sym} — Forecast vs Realized")
            plt.xlabel("Date")
            plt.ylabel("ret_crsp")
            if add_legend:
                plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            basepath = os.path.join(save_dir, f"{sym}_sample{sample_to_plot}")
            last_path = ""
            if save_png:
                png_path = basepath + ".png"
                plt.savefig(png_path, dpi=300)
                last_path = png_path
            if save_pdf:
                pdf_path = basepath + ".pdf"
                plt.savefig(pdf_path)
                last_path = pdf_path or last_path

            if show:
                plt.show()
            else:
                plt.close()

            if last_path:
                saved_paths[sym] = last_path

        return saved_paths

    def describe_target_ret_crsp(
            self,
            *,
            symbols_subset: Optional[Iterable[str]] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            rolling_window: int = 30,
            save_dir: str = "figures",  # main figures folder
            csv_dir: str = "results/descriptives",  # CSV folder
            standardize: bool = False,
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
            standardize=standardize,
            filter_features=filter_features,
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

        # ---- 6) Boxplot ----
        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(
            [ret_wide[s].dropna().values for s in symbols],
            labels=symbols, vert=True, showfliers=False, patch_artist=True
        )

        # Color each box with its symbol color
        for patch, s in zip(bp["boxes"], symbols):
            patch.set_facecolor(symbol_colors[s])
        for median, s in zip(bp["medians"], symbols):
            median.set_color("black")  # keep medians visible
        for whisker in bp["whiskers"]:
            whisker.set_color("gray")
        for cap in bp["caps"]:
            cap.set_color("gray")

        plt.xlabel("Symbol")
        plt.ylabel("ret_crsp")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
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


if __name__ == "__main__":
    # Example CLI-like usage; adjust as needed.
    os.makedirs("results/plots", exist_ok=True)

    PATH_CGM = "results/CGM/20250826-101253/samples_cgm.npy"
    PATH_TS  = "results/COMPARISON/20250819-232955/samples_two_step.npy"

    symbols_order = [
        "MSFT", "XOM", "GE", "AAPL", "CAT", "BA", "PFE", "JNJ", "MRK", "JPM"
    ]

    plotter = ForecastPlotter(DataHandler(split_point=0.9))
    plotter.plot_multivariate_forecasts(
        path_cgm=PATH_CGM,
        path_ts=PATH_TS,
        symbols_order=symbols_order,
        symbols_to_plot=symbols_order,
        sample_to_plot=0,
        save_dir="results/plots",
        standardize=False,
        filter_features=True,
        exclude_pandemic=True,
        show=False,
        save_png=True,
        save_pdf=False,
        add_legend=False,
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

