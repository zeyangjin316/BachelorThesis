import os
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from data.data_handling import DataHandler


# -----------------------------
# Helpers
# -----------------------------

def _auto_model_colors(model_names: List[str], provided: Optional[Dict[str, tuple]] = None) -> Dict[str, tuple]:
    """
    Produce a consistent RGBA color for each model.

    Parameters
    ----------
    model_names : list[str]
        Model labels to color.
    provided : dict[str, tuple] | None
        Optional explicit mapping model -> RGBA. If given, it is used as-is.

    Returns
    -------
    dict[str, tuple]
        Mapping model -> RGBA color.
    """
    if provided:
        return {k: (tuple(v) if not isinstance(v, tuple) else v) for k, v in provided.items()}
    n = len(model_names)
    if n <= 10:
        cmap = cm.get_cmap("tab10", n)
    elif n <= 20:
        cmap = cm.get_cmap("tab20", n)
    else:
        cmap = cm.get_cmap("hsv", n)
    return {m: mcolors.to_rgba(cmap(i)) for i, m in enumerate(model_names)}


def _load_and_align_models(
    model_paths: Dict[str, str],
    symbols_order: List[str],
    sample_to_plot: int
) -> Tuple[Dict[str, pd.DataFrame], int, int]:
    """
    Load (T, S, N) samples for each model, align to the shortest T, and select one sample.

    Parameters
    ----------
    model_paths : dict[str, str]
        Model label -> path to .npy array (T, S, N).
    symbols_order : list[str]
        Asset order matching axis=1 of the arrays.
    sample_to_plot : int
        Index along N (sample dimension) to extract.

    Returns
    -------
    frames : dict[str, pd.DataFrame]
        Model -> DataFrame (aligned T, columns=symbols_order). Index unset.
    T_align : int
        Aligned number of days (min T across models).
    min_N : int
        Minimum number of samples across models.
    """
    if not model_paths:
        raise ValueError("model_paths cannot be empty.")

    loaded = {}
    Ts, Ns = [], []
    for label, path in model_paths.items():
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f"{label}: expected (T,S,N). Got {arr.shape}.")
        T, S, N = arr.shape
        if S != len(symbols_order):
            raise ValueError(f"{label}: S={S} != len(symbols_order)={len(symbols_order)}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{label}: contains non-finite values.")
        loaded[label] = arr
        Ts.append(T)
        Ns.append(N)

    min_N = min(Ns)
    if not (0 <= sample_to_plot < min_N):
        raise IndexError(f"sample_to_plot={sample_to_plot} out of bounds (min N across models = {min_N}).")

    T_align = min(Ts)
    frames = {}
    for label, arr in loaded.items():
        arr_aligned = arr[-T_align:, :, :]
        sample = arr_aligned[:, :, sample_to_plot]  # (T_align, S)
        frames[label] = pd.DataFrame(sample, columns=symbols_order)

    return frames, T_align, min_N


def _get_realized_wide(
    handler: DataHandler,
    symbols_order: List[str],
    T_align: int,
    exclude_pandemic: bool
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build realized returns in wide format aligned to the last T_align dates.

    Parameters
    ----------
    handler : DataHandler
        Data accessor.
    symbols_order : list[str]
        Desired column order.
    T_align : int
        Number of test dates to match (from the end).
    exclude_pandemic : bool
        If True, drop dates >= 2020-01-01.

    Returns
    -------
    df_real : pd.DataFrame
        Wide frame (date index, columns=symbols_order).
    test_dates : np.ndarray
        Sorted array of aligned dates.
    """
    data_dict = handler.get_data(exclude_pandemic=exclude_pandemic, target_only=True)
    test_set = data_dict["test_set"]
    test_set = test_set[test_set["sym_root"].isin(symbols_order)]
    all_test_dates = np.array(sorted(test_set["date"].unique()))
    if len(all_test_dates) < T_align:
        raise ValueError(f"Not enough test dates ({len(all_test_dates)}) to match samples ({T_align}).")
    test_dates = all_test_dates[-T_align:]
    df_real = (
        test_set.pivot(index="date", columns="sym_root", values="ret_crsp")
        .reindex(test_dates)
        .reindex(columns=symbols_order)
    )
    return df_real, test_dates


def _save_show(fig_or_none, path: Optional[str], show: bool, dpi: int = 300):
    """
    Save current matplotlib figure to disk and/or display it.

    Parameters
    ----------
    fig_or_none : object
        Unused placeholder; present for a stable signature.
    path : str | None
        Destination path (including extension). If None, skip saving.
    show : bool
        If True, display the figure; otherwise close it.
    dpi : int, default=300
        Resolution for saved raster output.
    """
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------
# Public API
# -----------------------------

class ForecastPlotter:
    """
    Thin plotting orchestrator for visualizing model sample paths vs realized returns.
    """

    def __init__(self, data_handler: DataHandler):
        """
        Parameters
        ----------
        data_handler : DataHandler
            Data source used to retrieve realized returns.
        """
        self.data_handler = data_handler

    def plot_models_per_symbol(
        self,
        *,
        model_paths: Dict[str, str],
        symbols_order: List[str],
        symbols_to_plot: Optional[Iterable[str]] = None,
        sample_to_plot: int = 0,
        save_dir: str = "results/plots",
        exclude_pandemic: bool = True,
        show: bool = False,
        save_png: bool = True,
        save_pdf: bool = False,
        add_legend: bool = True,
        model_colors: Optional[Dict[str, tuple]] = None,
        realized_color: str = "black",
    ) -> Dict[str, str]:
        """
        Plot one figure per symbol: selected sample path from each model vs realized returns.

        Parameters
        ----------
        model_paths : dict[str, str]
            Model label -> .npy path (T,S,N).
        symbols_order : list[str]
            Asset order matching axis=1 of model arrays.
        symbols_to_plot : Iterable[str] | None, default=None
            Subset of symbols to plot; if None, plot all symbols_order.
        sample_to_plot : int, default=0
            Sample index along N to visualize.
        save_dir : str, default="results/plots"
            Output directory for figures.
        exclude_pandemic : bool, default=True
            If True, exclude dates >= 2020-01-01 when building realized panel.
        show : bool, default=False
            If True, display figures interactively.
        save_png : bool, default=True
            If True, save PNG files.
        save_pdf : bool, default=False
            If True, also save PDF files.
        add_legend : bool, default=True
            If True, include a legend.
        model_colors : dict[str, tuple] | None, default=None
            Optional explicit colors per model.
        realized_color : str, default="black"
            Color for realized return line.

        Returns
        -------
        dict[str, str]
            Mapping symbol -> saved path (PNG or PDF).
        """
        frames, T_align, _ = _load_and_align_models(model_paths, symbols_order, sample_to_plot)
        df_real, test_dates = _get_realized_wide(self.data_handler, symbols_order, T_align, exclude_pandemic)
        for m in frames:
            frames[m].index = test_dates

        symbols = list(symbols_order) if symbols_to_plot is None else list(symbols_to_plot)
        unknown = [s for s in symbols if s not in symbols_order]
        if unknown:
            raise ValueError(f"symbols_to_plot contains unknown symbols: {unknown}")

        model_list = list(model_paths.keys())
        colors = _auto_model_colors(model_list, provided=model_colors)

        saved = {}
        for sym in symbols:
            plt.figure(figsize=(10, 6))
            for model in model_list:
                series = frames[model][sym]
                plt.plot(series.index, series.values, linewidth=2, color=colors[model],
                         label=f"{model} (sample {sample_to_plot})")
            plt.plot(df_real.index, df_real[sym], linewidth=2, color=realized_color, label="Realized", alpha=0.9)

            # y padding
            all_vals = pd.concat([frames[m][sym] for m in model_list] + [df_real[sym]])
            y_min, y_max = all_vals.min(skipna=True), all_vals.max(skipna=True)
            if np.isfinite(y_min) and np.isfinite(y_max):
                pad = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
                plt.ylim(y_min - pad, y_max + pad)

            plt.title(f"{sym} — Forecast vs Realized")
            plt.xlabel("Date")
            plt.ylabel("ret_crsp")
            plt.grid(alpha=0.3)
            if add_legend:
                plt.legend()
            plt.tight_layout()

            base = os.path.join(save_dir, f"{sym}_sample{sample_to_plot}")
            if save_png:
                _save_show(None, base + ".png", show)
            if save_pdf:
                _save_show(None, base + ".pdf", show)
            if not (save_png or save_pdf):
                _save_show(None, None, show)

            saved[sym] = (base + (".pdf" if save_pdf else ".png")) if (save_png or save_pdf) else ""

        return saved

    def plot_grouped_by_sector(
        self,
        *,
        model_paths: Dict[str, str],
        symbols_order: List[str],
        symbol_to_company: Dict[str, str],
        symbol_to_sector: Dict[str, str],
        sample_to_plot: int = 0,
        save_dir: str = "results/plots/sectors",
        exclude_pandemic: bool = True,
        show: bool = False,
        add_legend: bool = True,
        model_colors: Optional[Dict[str, tuple]] = None,
        sector_order: Optional[List[str]] = None,
        sector_groups: Optional[Dict[str, List[str]]] = None,
        # fixed layout across all figures
        a4_width: float = 11.69,
        figsize: Tuple[float, float] = (16, 9),
        ncols: int = 2,
        legend_height: float = 0.70,
        wspace: float = 0.20,
        hspace: float = 0.22,
        inner_margins: Tuple[float, float, float, float] = (0.06, 0.995, 0.93, 0.12),
        use_compact_date_ticks: bool = True,
    ) -> Dict[str, str]:
        """
        Plot grouped sector figures with a fixed grid: each subplot shows one symbol.

        Parameters
        ----------
        model_paths : dict[str, str]
            Model label -> .npy path (T, S, N).
        symbols_order : list[str]
            Asset order matching axis=1 of model arrays.
        symbol_to_company : dict[str, str]
            Symbol -> company display name.
        symbol_to_sector : dict[str, str]
            Symbol -> sector name.
        sample_to_plot : int, default=0
            Sample index along N to visualize.
        save_dir : str, default="results/plots/sectors"
            Output directory for figures.
        exclude_pandemic : bool, default=True
            If True, exclude dates >= 2020-01-01 when building realized panel.
        show : bool, default=False
            If True, display figures interactively.
        add_legend : bool, default=True
            If True, include a shared legend at the bottom.
        model_colors : dict[str, tuple] | None, default=None
            Optional explicit colors per model.
        sector_order : list[str] | None, default=None
            Optional sector ordering for output.
        sector_groups : dict[str, list[str]] | None, default=None
            Optional merged groups: name -> list of sector names.
        a4_width : float, default=11.69
            Unused placeholder kept for compatibility (A4 width in inches).
        figsize : tuple[float, float], default=(16, 9)
            Fixed figure size for all sector figures.
        ncols : int, default=2
            Number of columns in the grid.
        legend_height : float, default=0.70
            Unused placeholder kept for compatibility.
        wspace : float, default=0.20
            Horizontal space between subplots.
        hspace : float, default=0.22
            Vertical space between subplots.
        inner_margins : tuple[float, float, float, float], default=(0.06, 0.995, 0.93, 0.12)
            (left, right, top, bottom) margins for subplots.
        use_compact_date_ticks : bool, default=True
            If True, use quarterly date ticks.

        Returns
        -------
        dict[str, str]
            Mapping group/sector name -> saved path.
        """
        import math
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        os.makedirs(save_dir, exist_ok=True)

        # load aligned forecasts + realized panel
        frames, T_align, _ = _load_and_align_models(model_paths, symbols_order, sample_to_plot)
        df_real, test_dates = _get_realized_wide(self.data_handler, symbols_order, T_align, exclude_pandemic)
        for m in frames:
            frames[m].index = test_dates

        model_list = list(model_paths.keys())
        colors = _auto_model_colors(model_list, provided=model_colors)

        # sector -> symbols
        sector_to_syms: Dict[str, List[str]] = {}
        for s in symbols_order:
            sec = symbol_to_sector.get(s, "Other")
            sector_to_syms.setdefault(sec, []).append(s)

        # merged groups first, then remaining single sectors (respect order)
        groups: Dict[str, List[str]] = {}
        group_sectors: Dict[str, List[str]] = {}
        used = set()

        if sector_groups:
            for gname, secs in sector_groups.items():
                seq_syms, clean_secs = [], []
                for sec in secs:
                    if sec in sector_to_syms:
                        seq_syms += sector_to_syms[sec]
                        clean_secs.append(sec)
                        used.add(sec)
                if seq_syms:
                    groups[gname] = seq_syms
                    group_sectors[gname] = clean_secs

        remaining = [s for s in (sector_order or list(sector_to_syms.keys()))
                     if s in sector_to_syms and s not in used]
        for sec in remaining:
            groups[sec] = sector_to_syms[sec]
            group_sectors[sec] = [sec]

        # global symmetric y-limits across all figures
        global_min, global_max = np.inf, -np.inf
        for sym in symbols_order:
            vals = [frames[m][sym] for m in model_list if sym in frames[m].columns]
            if sym in df_real.columns:
                vals.append(df_real[sym])
            if not vals:
                continue
            s_all = pd.concat(vals)
            vmin, vmax = s_all.min(skipna=True), s_all.max(skipna=True)
            if np.isfinite(vmin):
                global_min = min(global_min, vmin)
            if np.isfinite(vmax):
                global_max = max(global_max, vmax)
        abs_max = float(max(abs(global_min), abs(global_max))) if np.isfinite(global_min) and np.isfinite(global_max) else 1.0
        if abs_max == 0:
            abs_max = 1.0
        ylimits = (-abs_max, abs_max)

        # fixed grid for all figures
        nmax = max(len(syms) for syms in groups.values())
        nrows = max(1, math.ceil(nmax / ncols))

        saved = {}
        sorted_groups = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

        for gname, syms in sorted_groups:
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
            ax_flat = axes.ravel()

            left, right, top, bottom = inner_margins
            fig.subplots_adjust(
                left=left, right=right,
                top=top, bottom=(bottom if add_legend else max(0.06, bottom - 0.04)),
                wspace=wspace, hspace=hspace
            )

            # plot each symbol; hide unused slots
            for i, ax in enumerate(ax_flat):
                if i < len(syms):
                    sym = syms[i]
                    for model in model_list:
                        ser = frames[model][sym]
                        ax.plot(ser.index, ser.values, linewidth=2.0, color=colors[model], label=model)
                    ax.plot(df_real.index, df_real[sym], linewidth=2.2, color="black", label="Realized", alpha=0.95)

                    ax.set_ylim(*ylimits)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("ret_crsp")
                    company = symbol_to_company.get(sym, "")
                    ax.set_title(f"{sym} — {company}", fontsize=11)
                    ax.grid(alpha=0.3)
                    ax.tick_params(labelsize=9)

                    if use_compact_date_ticks:
                        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                else:
                    ax.set_visible(False)

            # title and shared legend
            is_merged_group = gname in (sector_groups or {})
            fig.suptitle(gname, fontsize=13, fontweight="bold", y=0.975 if is_merged_group else 0.965)

            if add_legend:
                handles, labels = [], []
                for ax in ax_flat:
                    if ax.get_visible():
                        h, l = ax.get_legend_handles_labels()
                        if h:
                            handles, labels = h, l
                            break
                if handles:
                    fig.legend(
                        handles, labels,
                        loc="lower center", bbox_to_anchor=(0.5, 0.03),
                        ncol=min(len(labels), 5),
                        frameon=True, framealpha=0.95, fontsize=10,
                        handlelength=2.0, handletextpad=0.6, borderaxespad=0.6,
                        columnspacing=1.2, labelspacing=0.5
                    )

            safe = gname.replace(" ", "_").replace("/", "-")
            for ext in ["png", "pdf"]:
                out_path = os.path.join(save_dir, f"{safe}.{ext}")
                fig.savefig(out_path, dpi=300, bbox_inches=None)
            saved[gname] = out_path

            if show:
                plt.show()
            else:
                plt.close(fig)

        return saved
