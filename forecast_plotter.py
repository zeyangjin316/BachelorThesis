import os
from typing import Iterable, List, Optional, Dict, Tuple

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
    if provided:
        # ensure tuples
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
    """Load (T,S,N) arrays, check shapes, align to min T from end, slice sample -> DataFrames with no index yet."""
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
        Ts.append(T); Ns.append(N)

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
        T_align: int, exclude_pandemic: bool) -> Tuple[pd.DataFrame, np.ndarray]:
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
    """Thin orchestrator around compact helpers."""
    def __init__(self, data_handler: DataHandler):
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

            # y-pad
            all_vals = pd.concat([frames[m][sym] for m in model_list] + [df_real[sym]])
            y_min, y_max = all_vals.min(skipna=True), all_vals.max(skipna=True)
            if np.isfinite(y_min) and np.isfinite(y_max):
                pad = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
                plt.ylim(y_min - pad, y_max + pad)

            plt.title(f"{sym} — Forecast vs Realized")
            plt.xlabel("Date"); plt.ylabel("ret_crsp"); plt.grid(alpha=0.3)
            if add_legend: plt.legend()
            plt.tight_layout()

            base = os.path.join(save_dir, f"{sym}_sample{sample_to_plot}")
            if save_png: _save_show(None, base + ".png", show)
            if save_pdf: _save_show(None, base + ".pdf", show)
            if not (save_png or save_pdf): _save_show(None, None, show)

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
            # e.g. {"Financials & Energy": ["Financials","Energy"]}
            # layout:
            a4_width: float = 11.69,
            per_row_height: float = 4.2,
            legend_height: float = 0.75,
            use_compact_date_ticks: bool = True,
    ) -> Dict[str, str]:
        import math, os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        os.makedirs(save_dir, exist_ok=True)

        frames, T_align, _ = _load_and_align_models(model_paths, symbols_order, sample_to_plot)
        df_real, test_dates = _get_realized_wide(self.data_handler, symbols_order, T_align, exclude_pandemic)
        for m in frames:
            frames[m].index = test_dates

        # sector -> symbols mapping
        sector_to_syms: Dict[str, List[str]] = {}
        for s in symbols_order:
            sec = symbol_to_sector.get(s, "Other")
            sector_to_syms.setdefault(sec, []).append(s)

        groups: Dict[str, List[str]] = {}
        group_sectors: Dict[str, List[str]] = {}
        used = set()

        # build merged groups first
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

        # then standalone
        remaining = [s for s in (sector_order or list(sector_to_syms.keys())) if s in sector_to_syms and s not in used]
        for sec in remaining:
            groups[sec] = sector_to_syms[sec]
            group_sectors[sec] = [sec]

        model_list = list(model_paths.keys())
        colors = _auto_model_colors(model_list, provided=model_colors)

        saved = {}
        for gname, syms in groups.items():
            if not syms:
                continue

            n = len(syms)
            ncols = 1 if n == 1 else 2
            nrows = math.ceil(n / ncols)
            fig_height = nrows * per_row_height + (legend_height if add_legend else 0.4)

            fig, axes = plt.subplots(nrows, ncols, figsize=(a4_width, fig_height), squeeze=False)
            ax_flat = axes.ravel()

            is_merged_group = gname in (sector_groups or {})

            fig.subplots_adjust(
                left=0.06, right=0.995,
                top=(0.90 if not is_merged_group else 0.93),  # more space if suptitle
                bottom=(0.16 if add_legend else 0.10),
                wspace=0.18, hspace=0.28
            )

            sym_sector = {s: symbol_to_sector.get(s, "Other") for s in syms}
            first_idx_of_sector: Dict[str, int] = {}
            for idx, s in enumerate(syms):
                sec = sym_sector[s]
                if sec not in first_idx_of_sector:
                    first_idx_of_sector[sec] = idx

            for ax, sym in zip(ax_flat, syms):
                for model in model_list:
                    ser = frames[model][sym]
                    ax.plot(ser.index, ser.values, linewidth=2.0, color=colors[model], label=model)
                ax.plot(df_real.index, df_real[sym], linewidth=2.2, color="black", label="Realized", alpha=0.95)

                all_vals = pd.concat([frames[m][sym] for m in model_list] + [df_real[sym]])
                y_min, y_max = all_vals.min(skipna=True), all_vals.max(skipna=True)
                if np.isfinite(y_min) and np.isfinite(y_max):
                    pad = 0.06 * (y_max - y_min if y_max > y_min else 1.0)
                    ax.set_ylim(y_min - pad, y_max + pad)

                company = symbol_to_company.get(sym, "")
                ax.set_title(f"{sym} — {company}", fontsize=11)
                ax.grid(alpha=0.3)
                ax.set_xlabel("Date")
                ax.set_ylabel("ret_crsp")
                ax.tick_params(labelsize=9)

                if use_compact_date_ticks:
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

            # sector headers for merged groups
            if is_merged_group:
                for sec in group_sectors.get(gname, []):
                    if sec not in first_idx_of_sector:
                        continue
                    first_ax = ax_flat[first_idx_of_sector[sec]]
                    first_ax.text(
                        0.5, 1.08, sec,
                        transform=first_ax.transAxes,
                        ha="center", va="bottom",
                        fontsize=12, fontweight="bold"
                    )
            else:
                # standalone sector: put bold figure title
                fig.suptitle(gname, fontsize=13, fontweight="bold", y=0.965)

            for ax in ax_flat[len(syms):]:
                ax.set_visible(False)

            if add_legend:
                handles, labels = [], []
                for ax in ax_flat[:len(syms)]:
                    h, l = ax.get_legend_handles_labels()
                    if h:
                        handles, labels = h, l
                        break
                if handles:
                    fig.legend(
                        handles, labels,
                        loc="lower center", bbox_to_anchor=(0.5, 0.02),
                        ncol=min(len(labels), 5),
                        frameon=True, framealpha=0.95, fontsize=10,
                        handlelength=2.0, handletextpad=0.6, borderaxespad=0.6,
                        columnspacing=1.2, labelspacing=0.5
                    )

            safe = gname.replace(" ", "_").replace("/", "-")
            out_path = os.path.join(save_dir, f"{safe}.png")
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)

            saved[gname] = out_path

        return saved




