"""
End-to-end plotting driver for:
  1) Standalone legend image for models.
  2) Per-symbol forecast vs realized plots.
  3) Sector-grouped forecast figures.
  4) Descriptive plots (per symbol and by sector).

Outputs are written under ./plots/.
"""

import os
from typing import List, Dict, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data.data_handling import DataHandler
from forecast_plotter import ForecastPlotter
from descriptives import describe_target_ret_crsp, plot_returns_and_vol_by_sector


def save_models_legend_image(
    *,
    model_list: List[str],
    model_colors: Dict[str, tuple],
    save_path: str = "plots/legend_models.png",
    ncol: Optional[int] = None,
    include_realized: bool = True,
    realized_label: str = "Realized",
    realized_color: str = "black",
) -> str:
    """
    Save a standalone legend image for models (and optional 'Realized') with a transparent background.

    Parameters
    ----------
    model_list : list[str]
        Display order of model labels in the legend.
    model_colors : dict[str, tuple]
        Mapping model -> RGBA tuple for line color.
    save_path : str, default="plots/legend_models.png"
        Output path for the legend image (PNG).
    ncol : int | None, default=None
        Number of legend columns. If None, it is set to min(#handles, 5).
    include_realized : bool, default=True
        If True, add a legend entry for realized series.
    realized_label : str, default="Realized"
        Display label for the realized series.
    realized_color : str, default="black"
        Color for the realized series line.

    Returns
    -------
    str
        The saved image path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    handles = [Line2D([0], [0], color=model_colors[m], lw=2.2, label=m) for m in model_list]
    if include_realized:
        handles.append(Line2D([0], [0], color=realized_color, lw=2.4, label=realized_label))

    # Short, wide figure that only contains a legend
    fig = plt.figure(figsize=(8.0, 0.8))
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="center",
        ncol=ncol or min(len(handles), 5),
        frameon=True, framealpha=0.95,
        handlelength=2.2, handletextpad=0.6, borderaxespad=0.6,
        columnspacing=1.2, labelspacing=0.5,
        fontsize=10,
    )
    fig.patch.set_alpha(0)  # transparent background
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.04)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    # Base output directories
    base_save = "plots"
    forecast_results_dir = os.path.join(base_save, "results")
    desc_dir = os.path.join(base_save, "descriptives")
    os.makedirs(forecast_results_dir, exist_ok=True)
    os.makedirs(desc_dir, exist_ok=True)

    # Model setup (update paths as needed)
    model_paths = {
        "CGM":              "results/CGM/20250919-142103/samples_cgm.npy",
        "Gaussian Copula":  "results/TWOSTEP/20250919-133921/samples_two_step.npy",
        "Student-t Copula": "results/TWOSTEP/20250919-135018/samples_two_step.npy",
        "Skewed-t Copula":  "results/TWOSTEP/20250919-135921/samples_two_step.npy",
    }
    symbols_order = ["MSFT", "XOM", "GE", "AAPL", "CAT", "BA", "PFE", "JNJ", "MRK", "JPM"]

    # Fixed display colors per model
    model_colors = {
        "CGM": mcolors.to_rgba("#1f77b4"),
        "Gaussian Copula": mcolors.to_rgba("#ff7f0e"),
        "Student-t Copula": mcolors.to_rgba("#2ca02c"),
        "Skewed-t Copula": mcolors.to_rgba("#d62728"),
    }

    # Company and sector metadata
    symbol_to_company = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
        "JPM": "JPMorgan Chase & Co.",
        "XOM": "Exxon Mobil Corp.",
        "GE": "General Electric Co.",
        "CAT": "Caterpillar Inc.",
        "BA": "Boeing Co.",
        "PFE": "Pfizer Inc.",
        "JNJ": "Johnson & Johnson",
        "MRK": "Merck & Co., Inc.",
    }
    symbol_to_sector = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "JPM": "Financials",
        "XOM": "Energy",
        "GE": "Industrials",
        "CAT": "Industrials",
        "BA": "Industrials",
        "PFE": "Healthcare",
        "JNJ": "Healthcare",
        "MRK": "Healthcare",
    }

    # Legend image (shared across plots/documents)
    model_list = list(model_paths.keys())
    legend_path = save_models_legend_image(
        model_list=model_list,
        model_colors=model_colors,
        save_path=os.path.join(forecast_results_dir, "legend_models.png"),
        ncol=5,
        include_realized=True,
        realized_label="Realized",
        realized_color="black",
    )

    # Plotter
    plotter = ForecastPlotter(DataHandler(split_point=0.9))

    # 1) Forecast per-symbol plots
    plotter.plot_models_per_symbol(
        model_paths=model_paths,
        symbols_order=symbols_order,
        symbols_to_plot=symbols_order,
        sample_to_plot=0,
        save_dir=os.path.join(forecast_results_dir, "per symbol"),
        exclude_pandemic=True,
        show=False,
        save_png=True,
        save_pdf=False,
        add_legend=True,
        model_colors=model_colors,
    )

    # 2) Forecasts grouped by industry (fixed layout)
    plotter.plot_grouped_by_sector(
        model_paths=model_paths,
        symbols_order=symbols_order,
        symbol_to_company=symbol_to_company,
        symbol_to_sector=symbol_to_sector,
        sample_to_plot=0,
        save_dir=os.path.join(forecast_results_dir, "by sectors"),
        exclude_pandemic=True,
        add_legend=False,
        model_colors=model_colors,
        sector_groups={
            "Financials & Energy": ["Financials", "Energy"],  # merged figure
        },
        sector_order=["Technology", "Industrials", "Healthcare"],
    )

    # 3) Descriptives per symbol
    describe_target_ret_crsp(
        DataHandler(split_point=0.9),
        rolling_window=30,
        save_dir=os.path.join(desc_dir, "per symbol"),
        show=False,
        save_png=True,
        add_legend=True,
    )

    # 4) Realized returns and rolling vol grouped by sector
    plot_returns_and_vol_by_sector(
        data_handler=DataHandler(split_point=0.9),
        symbol_to_industry=symbol_to_sector,
        symbols_order=symbols_order,
        rolling_window=30,
        save_dir=os.path.join(desc_dir, "bysector"),
        show=False,
        save_png=True,
    )
