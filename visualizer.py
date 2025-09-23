import matplotlib.colors as mcolors
import os
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data.data_handling import DataHandler
from forecast_plotter import ForecastPlotter
from descriptives import describe_target_ret_crsp, plot_returns_and_vol_by_sector

def save_models_legend_image(
    *,
    model_list: List[str],
    model_colors: Dict[str, tuple],  # RGBA tuples
    save_path: str = "results/plots/legend_models.png",
    ncol: Optional[int] = None,
    include_realized: bool = True,
    realized_label: str = "Realized",
    realized_color: str = "black",
):
    """Save a standalone legend image for models (and optional Realized) with transparent bg."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    handles = [Line2D([0], [0], color=model_colors[m], lw=2.2, label=m) for m in model_list]
    if include_realized:
        handles.append(Line2D([0], [0], color=realized_color, lw=2.4, label=realized_label))

    # A short, wide figure that only contains a legend
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
    # No axes, just legend
    fig.patch.set_alpha(0)  # transparent background
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.04)
    plt.close(fig)
    return save_path

if __name__ == "__main__":
    # --- base output directory ---
    base_save = "plots/results"
    plots_dir = os.path.join(base_save, "plots")
    desc_dir = os.path.join(base_save, "descriptives")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(desc_dir, exist_ok=True)

    # --- model setup ---
    model_paths = {
        "CGM":              "results/CGM/20250919-142103/samples_cgm.npy",
        "Gaussian Copula":  "results/TWOSTEP/20250919-133921/samples_two_step.npy",
        "Student-t Copula": "results/TWOSTEP/20250919-135018/samples_two_step.npy",
        "Skewed-t Copula":  "results/TWOSTEP/20250919-135921/samples_two_step.npy",
    }

    symbols_order = ["MSFT","XOM","GE","AAPL","CAT","BA","PFE","JNJ","MRK","JPM"]

    model_colors = {
        "CGM": mcolors.to_rgba("#1f77b4"),  # strong blue
        "Gaussian Copula": mcolors.to_rgba("#ff7f0e"),  # bright orange
        "Student-t Copula": mcolors.to_rgba("#2ca02c"),  # green
        "Skewed-t Copula": mcolors.to_rgba("#d62728"),  # red
    }

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

    model_list = list(model_paths.keys())

    legend_path = save_models_legend_image(
        model_list=model_list,
        model_colors=model_colors,
        save_path=os.path.join(plots_dir, "legend_models.png"),
        ncol=5,  # or 4 for compact
        include_realized=True,
        realized_label="Realized",
        realized_color="black",
    )

    plotter = ForecastPlotter(DataHandler(split_point=0.9))

    # 1) Forecast per-symbol plots
    """plotter.plot_models_per_symbol(
        model_paths=model_paths,
        symbols_order=symbols_order,
        symbols_to_plot=symbols_order,
        sample_to_plot=0,
        save_dir=plots_dir,
        exclude_pandemic=True,
        show=False,
        save_png=True,
        save_pdf=False,
        add_legend=True,
        model_colors=model_colors,
    )"""

    # 2) Forecasts grouped-by-industry
    plotter.plot_grouped_by_sector(
        model_paths=model_paths,
        symbols_order=symbols_order,
        symbol_to_company=symbol_to_company,
        symbol_to_sector=symbol_to_sector,
        sample_to_plot=0,
        save_dir=os.path.join(plots_dir, "sectors"),
        exclude_pandemic=True,
        add_legend=False,
        model_colors=model_colors,
        sector_groups={
            "Financials & Energy": ["Financials", "Energy"],  # combined figure
        },
        sector_order=["Technology", "Industrials", "Healthcare"],  # the rest print as usual
    )

    # 3) descriptives
    """describe_target_ret_crsp(
        DataHandler(split_point=0.9),
        start_date="2010-01-01",
        end_date="2023-12-31",
        rolling_window=30,
        save_dir=desc_dir,
        show=False,
        save_png=True,
        add_legend=True,
    )"""

    # 4) realized returns and vol grouped by sector
    """plot_returns_and_vol_by_sector(
        data_handler=DataHandler(split_point=0.9),
        symbol_to_industry=symbol_to_sector,
        symbols_order=symbols_order,
        rolling_window=30,
        save_dir=os.path.join(desc_dir, "sector"),
        show=False,
        save_png=True,
    )"""
