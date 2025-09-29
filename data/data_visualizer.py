import os
import numpy as np
import pandas as pd
import logging

from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _save_dataframe_png(df: pd.DataFrame, png_path: str | None, head_n: int, dpi: int) -> str:
    """
    Render the head of a DataFrame to a PNG table for inspection.

    Parameters
    ----------
    df : pd.DataFrame
        Data to render.
    png_path : str | None
        Destination path. If None, a timestamped file is created under `plots/`.
    head_n : int
        Number of rows to display.
    dpi : int
        Dots per inch for saved image.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if png_path is None:
        os.makedirs("plots", exist_ok=True)
        png_path = os.path.join(
            "plots",
            f"final_dataframe_head{min(head_n, len(df))}_{df.shape[1]}cols_{stamp}.png"
        )
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)

    df_for_png = df.head(head_n).copy()

    # Format datetime and float columns
    for c in df_for_png.columns:
        if np.issubdtype(df_for_png[c].dtype, np.datetime64):
            df_for_png[c] = pd.to_datetime(df_for_png[c], errors="coerce").dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(df_for_png[c]):
            df_for_png[c] = df_for_png[c].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            df_for_png[c] = df_for_png[c].astype(str)

    nrows, ncols = df_for_png.shape

    # Truncate long cell text if many columns
    max_cell_chars = max(10, int(34 - 0.7 * ncols))
    df_for_png = df_for_png.applymap(
        lambda x: x if len(x) <= max_cell_chars else (x[: max_cell_chars - 1] + "…")
    )

    # Wrap headers at underscores
    pretty_cols = [str(c).replace("_", "\n") for c in df_for_png.columns]
    max_header_lines = max(lbl.count("\n") + 1 for lbl in pretty_cols) if ncols else 1

    # Figure size scales with number of rows and columns
    fig_w = max(14.0, min(60.0, 1.0 * ncols + 6.0))
    fig_h = max(4.5, min(60.0, 0.35 * (nrows + 1) + 0.6 * (max_header_lines - 1)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.margins(0)

    tbl = ax.table(
        cellText=df_for_png.values,
        colLabels=pretty_cols,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        colWidths=[1.0 / ncols] * ncols if ncols else None,
        bbox=[0, 0, 1, 1],
    )

    # Font size based on number of columns
    if ncols <= 8:
        fs = 10
    elif ncols <= 12:
        fs = 9
    elif ncols <= 16:
        fs = 8
    elif ncols <= 22:
        fs = 7
    else:
        fs = 6
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fs)

    # Adjust header row for multi-line labels
    header_height_factor = 1.0 + 0.35 * (max_header_lines - 1)
    tbl.scale(1.0, 1.0 + 0.05 * (max_header_lines - 1))

    for j in range(ncols):
        cell = tbl.get_celld()[(0, j)]
        cell.set_text_props(weight="bold", va="center")
        cell.set_facecolor("#f0f0f0")
        cell.set_height(cell.get_height() * header_height_factor)

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return png_path


def visualize_ts(
    self,
    symbols: list[str] | None = None,
    save: bool = False,
    filename: str | None = None,
    plots_dir: str = "plots",
    dpi: int = 200,
    show_title: bool = True,
):
    """
    Plot the return time series per symbol from the merged dataset.

    Parameters
    ----------
    self : object
        Object that holds a `reader` with `.read_data()` and `.merge_all()` methods.
    symbols : list[str] | None
        Symbols to plot. If None, all available symbols are shown.
    save : bool, default=False
        If True, save the figure as PNG.
    filename : str | None, default=None
        Filename for saved figure. If None, a timestamped name is used.
    plots_dir : str, default="plots"
        Directory where the plot is saved.
    dpi : int, default=200
        Resolution of saved figure.
    show_title : bool, default=True
        If True, add a descriptive title.

    Returns
    -------
    dict
        {
          "panel": pd.DataFrame (all symbols),
          "panel_sel": pd.DataFrame (plotted symbols),
          "save_path": str | None
        }
    """
    # Load full dataset
    self.reader.read_data()
    self.reader.merge_all()
    full_df = self.reader.data.copy()

    # Verify required columns
    required = {"date", "sym_root", "ret_crsp"}
    missing = required - set(full_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Pivot into panel: date x symbol
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = full_df.sort_values(["date", "sym_root"], ignore_index=True)
    panel = (
        full_df.pivot_table(index="date", columns="sym_root", values="ret_crsp", aggfunc="first")
        .sort_index()
    )

    # Select subset of symbols
    available = list(panel.columns)
    if symbols is None:
        symbols = available
    else:
        symbols = [s for s in symbols if s in available]
        if not symbols:
            raise ValueError(f"No requested symbols found. Available: {available}")

    panel_sel = panel[symbols]

    # Dynamic sizing
    n_dates = panel_sel.shape[0]
    n_syms = len(symbols)
    width_ts = max(16.0, min(36.0, n_dates / 40.0))
    height_ts = 6.5

    # Plot
    fig, ax = plt.subplots(figsize=(width_ts, height_ts))
    for s in symbols:
        ax.plot(panel_sel.index, panel_sel[s], linewidth=1.8, label=s)

    if show_title:
        ax.set_title(f"ret_crsp — all dates ({n_dates} rows) — {n_syms} symbols")
    ax.set_xlabel("date")
    ax.set_ylabel("ret_crsp")
    ax.grid(alpha=0.3)
    ax.margins(x=0)
    fig.autofmt_xdate()

    # Legend outside the main axes
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.82, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    # Save figure if requested
    save_path = None
    if save:
        os.makedirs(plots_dir, exist_ok=True)
        if filename is None:
            start = panel_sel.index.min().strftime("%Y%m%d")
            end = panel_sel.index.max().strftime("%Y%m%d")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            symtag = f"{min(5, n_syms)}syms" if n_syms > 5 else f"{n_syms}syms"
            filename = f"ret_crsp_timeseries_{start}-{end}_{symtag}_{stamp}.png"
        save_path = os.path.join(plots_dir, filename)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved plot to %s", save_path)

    plt.show()
    return {"panel": panel, "panel_sel": panel_sel, "save_path": save_path}
