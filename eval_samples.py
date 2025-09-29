# compare_models_plots.py
"""
Visualization utilities to compare probabilistic forecast models
(CGM vs. two-step copula variants). Produces temporal line plots
and side-by-side boxplots for ES, VS, and DSS.

Run as a script to load samples, build evaluators, and save figures
under results/compare_models/.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.data_handling import DataHandler
from evaluator import ForecastEvaluator  # plotting-free

logger = logging.getLogger(__name__)

# =========================
# Global display config
# =========================
# Same figure geometry for all plots.
FIGSIZE = (12, 5.5)
MARGINS = dict(left=0.10, right=0.98, top=0.90, bottom=0.18)

# Metric labels
METRIC_LABELS = {
    "es":  "Energy Score (ES)",
    "vs":  "Variogram Score (VS)",
    "dss": "Dawid–Sebastiani Score (DSS)",
}

# Colors and pretty names per model
MODEL_COLORS = {
    "CGM":         plt.cm.tab10(0),  # blue
    "TS Gaussian": plt.cm.tab10(1),  # orange
    "TS Student":  plt.cm.tab10(2),  # green
    "TS Skewed":   plt.cm.tab10(3),  # red
}
DISPLAY_NAME = {
    "CGM":         "CGM",
    "TS Gaussian": "Gaussian Copula",
    "TS Student":  "Student-t Copula",
    "TS Skewed":   "Skewed-t Copula",
}


def _color_for(name: str):
    """Return a consistent color for the given model name."""
    return MODEL_COLORS.get(name, "black")


def _label_for(name: str):
    """Return a display label for the given model name."""
    return DISPLAY_NAME.get(name, name)


def _wins_clip(arr: np.ndarray, pct: tuple[float, float] | None) -> np.ndarray:
    """
    Winsorize an array to the given (low, high) percentiles.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    pct : tuple[float, float] | None
        (low, high) percentiles. If None, no clipping.

    Returns
    -------
    np.ndarray
        Clipped array (same shape).
    """
    if pct is None or arr.size == 0:
        return arr
    lo, hi = np.nanpercentile(arr, pct)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return arr
    return np.clip(arr, lo, hi)


def _compute_global_ylims(
    evaluators: dict,
    metrics: list[str],
    winsorize_pct: tuple[float, float] | None,
    pad_frac: float = 0.02,
) -> dict[str, tuple[float, float]]:
    """
    Compute global (min, max) per metric across models after optional winsorization.

    Adds symmetric padding so lines/boxes do not touch the frame.

    Parameters
    ----------
    evaluators : dict[str, ForecastEvaluator]
        Mapping model name -> evaluator object.
    metrics : list[str]
        Metric keys (e.g., ['es','vs','dss']).
    winsorize_pct : tuple[float, float] | None
        Percentiles for clipping, or None to disable.
    pad_frac : float, default=0.02
        Padding fraction of the observed range.

    Returns
    -------
    dict[str, tuple[float, float]]
        Metric -> (ymin, ymax).
    """
    global_ylims: dict[str, tuple[float, float]] = {}
    for key in metrics:
        stacks = []
        for ev in evaluators.values():
            y = ev.get_daily_scores()[key].dropna().values.astype(float)
            y = _wins_clip(y, winsorize_pct)
            if y.size:
                stacks.append(y)
        if not stacks:
            global_ylims[key] = (0.0, 1.0)
            continue

        stack = np.concatenate(stacks)
        lo, hi = float(np.nanmin(stack)), float(np.nanmax(stack))
        if not np.isfinite(lo) or not np.isfinite(hi):
            global_ylims[key] = (0.0, 1.0)
            continue

        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0  # degenerate safeguard

        rng = hi - lo
        lo_p = lo - pad_frac * rng
        hi_p = hi + pad_frac * rng
        global_ylims[key] = (lo_p, hi_p)
    return global_ylims


# =========================
# Plotting functions
# =========================
def plot_metric_boxplots_across_models(
    evaluators: dict,
    save_dir: str | None = None,
    show: bool = True,
    # display options
    show_fliers: bool = False,
    whis: tuple[float, float] = (5, 95),
    winsorize_pct: tuple[float, float] | None = (1, 99),
    use_symlog_for_dss: bool = False,
    symlog_linthresh: float = 1.0,
):
    """
    Draw one figure per metric (ES, VS, DSS) with side-by-side model boxplots.

    Uses global y-limits (shared with the line plots) and a fixed canvas.

    Parameters
    ----------
    evaluators : dict[str, ForecastEvaluator]
        Mapping model name -> evaluator (must provide get_daily_scores()).
    save_dir : str | None, default=None
        Output directory; if None, figures are not saved.
    show : bool, default=True
        If True, display figures; otherwise close them.
    show_fliers : bool, default=False
        Whether to draw outlier fliers.
    whis : tuple[float, float], default=(5, 95)
        Whisker percentiles.
    winsorize_pct : tuple[float, float] | None, default=(1, 99)
        Percentiles used to clip the series before plotting.
    use_symlog_for_dss : bool, default=False
        If True, apply symlog scale to DSS axis.
    symlog_linthresh : float, default=1.0
        Linear threshold for symlog scaling.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Metric key -> figure.
    """
    if not evaluators:
        raise ValueError("No evaluators provided.")

    metrics = list(METRIC_LABELS.keys())
    outdir = Path(save_dir) if save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    global_ylims = _compute_global_ylims(evaluators, metrics, winsorize_pct)

    figs = {}
    for key in metrics:
        label = METRIC_LABELS[key]

        # Gather per-model series
        raw_data, names = [], []
        for model_name, ev in evaluators.items():
            s = ev.get_daily_scores()[key].dropna().values.astype(float)
            s = _wins_clip(s, winsorize_pct)
            raw_data.append(s)
            names.append(model_name)

        # Fixed geometry
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.set_constrained_layout(False)
        fig.subplots_adjust(**MARGINS)

        bp = ax.boxplot(
            raw_data,
            tick_labels=[_label_for(n) for n in names],
            patch_artist=True,
            showfliers=show_fliers,
            whis=whis,
            flierprops=dict(marker="o", markersize=3, alpha=0.35, color="gray"),
        )

        # Color/style
        for patch, name in zip(bp["boxes"], names):
            patch.set_facecolor(_color_for(name))
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.8)
        for whisk in bp["whiskers"]:
            whisk.set_color("black")
        for cap in bp["caps"]:
            cap.set_color("black")

        ax.set_title(f"{label} — Boxplots")
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(*global_ylims[key])

        if use_symlog_for_dss and key == "dss":
            ax.set_yscale("symlog", linthresh=symlog_linthresh)

        # Save with fixed canvas
        if outdir:
            for ext in ("png", "pdf"):
                fig.savefig(outdir / f"{key}_boxplots.{ext}", dpi=200, bbox_inches=None)
            logger.info("Saved %s boxplots to %s", label, outdir)

        if show:
            plt.show()
        else:
            plt.close(fig)

        figs[key] = fig

    return figs


def plot_metrics_across_models(
    evaluators: dict,
    rolling: int | None = 14,
    save_dir: str | None = None,
    show: bool = True,
    winsorize_pct: tuple[float, float] | None = (1, 99),
    use_symlog_for_dss: bool = False,
    symlog_linthresh: float = 1.0,
):
    """
    Draw one line plot per metric (ES, VS, DSS) with all models overlaid.

    Uses global y-limits (shared with boxplots) and a fixed canvas.

    Parameters
    ----------
    evaluators : dict[str, ForecastEvaluator]
        Mapping model name -> evaluator (must provide get_daily_scores()).
    rolling : int | None, default=14
        Window for simple moving average overlay; None disables.
    save_dir : str | None, default=None
        Output directory; if None, figures are not saved.
    show : bool, default=True
        If True, display figures; otherwise close them.
    winsorize_pct : tuple[float, float] | None, default=(1, 99)
        Percentiles used to clip the series before plotting.
    use_symlog_for_dss : bool, default=False
        If True, apply symlog scale to DSS axis.
    symlog_linthresh : float, default=1.0
        Linear threshold for symlog scaling.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Metric key -> figure.
    """
    if not evaluators:
        raise ValueError("No evaluators provided.")

    metrics = list(METRIC_LABELS.keys())
    outdir = Path(save_dir) if save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    global_ylims = _compute_global_ylims(evaluators, metrics, winsorize_pct)

    figs = {}
    for key in metrics:
        metric_label = METRIC_LABELS[key]

        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.set_constrained_layout(False)
        fig.subplots_adjust(**MARGINS)

        for model_name, ev in evaluators.items():
            df = ev.get_daily_scores().sort_values("date").set_index("date")
            y = df[key].astype(float).values
            y = _wins_clip(y, winsorize_pct)

            color = _color_for(model_name)
            label = _label_for(model_name)

            # Daily line
            ax.plot(df.index, y, linewidth=1.2, alpha=0.75, color=color,
                    label=f"{label} — daily")

            # Moving average
            if rolling and rolling > 1:
                roll = pd.Series(y, index=df.index).rolling(
                    rolling, min_periods=max(1, rolling // 3)
                ).mean()
                ax.plot(df.index, roll, linestyle=":", linewidth=2.2, color=color, alpha=1.0,
                        label=f"{label} — MA{rolling}")

        ax.set_ylabel(metric_label)
        ax.set_xlabel("Date")
        ax.set_title(f"{metric_label} — Temporal Comparison")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(*global_ylims[key])

        if use_symlog_for_dss and key == "dss":
            ax.set_yscale("symlog", linthresh=symlog_linthresh)

        # De-duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        uniq = {l: h for h, l in zip(handles, labels)}
        #ax.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=False, ncol=len(uniq))

        if outdir:
            for ext in ("png", "pdf"):
                fig.savefig(outdir / f"{key}_multi.{ext}", dpi=200, bbox_inches=None)
            logger.info("Saved %s line plots to %s", metric_label, outdir)

        if show:
            plt.show()
        else:
            plt.close(fig)

        figs[key] = fig

    return figs

def export_legend(
    save_dir: str,
    models: list[str],
    ncol: int = 1,
    ext=("png", "pdf"),
    rolling: int = 14,  # label the MA line
):
    """
    Export legend as its own standalone figure (2 rows: daily + MA).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pathlib import Path

    outdir = Path(save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    handles, labels = [], []
    for m in models:
        c = _color_for(m)
        handles.append(Line2D([], [], color=c, lw=1.2, alpha=0.75))
        labels.append(f"{_label_for(m)} — daily")
        handles.append(Line2D([], [], color=c, lw=2.2, ls=":", alpha=1.0))
        labels.append(f"{_label_for(m)} — MA{rolling}")

    # force 2 rows (daily row + MA row)
    ncol = len(models)

    fig, ax = plt.subplots(figsize=(7.5, 2.0))
    ax.axis("off")
    leg = ax.legend(handles, labels, loc="center", frameon=False, ncol=ncol)

    # tight crop around legend
    for e in ext:
        fig.savefig(
            outdir / f"legend_scores.{e}",
            dpi=200,
            bbox_extra_artists=(leg,),
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=True,
        )
    plt.close(fig)


if __name__ == "__main__":
    # Data handlers with different split conventions
    ts_dh = DataHandler(split_point=datetime(2019, 1, 4))
    cgm_dh = DataHandler(0.9)

    ts_test_set = ts_dh.get_data(exclude_pandemic=True, filter_duplicates=True)["test_set"]
    cgm_test_set = cgm_dh.get_data(exclude_pandemic=True, filter_duplicates=True)["test_set"]

    # Sample files (update paths as needed)
    files = {
        "TS Gaussian": "results/TWOSTEP/20250919-133921/samples_two_step.npy",
        "TS Student":  "results/TWOSTEP/20250919-135018/samples_two_step.npy",
        "TS Skewed":   "results/TWOSTEP/20250919-135921/samples_two_step.npy",
        "CGM":         "results/CGM/20250919-142103/samples_cgm.npy",
    }

    # Evaluators per model
    evaluators = {}
    for name, path in files.items():
        samples = np.load(path)
        if name.startswith("TS"):
            ev = ForecastEvaluator(test_set=ts_test_set, samples=samples)
        else:
            ev = ForecastEvaluator(test_set=cgm_test_set, samples=samples)
        ev.evaluate(p=0.5)  # populate daily scores and aggregates
        evaluators[name] = ev

    # Plots (global y-lims + identical canvas; saves PNG + PDF)
    outdir = "results/compare_models"
    plot_metrics_across_models(evaluators, rolling=14, save_dir=outdir)
    plot_metric_boxplots_across_models(evaluators, save_dir=outdir)
    export_legend(outdir, list(evaluators.keys()), ncol=2)
