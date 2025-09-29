# compare_models_plots.py

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
# Same figure geometry for ALL plots, so plotting areas match exactly.
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
    return MODEL_COLORS.get(name, "black")

def _label_for(name: str):
    return DISPLAY_NAME.get(name, name)

def _wins_clip(arr: np.ndarray, pct: tuple[float, float] | None) -> np.ndarray:
    """Winsorize/clamp values to percentiles if requested."""
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
    Compute global min/max per metric across all models after optional winsorization.
    Adds a small padding so lines/boxes don't sit on the frame.
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

        # symmetric padding proportional to range
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
    # Robust display options
    show_fliers: bool = False,
    whis: tuple[float, float] = (5, 95),
    winsorize_pct: tuple[float, float] | None = (1, 99),
    use_symlog_for_dss: bool = False,
    symlog_linthresh: float = 1.0,
):
    """
    One figure per metric (ES, VS, DSS) with side-by-side model boxplots.
    Uses GLOBAL y-lims shared with line plots and fixed canvas/margins.
    """
    if not evaluators:
        raise ValueError("No evaluators provided.")

    metrics = list(METRIC_LABELS.keys())
    outdir = Path(save_dir) if save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    # compute global y-lims once (after winsorization)
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

        # Figure with fixed geometry
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

        # color boxes & style
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

        # Save with fixed canvas (no tight)
        if outdir:
            for ext in ("png", "pdf"):
                fig.savefig(outdir / f"{key}_boxplots.{ext}", dpi=200, bbox_inches=None)
            logger.info(f"Saved {label} boxplots to {outdir}")

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
    One line plot per metric (ES, VS, DSS) with all models overlaid.
    Uses GLOBAL y-lims shared with boxplots and fixed canvas/margins.
    """
    if not evaluators:
        raise ValueError("No evaluators provided.")

    metrics = list(METRIC_LABELS.keys())
    outdir = Path(save_dir) if save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    # compute global y-lims once (after winsorization)
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

            # daily line
            ax.plot(df.index, y, linewidth=1.2, alpha=0.75, color=color,
                    label=f"{label} — daily")

            # moving average
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

        # de-dup legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = {l: h for h, l in zip(handles, labels)}
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=False, ncol=2)

        if outdir:
            for ext in ("png", "pdf"):
                fig.savefig(outdir / f"{key}_multi.{ext}", dpi=200, bbox_inches=None)
            logger.info(f"Saved {metric_label} line plots to {outdir}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        figs[key] = fig

    return figs

# =========================
# Driver
# =========================
if __name__ == "__main__":
    # 1) Data handlers
    ts_dh = DataHandler(split_point=datetime(2019, 1, 4))
    cgm_dh = DataHandler(0.9)

    ts_test_set = ts_dh.get_data(exclude_pandemic=True, filter_duplicates=True)["test_set"]
    cgm_test_set = cgm_dh.get_data(exclude_pandemic=True, filter_duplicates=True)["test_set"]

    # 2) Load samples
    files = {
        "TS Gaussian": "results/TWOSTEP/20250919-133921/samples_two_step.npy",
        "TS Student":  "results/TWOSTEP/20250919-135018/samples_two_step.npy",
        "TS Skewed":   "results/TWOSTEP/20250919-135921/samples_two_step.npy",
        "CGM":         "results/CGM/20250919-142103/samples_cgm.npy",
    }

    # 3) Build evaluators
    evaluators = {}
    for name, path in files.items():
        samples = np.load(path)
        if name.startswith("TS"):
            ev = ForecastEvaluator(test_set=ts_test_set, samples=samples)
        else:  # CGM
            ev = ForecastEvaluator(test_set=cgm_test_set, samples=samples)
        ev.evaluate(p=0.5)
        evaluators[name] = ev

    # 4) Plots (global y-lims + identical canvas; saves PNG + PDF)
    outdir = "results/compare_models"
    plot_metrics_across_models(
        evaluators,
        rolling=14,
        save_dir=outdir,
    )
    plot_metric_boxplots_across_models(
        evaluators,
        save_dir=outdir,
    )
