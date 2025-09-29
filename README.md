# Generative Machine Learning Methods for Multivariate Excess Return Forecasting

Code for my Bachelor thesis comparing two approaches to **probabilistic daily equity excess-return forecasting**:

- **CGM** — Conditional Generative Model (deep generative sampler).
- **Two-Step Copula** — ARMA/GARCH marginals + copula (Gaussian / Student-t / Skewed-t).

Both are run in rolling windows and evaluated with proper scoring rules (ES, VS, DSS; per-asset CRPS).

---

## Quick Start

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Configure your data paths in `config.py`:
- **Required:** `data_for_kit.csv` with `date`, `sym_root`, `permno`, `ret_crsp`.
- **Optional:** `LTV_History.csv`, `VIX_History.csv`, `returns_5m.csv`.

---

## Main Run Types (program.py)

```bash
python program.py cgm         # train + sample + evaluate CGM
python program.py 2step       # train + sample + evaluate Two-Step
python program.py comparison  # run both pipelines and compare
python program.py data_only   # load/split data and print basic info
```

Outputs (metrics, params, samples) are saved under `results/<...>/<timestamp>/`.

---

## Standalone Entrypoints

### 1) Evaluate precomputed samples (“eval_samples” workflow)
```python
import numpy as np, pandas as pd
from evaluator import ForecastEvaluator

test_set = pd.read_csv("path/to/test_set.csv")   # or build via DataHandler
samples  = np.load("path/to/samples.npy")        # shape (T, S, N)

ev = ForecastEvaluator(test_set, samples, asset_order=None)  # pass explicit order if needed
summary = ev.evaluate(p=0.5)      # dict: mean_es/mean_vs/mean_dss + per-asset CRPS
daily   = ev.get_daily_scores()   # DataFrame: ['date','es','vs','dss']
```

### 2) Plotting (models vs realized, per-symbol and by-sector)
```python
from forecast_plotter import ForecastPlotter
from data.data_handling import DataHandler

plotter = ForecastPlotter(DataHandler(split_point=0.9))

# Per-symbol comparison (one figure per symbol)
plotter.plot_models_per_symbol(
    model_paths={"CGM":".../samples_cgm.npy", "Skewed-t Copula":".../samples_two_step.npy"},
    symbols_order=[...], symbols_to_plot=[...], sample_to_plot=0, save_dir="results/plots"
)

# Grouped by sector (uniform grid across figures)
plotter.plot_grouped_by_sector(
    model_paths={...}, symbols_order=[...],
    symbol_to_company={...}, symbol_to_sector={...}, save_dir="results/plots/sectors"
)
```
