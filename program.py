"""import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print("GPUs visible to TF:", gpus)
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)"""

import logging
import pandas as pd
import time
import argparse

from datetime import datetime
from data.data_handling import DataHandler
from results import ResultSaver
from run_helpers import run_experiment
from cgm_method.configs import CGMInitConfig, CGMFitConfig, CGMSampleConfig, CGMDataConfig
from copula_method.configs import TSInitConfig, TSFitConfig, TSSampleConfig, TSDataConfig

# pip install .r "/mnt/c/Users/Anwender/PycharmProjects/BachelorThesis/requirements.txt"

# === Logging Setup ===
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('model.log', mode='w'),
        logging.StreamHandler()
    ]
)
logging.getLogger('rpy2').setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'choice',
        choices=['cgm', '2step','comparison', 'data_only']
    )
    return parser.parse_args()

def main():
    args = parse_args()
    choice = args.choice
    metrics_rows = []  # ONLY metrics per model
    samples = {"samples_cgm": None, "samples_two_step": None}

    "Only relevant if data_only is chosen"
    data_handler_params = {
        "split_point": 0.95,
    }

    # === Parameter Definitions ===
    cgm_params = {
        "data_cfg": CGMDataConfig(
            split_point=0.9,
            filter_features=True,
            exclude_pandemic=True
        ),
        "cgm_init": CGMInitConfig(
            dim_latent=256,
            n_samples_train=100,
            emb_size=4
        ),
        "train_cfg": CGMFitConfig(
            n_epochs=100,
            batch_size=512,
            train_freq=30,
            train_window_size=50,
            learningrate=0.00005,  # or "decay"
            verbose=1,
            callbacks=None,
            validation_split=0.1,
            validation_data=None,
            sample_weight=None
        ),
        "pred_cfg": CGMSampleConfig(
            n_samples=1000,
            verbose=0
        ),
    }

    ts_params = {
        "data_config": TSDataConfig(
            split_point=datetime(2019, 1, 4),
        ),
        "init_config": TSInitConfig(
            univariate_type="ARMAGARCH",
            copula_type="skewed-t",
            copula_params={"df": 5},
            rolling_window_size=1,
            copula_refit_freq=30,
            uv_fit_percentage=0.2,
            uv_refit_freq=7
        ),
        "fit_config": TSFitConfig(
            arma_order=(1, 1),
            include_mean=True,
            arma_maxiter=600,
            on_nonconverge="drop_ma",
            variance_model="sGARCH",
            garch_order=(1, 1),
            dist="norm",
            garch_scale="auto",
            garch_target_std=10.0,
            suppress_convergence_warnings=True
        ),
        "sample_config": TSSampleConfig(
            n_samples=1000,
            n_samples_uv=1000,
        ),
    }

    if choice == "":
        return

    start_time = time.time()

    if choice in {"data_only"}:
        logging.info("Running Data Preparation")
        # Use a date split for clarity
        split_point = pd.Timestamp("2018-01-01")

        handler = DataHandler(split_point=split_point)

        data_dict = handler.get_data(
            exclude_pandemic=True,
            target_only=False,
            filter_duplicates=True,
            save_df=True,
            df_path="data/full_data.csv",
            save_png=False,
        )

        df = data_dict["full_data"]

        print("\n=== Dataset Shape ===")
        print(df.shape)

        print("\n=== Columns ===")
        print(df.columns.tolist())

        print("\n=== Head ===")
        print(df.head())

        print("\n=== Train/Test sizes ===")
        print(len(data_dict["train_set"]), len(data_dict["test_set"]))


    if choice in {"cgm", "comparison"}:
        logging.info("Running CGM Experiment")
        samples["samples_cgm"], results_cgm = run_experiment("cgm_method", cgm_params)
        if results_cgm:
            # ONLY metrics here (params are saved separately by ResultSaver)
            metrics_rows.append({
                "Model": "CGM",
                "runtime_s": round(time.time() - start_time, 2),
                **results_cgm
            })

    if choice in {"2step", "comparison"}:
        logging.info("Running Two-Step Experiment")
        samples["samples_two_step"], results_two_step = run_experiment("2step", ts_params)
        if results_two_step:
            metrics_rows.append({
                "Model": "Two-Step",
                "runtime_s": round(time.time() - start_time, 2),
                **results_two_step
            })

    # === Save Results ===
    if metrics_rows:
        saver = ResultSaver(choice, cgm_params, ts_params)
        saver.save(
            samples=samples,
            metrics_rows=metrics_rows,
            params={"cgm": cgm_params if choice in {"cgm", "comparison"} else None,
                    "two_step": ts_params if choice in {"2step", "comparison"} else None}
        )
        # Optional: also print a quick view of metrics to console
        print(pd.DataFrame(metrics_rows))
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
