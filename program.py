"""import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print("GPUs visible to TF:", gpus)
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)"""

import logging
import pandas as pd
import time
import argparse
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
        choices=['cgm', '2step','comparison', '']
    )
    return parser.parse_args()

def main():
    args = parse_args()
    choice = args.choice
    summary_rows = []
    samples_cgm, samples_two_step = None, None

    # === Parameter Definitions ===
    cgm_params = {
        "data_cfg": CGMDataConfig(
            split_point=0.8,
            standardize=True
        ),
        "cgm_init": CGMInitConfig(
            dim_latent=50,
            n_samples_train=100,
            emb_size=2
        ),
        "train_cfg": CGMFitConfig(
            n_epochs=100,
            batch_size=256,
            train_freq=30,
            train_window_size=20,
            learningrate=0.001,  # or "decay"
            verbose=1,
            callbacks=None,
            validation_split=0.0,
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
            split_point=0.8,
        ),
        "init_config": TSInitConfig(
            univariate_type="ARMAGARCH",
            copula_type="Gaussian",
            rolling_window_size=1,
            copula_refit_freq=30,
            uv_fit_percentage=0.2,
            uv_refit_freq=5
        ),
        "fit_config": TSFitConfig(
            arma_order=(1, 1),
            include_mean=True,
            arma_maxiter=600,  # more iterations for first optimizer
            on_nonconverge="drop_ma",  # or "drop_ar" or "warn"
            variance_model="sGARCH",  # or "gjrGARCH", "eGARCH"
            garch_order=(1, 1),
            dist="norm",
            garch_scale="auto",  # or fixed, e.g. 100.0
            garch_target_std=10.0,
            suppress_convergence_warnings=True
        ),
        "sample_config": TSSampleConfig(
            n_samples=1000,
            n_samples_uv=1000,
        ),
    }
    # === Run Experiments ===
    start_time = time.time()

    if args.choice == "":
        return
    if args.choice in {"cgm", "comparison"}:
        logging.info("Running CGM Experiment")
        samples_cgm, results_cgm = run_experiment("cgm_method", cgm_params)
        if results_cgm:
            summary_rows.append({
                "Model": "CGM",
                **cgm_params,
                "Time (s)": round(time.time() - start_time, 2),
                **results_cgm
            })

    if args.choice in {"2step", "comparison"}:
        logging.info("Running Two-Step Experiment")
        samples_two_step, results_two_step = run_experiment("2step", ts_params)
        if results_two_step:
            summary_rows.append({
                "Model": "Two-Step",
                **ts_params,
                "Time (s)": round(time.time() - start_time, 2),
                **results_two_step
            })

    # === Save Results ===
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        """saver = ResultSaver(choice, cgm_params, ts_params)
        saver.save(samples_cgm, samples_two_step, df_summary)"""
        print(df_summary)
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
