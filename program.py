import logging
import pandas as pd
import time
import argparse
from results import ResultSaver
from run_helpers import run_experiment
from cgm_method.configs import CGMInitConfig, CGMFitConfig, CGMPredictConfig, DataConfig

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
        choices=['cgm', 'two-step','comparison']
    )
    return parser.parse_args()

def main():
    args = parse_args()
    choice = args.choice
    summary_rows = []
    samples_cgm, samples_two_step = None, None

    # === Parameter Definitions ===
    cgm_params = {
        "data_cfg": DataConfig(
            split_point=0.99,
            standardize=True
        ),
        "cgm_init": CGMInitConfig(
            dim_latent=50,
            n_samples_train=100,
            emb_size=2
        ),
        "train_cfg": CGMFitConfig(
            n_epochs=10,
            batch_size=256,
            train_freq=20,
            train_window_size=20,
            learningrate=0.01,  # or "decay"
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            sample_weight=None
        ),
        "pred_cfg": CGMPredictConfig(
            n_samples=100,
            verbose=0
        ),
    }

    two_step_params = {
        "split_point": 0.99,
        "fixed_uv_window": 0.005,
        "uv_train_freq": 20,
        "copula_window_size": 0.005,
        "uv_method": "ARMAGARCH",
        "copula_type": "Gaussian",
        "n_samples_per_day": 100,
        "n_samples": 1000
    }

    # === Run Experiments ===
    start_time = time.time()

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
        samples_two_step, results_two_step = run_experiment("2step", two_step_params)
        if results_two_step:
            summary_rows.append({
                "Model": "Two-Step",
                **two_step_params,
                "Time (s)": round(time.time() - start_time, 2),
                **results_two_step
            })

    # === Save Results ===
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        saver = ResultSaver(choice, cgm_params, two_step_params)
        saver.save(samples_cgm, samples_two_step, df_summary)
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
