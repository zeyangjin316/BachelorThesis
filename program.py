import logging
import pandas as pd
import time
from results import ResultSaver
from run_helpers import run_cgm_experiment, run_two_step_experiment

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


def main():
    summary_rows = []
    samples_cgm, samples_two_step = None, None

    # === User Selection ===
    choice = input("Run CGM, Two-Step model, or comparison? Enter 'cgm', '2step', or 'comparison': ").strip().lower()
    if choice not in {"cgm", "2step", "comparison"}:
        print("Invalid choice")
        return

    # === Parameter Definitions ===
    cgm_params = {
        "split_point": 0.99,
        "train_freq": 20,
        "train_window_size": 20,
        "loss_type": "ES",
        "n_epochs": 10,
        "batch_size": 256,
        "n_samples": 100
    }

    two_step_params = {
        "split_point": 0.99,
        "uv_train_freq": 20,
        "copula_window_size": 0.005,
        "uv_method": "ARMAGARCH",
        "copula_type": "Gaussian"
    }

    # === Run Experiments ===
    start_time = time.time()

    if choice in {"cgm", "comparison"}:
        logging.info("Running CGM Experiment")
        samples_cgm, results_cgm = run_cgm_experiment(
            **cgm_params,
            fit_model=True,
            sample_model=True,
            evaluate=True
        )
        if results_cgm:
            summary_rows.append({
                "Model": "CGM",
                "Split Point": cgm_params["split_point"],
                "Train Freq": cgm_params["train_freq"],
                "Window Size": cgm_params["train_window_size"],
                "Loss": cgm_params["loss_type"],
                "Epochs": cgm_params["n_epochs"],
                "Batch Size": cgm_params["batch_size"],
                "Samples": cgm_params["n_samples"],
                "Time (s)": round(time.time() - start_time, 2),
                **results_cgm
            })

    if choice in {"2step", "comparison"}:
        logging.info("Running Two-Step Experiment")
        samples_two_step, results_two_step = run_two_step_experiment(
            **two_step_params,
            fit_model=True,
            sample_model=True,
            evaluate=True
        )
        if results_two_step:
            summary_rows.append({
                "Model": "Two-Step",
                "Split Point": two_step_params["split_point"],
                "UV Train Freq": two_step_params["uv_train_freq"],
                "Copula Window": two_step_params["copula_window_size"],
                "UV Method": two_step_params["uv_method"],
                "Copula": two_step_params["copula_type"],
                "Epochs": "-",
                "Batch Size": "-",
                "Samples": "-",
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
