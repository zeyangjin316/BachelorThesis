import logging
import matplotlib.pyplot as plt
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

# === Main Entry Point ===
def main():
    # Run CGM
    print("\n=== Running CGM Experiment ===")
    samples_cgm, results_cgm = run_cgm_experiment(
        split_point=0.9,
        n_epochs=5,
        batch_size=100,
        n_samples=10,
        fit_model=True,
        sample_model=False,
        evaluate=False
    )

    # Run Two-Step
    print("\n=== Running Two-Step Experiment ===")
    samples_two_step, results_two_step = run_two_step_experiment(
        split_point=0.99,
        uv_method="ARMAGARCH",
        copula_type="Gaussian",
        fit_model=False,
        sample_model=False,
        evaluate=False
    )

    print("\n=== Summary ===")

    if results_cgm:
        print("CGM Evaluation Metrics:")
        for k, v in results_cgm.items():
            print(f"  {k}: {v:.4f}")

    if results_two_step:
        print("Two-Step Evaluation Metrics:")
        for k, v in results_two_step.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
