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
    results_cgm, results_two_step = None, None

    # === User Selection ===
    choice = input("Run CGM, Two-Step model, or both? Enter 'cgm', '2step', or 'both': ").strip().lower()

    if choice in {"cgm", "both"}:
        print("\n=== Running CGM Experiment ===")
        samples_cgm, results_cgm = run_cgm_experiment(
            split_point=0.8,
            window_size=7,
            loss_type='ES',
            n_epochs=10,
            batch_size=256,
            n_samples=100,
            fit_model=True,
            sample_model=True,
            evaluate=True
        )

    if choice in {"2step", "both"}:
        print("\n=== Running Two-Step Experiment ===")
        samples_two_step, results_two_step = run_two_step_experiment(
            split_point=0.99,
            uv_train_freq=7,
            copula_train_freq=1,
            uv_method="ARMAGARCH",
            copula_type="Gaussian",
            fit_model=True,
            sample_model=True,
            evaluate=True
        )

    if choice not in {"cgm", "2step", "both"}:
        print("Invalid choice")
        return

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
