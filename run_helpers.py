from cgm.cgm import CGMModel
from copula_method.two_step_model import TwoStepModel


def run_cgm_experiment(
    split_point=0.8,
    window_size=7,
    loss_type='ES',
    n_epochs=10,
    batch_size=256,
    n_samples=100,
    fit_model=True,
    sample_model=True,
    evaluate=True
):
    model = CGMModel(split_point=split_point, window_size=window_size, loss_type=loss_type)

    if fit_model:
        model.fit(n_epochs=n_epochs, batch_size=batch_size)

    samples = None
    if sample_model:
        samples = model.sample(n_samples=n_samples)
        print(f"Generated CGM samples shape: {samples.shape}")

    results = None
    if evaluate and samples is not None:
        results = model.evaluate(samples)
        """try:
            results = model.evaluate(samples)
            print("CGM Evaluation Results:")
            for k, v in results.items():
                print(f"{k}: {v:.4f}")
        except Exception as e:
            print("CGM Evaluation skipped:", e)"""

    return samples, results


# === Two-Step Experiment Helper ===
def run_two_step_experiment(
    split_point=0.99,
    uv_train_freq=1,
    copula_window_size=0.05,
    uv_method="ARMAGARCH",
    copula_type="Gaussian",
    fit_model=True,
    sample_model=True,
    evaluate=True
):
    model = TwoStepModel(split_point=split_point,
                         uv_train_freq=uv_train_freq, copula_window_size=copula_window_size,
                         univariate_type=uv_method, copula_type=copula_type)

    if fit_model:
        model.fit(n_samples_per_day=100)

    samples = None
    if sample_model:
        samples = model.sample(n_samples=100)
        print("Sampled Two-Step forecast shape:", samples.shape)

    results = None
    if evaluate and samples is not None:
        try:
            results = model.evaluate(samples)
            print("Two-Step Evaluation Results:")
            for k, v in results.items():
                print(f"{k}: {v:.4f}")
        except Exception as e:
            print("Two-Step Evaluation skipped:", e)

    return samples, results