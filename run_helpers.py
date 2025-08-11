from cgm_method.cgm_experiment import CGMModel
from copula_method.two_step_model import TwoStepModel

def run_experiment(model_type, params, fit_model=True, sample_model=True, evaluate=True):
    if model_type == "cgm_method":
        model = CGMModel(**params)

    elif model_type == "2step":
        model = TwoStepModel(
            split_point=params["split_point"],
            fixed_uv_window=params["fixed_uv_window"],
            uv_train_freq=params["uv_train_freq"],
            copula_window_size=params["copula_window_size"],
            univariate_type=params["uv_method"],
            copula_type=params["copula_type"],
            n_samples_per_day=params["n_samples_per_day"],
            n_samples=params["n_samples"]
        )

    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    if fit_model:
        model.fit()

    samples = None
    if sample_model:
        samples = model.sample()
        print(f"Generated {model_type} samples shape: {samples.shape}")

    results = None
    if evaluate and samples is not None:
        try:
            results = model.evaluate(samples)
        except Exception as e:
            print(f"{model_type} Evaluation skipped:", e)

    return samples, results